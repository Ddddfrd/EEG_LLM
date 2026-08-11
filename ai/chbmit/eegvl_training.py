"""Reusable training loop for EEG-VL experiments."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.v2.metrics import evaluate, find_optimal_threshold_exact

from .eegvl_s1_data import S1ImageDataset


EEGVL_TRAINING_VERSION = "eegvl_training_v1"


@dataclass(frozen=True)
class EegvlTrainingConfig:
    seed: int = 42
    max_epochs: int = 20
    patience: int = 4
    micro_batch_size: int = 64
    effective_batch_size: int = 256
    prediction_batch_size: int = 128
    new_layer_learning_rate: float = 5e-4
    encoder_learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    logit_adjustment_tau: float = 1.0
    minimum_calibration_recall: float = 0.80
    num_workers: int = 0

    def validate(self) -> None:
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Epoch and patience limits must be positive")
        if self.micro_batch_size < 1 or self.effective_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.prediction_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Prediction batch size/workers are invalid")
        if (
            self.new_layer_learning_rate <= 0
            or self.encoder_learning_rate <= 0
            or self.minimum_learning_rate < 0
            or self.weight_decay < 0
        ):
            raise ValueError("Optimizer settings are invalid")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.logit_adjustment_tau < 0:
            raise ValueError("logit_adjustment_tau must be non-negative")
        if not 0 <= self.minimum_calibration_recall <= 1:
            raise ValueError("Minimum calibration recall must be in [0, 1]")

    @property
    def accumulation_steps(self) -> int:
        return max(
            1,
            math.ceil(self.effective_batch_size / self.micro_batch_size),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "accumulation_steps": self.accumulation_steps,
            "actual_effective_batch_size": (
                self.micro_batch_size * self.accumulation_steps
            ),
        }


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            parameter = parameters[name]
            shadow.mul_(self.decay).add_(
                parameter.detach(),
                alpha=1.0 - self.decay,
            )

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        parameters = dict(model.named_parameters())
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                parameter = parameters[name]
                backup[name] = parameter.detach().clone()
                parameter.copy_(shadow)
        return backup

    @staticmethod
    def restore(
        model: nn.Module,
        backup: Mapping[str, torch.Tensor],
    ) -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in backup.items():
                parameters[name].copy_(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _autocast_context(
    device: torch.device,
    precision: str,
) -> Any:
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def select_precision(device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def class_priors(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=2).astype(np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(counts == 0):
        raise ValueError("Training labels must contain both classes")
    return counts / counts.sum()


def logit_adjusted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    log_priors: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Binary logits must be shaped (batch, 2)")
    if log_priors.shape != (2,):
        raise ValueError("log_priors must contain two class values")
    adjusted = logits.float() + float(tau) * log_priors[None]
    return nn.functional.cross_entropy(adjusted, labels)


def _optimizer_groups(
    model: nn.Module,
    *,
    config: EegvlTrainingConfig,
) -> list[dict[str, Any]]:
    encoder: list[nn.Parameter] = []
    new_layers: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("backbone.encoder."):
            encoder.append(parameter)
        else:
            new_layers.append(parameter)
    groups: list[dict[str, Any]] = []
    if encoder:
        groups.append({
            "params": encoder,
            "lr": config.encoder_learning_rate,
            "group_name": "encoder",
        })
    if new_layers:
        groups.append({
            "params": new_layers,
            "lr": config.new_layer_learning_rate,
            "group_name": "new_layers",
        })
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return groups


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def schedule(base_lr: float) -> Any:
        minimum_factor = (
            minimum_learning_rate / base_lr if base_lr else 0.0
        )

        def multiplier(step: int) -> float:
            current = min(step + 1, total_steps)
            if current <= warmup_steps:
                return current / warmup_steps
            progress = (
                (current - warmup_steps)
                / max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [schedule(base_lr) for base_lr in base_lrs],
    )


def forward_eeg_batch(
    model: nn.Module,
    images: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    baseline = batch.get("baseline_log_magnitude")
    if baseline is None:
        return model(images)
    if not isinstance(baseline, torch.Tensor):
        raise TypeError("baseline_log_magnitude batch must be a tensor")
    return model(
        images,
        baseline_log_magnitude=baseline.to(
            images.device, non_blocking=True
        ),
    )

@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: S1ImageDataset,
    *,
    device: torch.device,
    batch_size: int,
    precision: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected_precision = precision or select_precision(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with _autocast_context(device, selected_precision):
            logits = forward_eeg_batch(model, images, batch)
        probabilities.append(
            torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        )
        labels.append(np.asarray(batch["label"], dtype=np.int64))
    return (
        np.concatenate(probabilities).astype(np.float32, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_s1_model(
    model: nn.Module,
    training_dataset: S1ImageDataset,
    calibration_dataset: S1ImageDataset,
    *,
    device: torch.device,
    config: EegvlTrainingConfig,
    selection_metric: str = "auprc",
    sample_duration_seconds: float = 2.0,
) -> dict[str, Any]:
    config.validate()
    if selection_metric not in {"auprc", "mean_auroc_auprc"}:
        raise ValueError(f"Unknown S1 selection metric: {selection_metric}")
    if sample_duration_seconds <= 0:
        raise ValueError("Sample duration must be positive")
    if training_dataset.augmentation is None:
        augmentation_status = "disabled"
    else:
        augmentation_status = training_dataset.augmentation.recipe_id
    if calibration_dataset.augmentation is not None:
        raise ValueError("Calibration dataset must not use augmentation")
    seed_everything(config.seed)
    model.to(device)
    precision = select_precision(device)
    priors = class_priors(np.asarray(
        training_dataset.source_cache.labels[training_dataset.indices],
        dtype=np.int64,
    ))
    log_priors = torch.log(
        torch.from_numpy(priors.astype(np.float32)).to(device)
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        training_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, config=config),
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loader) / config.accumulation_steps
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=max(1, updates_per_epoch * config.max_epochs),
        warmup_fraction=config.warmup_fraction,
        minimum_learning_rate=config.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )
    ema = ExponentialMovingAverage(model, decay=config.ema_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_probabilities: np.ndarray | None = None
    best_epoch = 0
    best_threshold = 0.5
    best_selection_score = float("-inf")
    stale_epochs = 0
    optimizer_steps = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.max_epochs + 1):
        training_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_rows = 0
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with _autocast_context(device, precision):
                logits = forward_eeg_batch(model, images, batch)
                loss = logit_adjusted_cross_entropy(
                    logits,
                    labels,
                    log_priors=log_priors,
                    tau=config.logit_adjustment_tau,
                )
            scaler.scale(loss / config.accumulation_steps).backward()
            is_update = (
                batch_index % config.accumulation_steps == 0
                or batch_index == len(loader)
            )
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                optimizer_steps += 1
            rows = int(labels.shape[0])
            epoch_loss += float(loss.detach()) * rows
            epoch_rows += rows

        backup = ema.apply(model)
        probabilities, calibration_labels = predict_dataset(
            model,
            calibration_dataset,
            device=device,
            batch_size=config.prediction_batch_size,
            precision=precision,
        )
        threshold = find_optimal_threshold_exact(
            calibration_labels,
            probabilities,
            min_recall=config.minimum_calibration_recall,
        )
        metrics = evaluate(
            calibration_labels,
            probabilities,
            threshold=threshold,
            print_report=False,
            sample_duration_seconds=sample_duration_seconds,
        )
        epoch_state = _state_to_cpu(model)
        ExponentialMovingAverage.restore(model, backup)
        auprc = metrics["auprc"]
        auroc = metrics["auroc"]
        if auprc is None or auroc is None:
            raise ValueError(
                "Source calibration ranking metrics are not computable"
            )
        selection_score = (
            float(auprc)
            if selection_metric == "auprc"
            else 0.5 * (float(auprc) + float(auroc))
        )
        improved = selection_score > best_selection_score + 1e-12
        history.append({
            "epoch": epoch,
            "training_loss": epoch_loss / max(1, epoch_rows),
            "calibration_auprc": float(auprc),
            "calibration_auroc": float(auroc),
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "calibration_f1": metrics["f1"],
            "calibration_recall": metrics["recall"],
            "threshold": float(threshold),
            "learning_rates": {
                str(group.get("group_name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
            "selected": improved,
        })
        print(
            f"EEG-VL epoch {epoch}/{config.max_epochs}: "
            f"loss={history[-1]['training_loss']:.6f}, "
            f"cal_auprc={float(auprc):.6f}, "
            f"selection={selection_score:.6f}, "
            f"threshold={float(threshold):.6f}, selected={improved}",
            flush=True,
        )
        if improved:
            best_selection_score = selection_score
            best_state = epoch_state
            best_metrics = metrics
            best_probabilities = probabilities.copy()
            best_epoch = epoch
            best_threshold = float(threshold)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if (
        best_state is None
        or best_metrics is None
        or best_probabilities is None
    ):
        raise RuntimeError("Training did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    duration = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "schema_version": EEGVL_TRAINING_VERSION,
        "model": model,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "calibration_metrics": best_metrics,
        "calibration_probabilities": best_probabilities,
        "selection": {
            "metric": selection_metric,
            "score": best_selection_score,
        },
        "history": history,
        "class_priors": {
            "normal": float(priors[0]),
            "ictal": float(priors[1]),
        },
        "logit_adjustment": {
            "formula": "cross_entropy(logits + tau * log(source_train_priors))",
            "tau": config.logit_adjustment_tau,
        },
        "augmentation": augmentation_status,
        "precision": precision,
        "optimizer_steps": optimizer_steps,
        "epochs_completed": len(history),
        "duration_seconds": duration,
        "peak_allocated_gpu_bytes": peak_memory,
        "config": config.to_dict(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_s1_checkpoint(
    training_result: Mapping[str, Any],
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".{model_name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": EEGVL_TRAINING_VERSION,
        "model_name": model_name,
        "model_config": dict(model_config),
        "state_dict": copy.deepcopy(training_result["best_state_dict"]),
        "best_epoch": int(training_result["best_epoch"]),
        "threshold": float(training_result["threshold"]),
        "class_priors": dict(training_result["class_priors"]),
        "logit_adjustment": dict(training_result["logit_adjustment"]),
    }
    torch.save(payload, temporary)
    digest = _file_sha256(temporary)
    destination = output_dir / f"{model_name}_{digest[:12]}.pt"
    if destination.exists():
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return destination, digest
