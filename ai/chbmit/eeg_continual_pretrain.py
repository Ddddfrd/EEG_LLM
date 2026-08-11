"""Reproduce the PDF pretraining protocol for one CHB-MIT patient fold."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.v2.metrics import evaluate, find_optimal_threshold_exact

from .cache import ChbmitWindowCache, build_window_caches
from .deep_timeline import DeepTargetTimeline, build_deep_target_timeline
from .eeg_continual_eval_cache import (
    NaturalFoldImageCache,
    NaturalFoldImageDataset,
    build_natural_fold_image_cache,
)
from .eeg_continual_pretrain_model import (
    SERVER_PRETRAIN_MODEL_VERSION,
    ServerEEGVLPretrainModel,
    ServerSTFTConfig,
    ServerSTFTEfficientNetEncoder,
    load_portable_pretrain_state_dict,
    portable_pretrain_state_dict,
)
from .eegvl_experiment import (
    _save_prediction_artifact,
    runtime_metadata,
)
from .eegvl_m9_model import LoRAConfig
from .eegvl_s1_data import (
    S1ImageDataset,
    S1PreprocessConfig,
    S1PreprocessedCache,
    build_s1_preprocessed_cache,
)
from .eegvl_training import (
    forward_eeg_batch,
    predict_dataset,
    select_precision,
)
from .index import canonical_hash
from .windows import (
    WindowConfig,
    build_window_manifest,
    load_chbmit_index,
    validate_group_integrity,
)


SERVER_PRETRAIN_SCHEMA_VERSION = "eeg_continual_pretrain_fold_v1"
SERVER_SPLIT_SCHEMA_VERSION = "eeg_continual_pretrain_split_v1"
PAPER_FOLDS: dict[int, tuple[str, ...]] = {
    0: ("chb01", "chb02", "chb03", "chb04", "chb21"),
    1: ("chb05", "chb06", "chb07", "chb08", "chb09"),
    2: ("chb10", "chb11", "chb12", "chb13", "chb14"),
    3: ("chb15", "chb16", "chb17", "chb18", "chb19"),
    4: ("chb20", "chb22", "chb23", "chb24"),
}


@dataclass(frozen=True)
class ServerPretrainConfig:
    seed: int = 42
    max_epochs: int = 20
    micro_batch_size: int = 64
    effective_batch_size: int = 128
    prediction_batch_size: int = 128
    base_learning_rate: float = 1e-4
    lora_learning_rate: float = 2e-5
    llm_learning_rate_multiplier: float = 0.1
    head_learning_rate_multiplier: float = 1.0
    minimum_learning_rate: float = 1e-6
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    minimum_evaluation_recall: float = 0.6
    enrollment_baseline_windows: int = 128
    num_workers: int = 0

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.micro_batch_size < 1 or self.effective_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.prediction_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Prediction batch/workers are invalid")
        if (
            self.base_learning_rate <= 0
            or self.lora_learning_rate <= 0
            or self.minimum_learning_rate < 0
            or self.weight_decay < 0
        ):
            raise ValueError("Optimizer settings are invalid")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.enrollment_baseline_windows < 1:
            raise ValueError("enrollment_baseline_windows must be positive")
        if not 0 <= self.minimum_evaluation_recall <= 1:
            raise ValueError("minimum_evaluation_recall must be in [0, 1]")

    @property
    def accumulation_steps(self) -> int:
        return max(1, math.ceil(
            self.effective_batch_size / self.micro_batch_size
        ))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "accumulation_steps": self.accumulation_steps,
            "actual_effective_batch_size": (
                self.micro_batch_size * self.accumulation_steps
            ),
            "loss": "cross_entropy",
            "optimizer": "AdamW",
            "scheduler": "linear_warmup_then_cosine",
        }


def _partition_summary(
    windows: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
) -> dict[str, Any]:
    selected = [
        row for row in windows if str(row["subject_id"]) in set(subjects)
    ]
    by_subject: dict[str, dict[str, int]] = {}
    for subject in subjects:
        rows = [row for row in selected if row["subject_id"] == subject]
        by_subject[subject] = {
            "windows": len(rows),
            "normal": sum(int(row["label"]) == 0 for row in rows),
            "ictal": sum(int(row["label"]) == 1 for row in rows),
        }
    return {
        "subjects": list(subjects),
        "subject_count": len(subjects),
        "window_count": len(selected),
        "normal_windows": sum(int(row["label"]) == 0 for row in selected),
        "ictal_windows": sum(int(row["label"]) == 1 for row in selected),
        "by_subject": by_subject,
    }


def build_pretrain_fold_split(
    window_manifest: Mapping[str, Any],
    *,
    outer_fold: int,
    validation_fold: int | None = None,
) -> dict[str, Any]:
    if outer_fold not in PAPER_FOLDS:
        raise ValueError(f"Unknown paper fold: {outer_fold}")
    paper_subjects = {
        subject for fold in PAPER_FOLDS.values() for subject in fold
    }
    if len(paper_subjects) != 24:
        raise ValueError("Paper folds must contain 24 unique subjects")
    windows = list(window_manifest["windows"])
    manifest_subjects = {str(row["subject_id"]) for row in windows}
    if manifest_subjects != paper_subjects:
        raise ValueError("Window manifest subjects do not match paper folds")
    test_subjects = PAPER_FOLDS[outer_fold]
    if validation_fold is not None:
        if validation_fold not in PAPER_FOLDS:
            raise ValueError(f"Unknown validation fold: {validation_fold}")
        if validation_fold == outer_fold:
            raise ValueError("Validation fold must differ from outer test fold")
        validation_subjects = PAPER_FOLDS[validation_fold]
    else:
        validation_subjects = ()
    excluded_folds = {outer_fold}
    if validation_fold is not None:
        excluded_folds.add(validation_fold)
    train_subjects = tuple(
        subject
        for fold in sorted(PAPER_FOLDS)
        if fold not in excluded_folds
        for subject in PAPER_FOLDS[fold]
    )
    partition_sets = (
        set(train_subjects), set(validation_subjects), set(test_subjects)
    )
    if any(left & right for index, left in enumerate(partition_sets)
           for right in partition_sets[index + 1:]):
        raise ValueError("Train/validation/test patients overlap")
    if set().union(*partition_sets) != paper_subjects:
        raise ValueError("Train/validation/test patients are incomplete")
    partitions = {
        "source_train": _partition_summary(windows, train_subjects),
        "outer_test_sampled_manifest": _partition_summary(
            windows, test_subjects
        ),
    }
    if validation_subjects:
        partitions["source_validation_sampled_manifest"] = (
            _partition_summary(windows, validation_subjects)
        )
    strict_protocol = validation_fold is not None
    body = {
        "schema_version": SERVER_SPLIT_SCHEMA_VERSION,
        "window_manifest_sha256": str(
            window_manifest["window_manifest_sha256"]
        ),
        "outer_fold": outer_fold,
        "validation_fold": validation_fold,
        "paper_folds": {
            str(fold): list(subjects)
            for fold, subjects in PAPER_FOLDS.items()
        },
        "partitions": partitions,
        "training_contract": (
            "paper-fold patients excluding validation and outer test folds"
            if strict_protocol else
            "all paper-fold patients except the outer test fold"
        ),
        "evaluation_contract": (
            "select checkpoint and threshold on full natural validation "
            "timelines; evaluate outer test once with frozen threshold"
            if strict_protocol else
            "full natural outer-fold timelines evaluated every epoch"
        ),
        "selection_warning": (
            None if strict_protocol else
            "The source PDF selects the best epoch on the zero-shot test fold; "
            "this reproduces that behavior but is not a held-out final test."
        ),
    }
    return {**body, "split_sha256": canonical_hash(body)}


def partition_indices(
    window_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    partition: str,
) -> np.ndarray:
    try:
        subjects = set(split["partitions"][partition]["subjects"])
    except KeyError as exc:
        raise ValueError(f"Unknown pretrain partition: {partition}") from exc
    return np.asarray([
        index
        for index, row in enumerate(window_manifest["windows"])
        if str(row["subject_id"]) in subjects
    ], dtype=np.int64)


def validate_server_training_manifest(
    manifest: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(manifest["config"])
    expected_ratio = 7.0 / 3.0
    partitions = split["partitions"]
    source_subjects = set(partitions["source_train"]["subjects"])
    outer_subjects = set(
        partitions["outer_test_sampled_manifest"]["subjects"]
    )
    validation_subjects = set(
        partitions.get(
            "source_validation_sampled_manifest", {}
        ).get("subjects", [])
    )
    checks = {
        "window_seconds": float(config["window_seconds"]) == 4.0,
        "stride_seconds": float(config["stride_seconds"]) == 4.0,
        "seizure_guard_seconds": float(config["seizure_guard_seconds"]) == 0.0,
        "normal_to_ictal_ratio": math.isclose(
            float(config["normal_to_ictal_ratio"]),
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "source_subject_count": (
            len(source_subjects)
            == 24 - len(validation_subjects) - len(outer_subjects)
        ),
        "outer_subject_count": (
            int(partitions["outer_test_sampled_manifest"]["subject_count"])
            in {4, 5}
        ),
        "patient_partitions_disjoint": not (
            source_subjects & validation_subjects
            or source_subjects & outer_subjects
            or validation_subjects & outer_subjects
        ),
        "patient_partitions_complete": len(
            source_subjects | validation_subjects | outer_subjects
        ) == 24,
    }
    source_by_subject = partitions["source_train"]["by_subject"]
    maximum = max(
        int(summary["windows"])
        for summary in source_by_subject.values()
    )
    checks["per_subject_at_most_20000"] = maximum <= 20_000
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise ValueError(f"Server training manifest checks failed: {failed}")
    return {
        "checks": checks,
        "maximum_source_windows_per_subject": maximum,
        "passed": True,
    }


def _autocast(device: torch.device, precision: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_groups(
    model: ServerEEGVLPretrainModel,
    *,
    config: ServerPretrainConfig,
) -> list[dict[str, Any]]:
    efficientnet: list[nn.Parameter] = []
    lora: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_encoder.features."):
            efficientnet.append(parameter)
        elif name.startswith("language_model."):
            lora.append(parameter)
        else:
            head.append(parameter)
    if not efficientnet or not lora or not head:
        raise ValueError("Server optimizer groups are incomplete")
    return [
        {
            "params": efficientnet,
            "lr": config.base_learning_rate,
            "weight_decay": config.weight_decay,
            "group_name": "efficientnet",
        },
        {
            "params": lora,
            "lr": config.lora_learning_rate,
            "weight_decay": 0.0,
            "group_name": "qwen_lora_qv",
        },
        {
            "params": head,
            "lr": (
                config.base_learning_rate
                * config.head_learning_rate_multiplier
            ),
            "weight_decay": config.weight_decay,
            "group_name": "projection_prompt_head",
        },
    ]


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def schedule(base_lr: float) -> Callable[[int], float]:
        floor = min(1.0, minimum_learning_rate / base_lr)

        def multiplier(step: int) -> float:
            current = min(step + 1, total_steps)
            if current <= warmup_steps:
                return current / warmup_steps
            progress = (
                (current - warmup_steps)
                / max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [schedule(lr) for lr in base_lrs],
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes()
    ).hexdigest()


@torch.no_grad()
def compute_subject_log_spectral_baselines(
    encoder: ServerSTFTEfficientNetEncoder,
    *,
    images: np.ndarray,
    normal_indices_by_subject: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("Baseline batch size must be positive")
    encoder.to(device)
    encoder.eval()
    baselines: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for subject, indices_value in normal_indices_by_subject.items():
        indices = np.asarray(indices_value, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError(f"E2 baseline has no normal windows for {subject}")
        total = torch.zeros(
            encoder.config.eeg_channels,
            encoder.config.frequency_bins,
            dtype=torch.float64,
        )
        observations = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            waveform = torch.from_numpy(np.array(
                images[selected],
                dtype=np.float32,
                copy=True,
            )).to(device, non_blocking=True)
            log_magnitude = encoder.log_magnitude(waveform)
            total += log_magnitude.sum(dim=(0, 3)).double().cpu()
            observations += int(
                log_magnitude.shape[0] * log_magnitude.shape[3]
            )
        baseline = (total / observations).float().numpy()
        baselines[str(subject)] = baseline
        counts[str(subject)] = int(len(indices))
    summary = {
        "definition": (
            "mean log1p(abs(STFT)) over known-normal enrollment windows "
            "and STFT time frames"
        ),
        "window_counts": counts,
        "baseline_sha256": {
            subject: _sha256_array(values)
            for subject, values in baselines.items()
        },
    }
    return baselines, summary

def evaluate_natural_fold(
    model: ServerEEGVLPretrainModel,
    *,
    cache: NaturalFoldImageCache,
    dataset: NaturalFoldImageDataset,
    device: torch.device,
    batch_size: int,
    minimum_recall: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    probabilities, labels = predict_dataset(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    probabilities_by_subject: dict[str, np.ndarray] = {}
    labels_by_subject: dict[str, np.ndarray] = {}
    for subject in cache.metadata["subject_order"]:
        rows = cache.subject_slice(str(subject))
        probabilities_by_subject[str(subject)] = probabilities[rows]
        labels_by_subject[str(subject)] = labels[rows]
    selected_threshold = (
        float(threshold)
        if threshold is not None
        else find_optimal_threshold_exact(
            labels,
            probabilities,
            min_recall=minimum_recall,
        )
    )
    pooled = evaluate(
        labels,
        probabilities,
        threshold=selected_threshold,
        print_report=False,
        sample_duration_seconds=4.0,
    )
    patient_metrics = {
        subject: evaluate(
            labels_by_subject[subject],
            probabilities_by_subject[subject],
            threshold=selected_threshold,
            print_report=False,
            sample_duration_seconds=4.0,
        )
        for subject in cache.metadata["subject_order"]
    }
    macro = {
        metric: float(np.mean([
            float(values[metric])
            for values in patient_metrics.values()
            if values[metric] is not None
        ]))
        for metric in ("auroc", "auprc")
    }
    return {
        "threshold": float(selected_threshold),
        "threshold_selected_on_this_dataset": threshold is None,
        "pooled_metrics": pooled,
        "macro_patient_metrics": macro,
        "patient_metrics": patient_metrics,
        "labels": labels,
        "probabilities": probabilities,
        "labels_by_subject": labels_by_subject,
        "probabilities_by_subject": probabilities_by_subject,
    }


def train_server_pretrain(
    model: ServerEEGVLPretrainModel,
    training_dataset: S1ImageDataset,
    *,
    evaluate_epoch: Callable[[ServerEEGVLPretrainModel], dict[str, Any]],
    device: torch.device,
    config: ServerPretrainConfig,
) -> dict[str, Any]:
    config.validate()
    if training_dataset.augmentation is not None:
        raise ValueError("PDF base training does not declare augmentation")
    _seed_everything(config.seed)
    model.to(device)
    precision = select_precision(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        training_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config=config))
    updates_per_epoch = math.ceil(len(loader) / config.accumulation_steps)
    scheduler = _scheduler(
        optimizer,
        total_steps=max(1, updates_per_epoch * config.max_epochs),
        warmup_fraction=config.warmup_fraction,
        minimum_learning_rate=config.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    optimizer_steps = 0
    started = time.perf_counter()
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
            with _autocast(device, precision):
                logits = forward_eeg_batch(model, images, batch)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss / config.accumulation_steps).backward()
            update = (
                batch_index % config.accumulation_steps == 0
                or batch_index == len(loader)
            )
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    config.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            rows = int(labels.shape[0])
            epoch_loss += float(loss.detach()) * rows
            epoch_rows += rows

        evaluation = evaluate_epoch(model)
        pooled = evaluation["pooled_metrics"]
        score = pooled["auprc"]
        if score is None:
            raise ValueError("Fold evaluation AUPRC is undefined")
        improved = float(score) > best_score + 1e-12
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = portable_pretrain_state_dict(model)
            best_evaluation = evaluation
        history.append({
            "epoch": epoch,
            "training_loss": epoch_loss / max(1, epoch_rows),
            "pooled_auroc": pooled["auroc"],
            "pooled_auprc": pooled["auprc"],
            "pooled_f1": pooled["f1"],
            "threshold": evaluation["threshold"],
            "macro_patient_metrics": evaluation["macro_patient_metrics"],
            "patient_metrics": {
                subject: {
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                }
                for subject, metrics in evaluation["patient_metrics"].items()
            },
            "selected": improved,
            "learning_rates": {
                str(group["group_name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
        })
        print(
            f"Server pretrain epoch {epoch}/{config.max_epochs}: "
            f"loss={history[-1]['training_loss']:.6f}, "
            f"fold_auroc={float(pooled['auroc']):.6f}, "
            f"fold_auprc={float(score):.6f}, selected={improved}",
            flush=True,
        )

    if best_state is None or best_evaluation is None:
        raise RuntimeError("Server pretraining produced no checkpoint")
    load_portable_pretrain_state_dict(model, best_state)
    return {
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_evaluation": best_evaluation,
        "history": history,
        "precision": precision,
        "optimizer_steps": optimizer_steps,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "config": config.to_dict(),
    }


def _public_training(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in training.items()
        if key not in {"best_state_dict", "best_evaluation"}
    }


def save_server_checkpoint(
    model: ServerEEGVLPretrainModel,
    *,
    fold: int,
    training: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"fold{fold}_lora_stft_best.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": SERVER_PRETRAIN_SCHEMA_VERSION,
        "model_version": SERVER_PRETRAIN_MODEL_VERSION,
        "fold": fold,
        "model_contract": model.contract(),
        "training_config": dict(training["config"]),
        "best_epoch": int(training["best_epoch"]),
        "best_score": float(training["best_score"]),
        "state_dict": training["best_state_dict"],
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination, _file_sha256(destination)


def load_server_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[ServerEEGVLPretrainModel, dict[str, Any]]:
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        raise ValueError("Server pretrain checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_version") != SERVER_PRETRAIN_MODEL_VERSION:
        raise ValueError("Server pretrain checkpoint model version mismatch")
    contract = dict(payload["model_contract"])
    stft = dict(contract["stft"])
    model = ServerEEGVLPretrainModel.from_pretrained(
        local_files_only=True,
        pretrained_visual_encoder=True,
        stft_config=ServerSTFTConfig(**{
            key: stft[key]
            for key in ServerSTFTConfig.__dataclass_fields__
        }),
        lora_config=LoRAConfig(
            rank=int(contract["lora"]["rank"]),
            alpha=float(contract["lora"]["alpha"]),
            dropout=float(contract["lora"]["dropout"]),
            target_modules=tuple(contract["lora"]["target_modules"]),
        ),
        pooling=str(contract["pooling"]),
        visual_bypass=bool(contract["visual_bypass"]),
        relative_spectral_bypass=bool(
            contract.get("relative_spectral_bypass", False)
        ),
    )
    current_contract = model.contract()
    if "relative_spectral_bypass" not in contract:
        current_contract.pop("relative_spectral_bypass", None)
        current_contract.pop("e2", None)
    if current_contract != contract:
        raise ValueError("Server pretrain model contract changed")
    load_portable_pretrain_state_dict(model, payload["state_dict"])
    return model, payload


def prepare_server_training_data(
    *,
    index: Mapping[str, Any],
    data_root: Path,
    output_dir: Path,
    seed: int,
    progress_every: int = 500,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest = build_window_manifest(
        index,
        config=WindowConfig(
            window_seconds=4.0,
            stride_seconds=4.0,
            ictal_overlap_fraction=0.5,
            seizure_guard_seconds=0.0,
            normal_to_ictal_ratio=7.0 / 3.0,
            sampling_frequency_hz=256,
            sampling_seed=seed,
        ),
    )
    validate_group_integrity(manifest["windows"])
    manifest_path = write_content_addressed_json(
        manifest,
        Path(output_dir).resolve() / "data" / (
            "chbmit_server_pretrain_windows_"
            f"{manifest['window_manifest_sha256'][:12]}.json"
        ),
        hash_field="window_manifest_sha256",
    )
    cache_path = build_window_caches(
        index,
        manifest,
        data_root=data_root,
        output_dir=Path(output_dir).resolve() / "cache",
        progress_every=progress_every,
    )
    return manifest_path, cache_path, manifest


def run_server_pretrain_fold(
    *,
    index_path: Path,
    data_root: Path,
    output_dir: Path,
    fold: int = 0,
    validation_fold: int = 4,
    device: torch.device | None = None,
    config: ServerPretrainConfig | None = None,
    stft_config: ServerSTFTConfig | None = None,
    visual_bypass: bool = False,
    relative_spectral_bypass: bool = False,
    shared_data_dir: Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type != "cuda":
        raise RuntimeError("Server pretraining requires CUDA")
    if validation_fold == fold:
        raise ValueError("Validation fold must differ from outer test fold")
    settings = config or ServerPretrainConfig()
    settings.validate()
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    shared_dir = (
        Path(shared_data_dir).resolve()
        if shared_data_dir is not None else output_dir
    )
    index = load_chbmit_index(index_path)
    manifest_path, cache_path, manifest = prepare_server_training_data(
        index=index,
        data_root=data_root,
        output_dir=shared_dir,
        seed=settings.seed,
    )
    split = build_pretrain_fold_split(
        manifest,
        outer_fold=fold,
        validation_fold=validation_fold,
    )
    manifest_validation = validate_server_training_manifest(manifest, split)
    split_path = write_content_addressed_json(
        split,
        output_dir / "splits" / (
            f"fold{fold}_val{validation_fold}_"
            f"{split['split_sha256'][:12]}.json"
        ),
        hash_field="split_sha256",
    )
    source_cache = ChbmitWindowCache(cache_path)
    train_indices = partition_indices(manifest, split, "source_train")
    training_subjects = {
        str(manifest["windows"][int(row)]["subject_id"])
        for row in train_indices
    }
    held_out_subjects = (
        set(PAPER_FOLDS[fold]) | set(PAPER_FOLDS[validation_fold])
    )
    if training_subjects & held_out_subjects:
        raise ValueError(
            "Validation or outer-test patient leaked into source training"
        )
    preprocess = S1PreprocessConfig(recipe_id="p1_bandpass_clip_scale")
    preprocessed_path = build_s1_preprocessed_cache(
        source_cache,
        output_dir=shared_dir / "preprocessed",
        config=preprocess,
    )
    preprocessed = S1PreprocessedCache(preprocessed_path)
    training_dataset = S1ImageDataset(
        preprocessed,
        source_cache,
        manifest,
        train_indices,
        augmentation=None,
        seed=settings.seed,
    )
    timeline_config = WindowConfig(
        window_seconds=4.0,
        stride_seconds=4.0,
        ictal_overlap_fraction=0.5,
        seizure_guard_seconds=0.0,
        normal_to_ictal_ratio=0.0,
        sampling_frequency_hz=256,
        sampling_seed=settings.seed,
    )

    def build_natural_partition(
        subjects: Sequence[str],
    ) -> tuple[
        dict[str, DeepTargetTimeline],
        Path,
        NaturalFoldImageCache,
        NaturalFoldImageDataset,
    ]:
        timelines = {
            subject: DeepTargetTimeline(build_deep_target_timeline(
                index,
                target_subject=subject,
                output_dir=shared_dir / "timelines",
                window_config=timeline_config,
            ))
            for subject in subjects
        }
        cache_path = build_natural_fold_image_cache(
            index,
            timelines,
            data_root=data_root,
            output_dir=shared_dir / "natural_evaluation_cache",
            preprocess=preprocess,
            batch_size=128,
            progress_every=10_000 if progress else 0,
        )
        cache = NaturalFoldImageCache(cache_path)
        return timelines, cache_path, cache, NaturalFoldImageDataset(cache)

    def natural_baseline_indices(
        cache: NaturalFoldImageCache,
        subjects: Sequence[str],
    ) -> dict[str, np.ndarray]:
        selected: dict[str, np.ndarray] = {}
        for subject in subjects:
            rows = cache.subject_slice(subject)
            indices = np.arange(
                int(rows.start or 0),
                int(rows.stop or 0),
                dtype=np.int64,
            )
            normal = indices[
                np.asarray(cache.labels[indices], dtype=np.int64) == 0
            ]
            selected[subject] = normal[
                : settings.enrollment_baseline_windows
            ]
        return selected
    (
        validation_timelines,
        validation_cache_path,
        validation_cache,
        validation_dataset,
    ) = build_natural_partition(PAPER_FOLDS[validation_fold])
    _seed_everything(settings.seed)
    model = ServerEEGVLPretrainModel.from_pretrained(
        local_files_only=True,
        pretrained_visual_encoder=True,
        stft_config=(
            stft_config or ServerSTFTConfig(zscore_input=False)
        ),
        lora_config=LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        pooling="mean",
        visual_bypass=visual_bypass,
        relative_spectral_bypass=relative_spectral_bypass,
    )
    model_contract = model.contract()

    baseline_protocol: dict[str, Any] = {
        "enabled": relative_spectral_bypass,
        "enrollment_windows_per_patient": (
            settings.enrollment_baseline_windows
        ),
        "selection": "earliest known-normal windows in partition order",
        "patient_data_scope": "same patient only",
    }
    if relative_spectral_bypass:
        source_normal_indices = {
            subject: np.asarray([
                int(row)
                for row in train_indices
                if (
                    str(manifest["windows"][int(row)]["subject_id"])
                    == subject
                    and int(source_cache.labels[int(row)]) == 0
                )
            ][: settings.enrollment_baseline_windows], dtype=np.int64)
            for subject in sorted(training_subjects)
        }
        training_baselines, training_baseline_summary = (
            compute_subject_log_spectral_baselines(
                model.visual_encoder,
                images=preprocessed.images,
                normal_indices_by_subject=source_normal_indices,
                device=selected_device,
            )
        )
        validation_baselines, validation_baseline_summary = (
            compute_subject_log_spectral_baselines(
                model.visual_encoder,
                images=validation_cache.images,
                normal_indices_by_subject=natural_baseline_indices(
                    validation_cache,
                    PAPER_FOLDS[validation_fold],
                ),
                device=selected_device,
            )
        )
        training_dataset.subject_baselines = training_baselines
        validation_dataset.subject_baselines = validation_baselines
        baseline_protocol["source_train"] = training_baseline_summary
        baseline_protocol["source_validation"] = (
            validation_baseline_summary
        )
    def evaluate_epoch(current: ServerEEGVLPretrainModel) -> dict[str, Any]:
        return evaluate_natural_fold(
            current,
            cache=validation_cache,
            dataset=validation_dataset,
            device=selected_device,
            batch_size=settings.prediction_batch_size,
            minimum_recall=settings.minimum_evaluation_recall,
        )

    training = train_server_pretrain(
        model,
        training_dataset,
        evaluate_epoch=evaluate_epoch,
        device=selected_device,
        config=settings,
    )
    selection_evaluation = training["best_evaluation"]
    frozen_threshold = float(selection_evaluation["threshold"])
    checkpoint_path, checkpoint_sha256 = save_server_checkpoint(
        model,
        fold=fold,
        training=training,
        output_dir=output_dir / "checkpoints",
    )

    (
        outer_timelines,
        outer_cache_path,
        outer_cache,
        outer_dataset,
    ) = build_natural_partition(PAPER_FOLDS[fold])
    if relative_spectral_bypass:
        outer_baselines, outer_baseline_summary = (
            compute_subject_log_spectral_baselines(
                model.visual_encoder,
                images=outer_cache.images,
                normal_indices_by_subject=natural_baseline_indices(
                    outer_cache,
                    PAPER_FOLDS[fold],
                ),
                device=selected_device,
            )
        )
        outer_dataset.subject_baselines = outer_baselines
        baseline_protocol["outer_test"] = outer_baseline_summary
    outer_evaluation = evaluate_natural_fold(
        model,
        cache=outer_cache,
        dataset=outer_dataset,
        device=selected_device,
        batch_size=settings.prediction_batch_size,
        minimum_recall=settings.minimum_evaluation_recall,
        threshold=frozen_threshold,
    )
    if outer_evaluation["threshold_selected_on_this_dataset"]:
        raise RuntimeError("Outer-test evaluation unexpectedly selected a threshold")

    prediction_artifacts: dict[str, Any] = {}
    for subject, timeline in outer_timelines.items():
        probabilities = outer_evaluation[
            "probabilities_by_subject"
        ][subject]
        path, digest = _save_prediction_artifact(
            {"server_pretrain": probabilities},
            timeline=cast(Any, timeline),
            output_dir=output_dir / "predictions",
        )
        prediction_artifacts[subject] = {
            "path": str(path),
            "sha256": digest,
            "probability_sha256": _sha256_array(probabilities),
            "window_count": int(len(probabilities)),
        }

    def public_evaluation(
        evaluation: Mapping[str, Any],
        *,
        predictions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        public = {
            "threshold": evaluation["threshold"],
            "threshold_selected_on_this_dataset": evaluation[
                "threshold_selected_on_this_dataset"
            ],
            "pooled_metrics": evaluation["pooled_metrics"],
            "macro_patient_metrics": evaluation["macro_patient_metrics"],
            "patient_metrics": evaluation["patient_metrics"],
            "pooled_label_sha256": _sha256_array(evaluation["labels"]),
            "pooled_probability_sha256": _sha256_array(
                evaluation["probabilities"]
            ),
            "pooled_window_count": int(len(evaluation["labels"])),
        }
        if predictions is not None:
            public["prediction_artifacts"] = dict(predictions)
        return public

    selection_public = public_evaluation(selection_evaluation)
    evaluation_public = public_evaluation(
        outer_evaluation,
        predictions=prediction_artifacts,
    )
    body = {
        "schema_version": SERVER_PRETRAIN_SCHEMA_VERSION,
        "source_document": str(
            Path("C:/Users/Polaris/Downloads/eeg_continual_learning_doc.pdf")
        ),
        "fold": fold,
        "validation_fold": validation_fold,
        "split": {
            "path": str(split_path),
            "sha256": split["split_sha256"],
            "partitions": split["partitions"],
        },
        "methodology": {
            "reproduces_test_fold_epoch_selection": False,
            "outer_fold_is_finally_held_out": True,
            "checkpoint_selection_partition": "source_validation",
            "threshold_selection_partition": "source_validation",
            "outer_test_evaluations": 1,
            "reason": (
                "Checkpoint and threshold are selected on a patient-disjoint "
                "source validation fold; the outer fold is evaluated once."
            ),
        },
        "baseline_protocol": baseline_protocol,
        "manifest_validation": manifest_validation,
        "model_contract": model_contract,
        "training": _public_training(training),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": int(checkpoint_path.stat().st_size),
        },
        "selection_evaluation": selection_public,
        "evaluation": evaluation_public,
        "source": {
            "index": str(Path(index_path).resolve()),
            "index_sha256": index["index_sha256"],
            "window_manifest": str(manifest_path),
            "window_manifest_sha256": manifest[
                "window_manifest_sha256"
            ],
            "raw_cache": str(cache_path),
            "raw_cache_metadata_sha256": source_cache.metadata[
                "metadata_sha256"
            ],
            "preprocessed_cache": str(preprocessed_path),
            "validation_timelines": sorted(validation_timelines),
            "validation_natural_cache": str(validation_cache_path),
            "validation_natural_cache_metadata_sha256": (
                selection_evaluation.get("cache_metadata_sha256")
                or NaturalFoldImageCache(validation_cache_path).metadata[
                    "metadata_sha256"
                ]
            ),
            "outer_timelines": sorted(outer_timelines),
            "outer_natural_cache": str(outer_cache_path),
            "outer_natural_cache_metadata_sha256": (
                outer_cache.metadata["metadata_sha256"]
            ),
        },
        "runtime": runtime_metadata(selected_device),
        "duration_seconds": time.perf_counter() - started,
    }
    artifact_sha256 = canonical_hash(body)
    artifact = {**body, "artifact_sha256": artifact_sha256}
    artifact_path = write_content_addressed_json(
        artifact,
        output_dir / f"fold{fold}_pretrain_{artifact_sha256[:12]}.json",
        hash_field="artifact_sha256",
    )
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": int(training["best_epoch"]),
        "validation_auprc": selection_public["pooled_metrics"]["auprc"],
        "pooled_auroc": evaluation_public["pooled_metrics"]["auroc"],
        "pooled_auprc": evaluation_public["pooled_metrics"]["auprc"],
        "duration_seconds": body["duration_seconds"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "artifacts/chbmit/chbmit_index_v1_1e1f0a81ecde.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/chbmit/1.0.0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/eeg_continual_pretrain"),
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--validation-fold",
        type=int,
        choices=range(5),
        default=4,
        help=(
            "Patient-disjoint source fold used for checkpoint and threshold "
            "selection."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--win-length", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--visual-bypass", action="store_true")
    parser.add_argument(
        "--relative-spectral-bypass", action="store_true"
    )
    parser.add_argument(
        "--enrollment-baseline-windows", type=int, default=128
    )
    parser.add_argument(
        "--shared-data-dir",
        type=Path,
        default=None,
        help=(
            "Reuse manifests and preprocessed/natural caches from this "
            "directory while writing run artifacts to --output-dir."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    result = run_server_pretrain_fold(
        index_path=args.index,
        data_root=args.data_root,
        output_dir=args.output_dir,
        fold=args.fold,
        validation_fold=args.validation_fold,
        config=ServerPretrainConfig(
            max_epochs=args.max_epochs,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=128,
            prediction_batch_size=args.prediction_batch_size,
            enrollment_baseline_windows=(
                args.enrollment_baseline_windows
            ),
        ),
        stft_config=ServerSTFTConfig(
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            zscore_input=False,
        ),
        visual_bypass=args.visual_bypass,
        relative_spectral_bypass=args.relative_spectral_bypass,
        shared_data_dir=args.shared_data_dir,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
