"""Train and evaluate Mamba-B on the strict CHB-MIT Fold 0 protocol."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import re
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.v2.lightweight_dataset import write_content_addressed_json

from .cache import ChbmitWindowCache
from .eeg_continual_eval_cache import (
    NaturalFoldImageCache,
    NaturalFoldImageDataset,
)
from .eeg_continual_pretrain import (
    PAPER_FOLDS,
    compute_subject_log_spectral_baselines,
    evaluate_natural_fold,
    partition_indices,
)
from .eeg_continual_pretrain_model import ServerSTFTConfig
from .eegmamba_b import (
    OFFICIAL_EEGMAMBA_CHECKPOINT_SHA256,
    EEGMambaBE2Classifier,
    EEGMambaInputConfig,
    file_sha256,
)
from .eegvl_s1_data import S1ImageDataset, S1PreprocessedCache
from .eegvl_training import forward_eeg_batch, select_precision
from .index import canonical_hash


MAMBA_B_EXPERIMENT_SCHEMA_VERSION = "eegmamba_b_fold_v1"


@dataclass(frozen=True)
class MambaBTrainingConfig:
    seed: int = 42
    max_epochs: int = 1
    micro_batch_size: int = 64
    effective_batch_size: int = 128
    prediction_batch_size: int = 256
    backbone_learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
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
        if self.backbone_learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        if self.minimum_learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("Optimizer settings are invalid")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.minimum_evaluation_recall <= 1:
            raise ValueError("minimum_evaluation_recall must be in [0, 1]")
        if self.enrollment_baseline_windows < 1:
            raise ValueError("enrollment_baseline_windows must be positive")

    @property
    def accumulation_steps(self) -> int:
        return max(1, math.ceil(self.effective_batch_size / self.micro_batch_size))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "accumulation_steps": self.accumulation_steps,
            "actual_effective_batch_size": (self.micro_batch_size * self.accumulation_steps),
            "loss": "cross_entropy",
            "optimizer": "AdamW",
            "scheduler": "linear_warmup_then_cosine",
        }


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def runtime_path(value: str | Path) -> Path:
    """Translate recorded Windows artifact paths when running under WSL."""
    text = str(value)
    if os.name != "nt":
        match = re.fullmatch(r"([A-Za-z]):\\(.*)", text)
        if match:
            drive, suffix = match.groups()
            return Path(f"/mnt/{drive.lower()}/{suffix.replace(chr(92), '/')}")
    return Path(text)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))

    def schedule(base_lr: float) -> Callable[[int], float]:
        floor = min(1.0, minimum_learning_rate / base_lr)

        def multiplier(step: int) -> float:
            current = min(step + 1, total_steps)
            if current <= warmup_steps:
                return current / warmup_steps
            progress = (current - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine

        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [schedule(float(group["lr"])) for group in optimizer.param_groups],
    )


def _optimizer_groups(
    model: EEGMambaBE2Classifier,
    *,
    config: MambaBTrainingConfig,
) -> list[dict[str, Any]]:
    backbone = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    ]
    head = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    if not backbone or not head:
        raise ValueError("Mamba-B optimizer groups are incomplete")
    return [
        {
            "params": backbone,
            "lr": config.backbone_learning_rate,
            "weight_decay": config.weight_decay,
            "group_name": "eegmamba_backbone",
        },
        {
            "params": head,
            "lr": config.head_learning_rate,
            "weight_decay": config.weight_decay,
            "group_name": "e2_fusion_head",
        },
    ]


def train_mamba_b(
    model: EEGMambaBE2Classifier,
    training_dataset: S1ImageDataset,
    *,
    evaluate_epoch: Callable[[EEGMambaBE2Classifier], dict[str, Any]],
    device: torch.device,
    config: MambaBTrainingConfig,
) -> dict[str, Any]:
    config.validate()
    if training_dataset.augmentation is not None:
        raise ValueError("Mamba-B screening uses no training augmentation")
    _seed_everything(config.seed)
    model.to(device)
    precision = select_precision(device)
    loader = DataLoader(
        training_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
            update = batch_index % config.accumulation_steps == 0 or batch_index == len(loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
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
            raise ValueError("Validation AUPRC is undefined")
        improved = float(score) > best_score + 1e-12
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_evaluation = evaluation
        history.append(
            {
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
                    str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups
                },
            }
        )
        print(
            f"Mamba-B epoch {epoch}/{config.max_epochs}: "
            f"loss={history[-1]['training_loss']:.6f}, "
            f"val_auroc={float(pooled['auroc']):.6f}, "
            f"val_auprc={float(score):.6f}, selected={improved}",
            flush=True,
        )

    if best_state is None or best_evaluation is None:
        raise RuntimeError("Mamba-B training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
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
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "config": config.to_dict(),
    }


def _normal_indices(
    cache: NaturalFoldImageCache,
    subjects: Sequence[str],
    *,
    maximum: int,
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for subject in subjects:
        rows = cache.subject_slice(subject)
        indices = np.arange(int(rows.start or 0), int(rows.stop or 0), dtype=np.int64)
        normal = indices[np.asarray(cache.labels[indices], dtype=np.int64) == 0]
        selected[subject] = normal[:maximum]
    return selected


def _sha256_array(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "threshold": evaluation["threshold"],
        "threshold_selected_on_this_dataset": evaluation["threshold_selected_on_this_dataset"],
        "pooled_metrics": evaluation["pooled_metrics"],
        "macro_patient_metrics": evaluation["macro_patient_metrics"],
        "patient_metrics": evaluation["patient_metrics"],
        "pooled_label_sha256": _sha256_array(evaluation["labels"]),
        "pooled_probability_sha256": _sha256_array(evaluation["probabilities"]),
        "pooled_window_count": int(len(evaluation["labels"])),
    }


def _save_checkpoint(
    model: EEGMambaBE2Classifier,
    *,
    training: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "fold0_mamba_b_best.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": MAMBA_B_EXPERIMENT_SCHEMA_VERSION,
            "model_version": model.model_version,
            "model_contract": model.contract(),
            "training_config": dict(training["config"]),
            "best_epoch": int(training["best_epoch"]),
            "best_score": float(training["best_score"]),
            "state_dict": training["best_state_dict"],
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination, file_sha256(destination)


def run_mamba_b_fold0(
    *,
    reference_artifact_path: Path,
    official_checkpoint_path: Path,
    output_dir: Path,
    config: MambaBTrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    settings = config or MambaBTrainingConfig()
    settings.validate()
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Mamba-B experiment requires CUDA")
    started = time.perf_counter()
    reference_path = runtime_path(reference_artifact_path).resolve()
    reference = _load_json(reference_path)
    if int(reference.get("fold", -1)) != 0:
        raise ValueError("Mamba-B screening requires the Fold 0 reference")
    if int(reference.get("validation_fold", -1)) != 4:
        raise ValueError("Mamba-B screening requires Fold 4 validation")
    if not reference.get("methodology", {}).get("outer_fold_is_finally_held_out"):
        raise ValueError("Reference artifact does not use strict held-out Fold 0")
    split = _load_json(runtime_path(reference["split"]["path"]))
    if split["split_sha256"] != reference["split"]["sha256"]:
        raise ValueError("Reference split SHA256 mismatch")
    manifest = _load_json(runtime_path(reference["source"]["window_manifest"]))
    source_cache = ChbmitWindowCache(runtime_path(reference["source"]["raw_cache"]))
    preprocessed = S1PreprocessedCache(runtime_path(reference["source"]["preprocessed_cache"]))
    train_indices = partition_indices(manifest, split, "source_train")
    training_dataset = S1ImageDataset(
        preprocessed,
        source_cache,
        manifest,
        train_indices,
        augmentation=None,
        seed=settings.seed,
    )
    validation_cache = NaturalFoldImageCache(
        runtime_path(reference["source"]["validation_natural_cache"])
    )
    validation_dataset = NaturalFoldImageDataset(validation_cache)
    outer_cache = NaturalFoldImageCache(runtime_path(reference["source"]["outer_natural_cache"]))
    outer_dataset = NaturalFoldImageDataset(outer_cache)

    official_checkpoint = runtime_path(official_checkpoint_path).resolve()
    if file_sha256(official_checkpoint) != OFFICIAL_EEGMAMBA_CHECKPOINT_SHA256:
        raise ValueError("Official EEGMamba checkpoint SHA256 mismatch")
    _seed_everything(settings.seed)
    stft_config = ServerSTFTConfig(
        n_fft=64,
        win_length=64,
        hop_length=32,
        zscore_input=False,
    )
    model = EEGMambaBE2Classifier.from_official_checkpoint(
        official_checkpoint,
        input_config=EEGMambaInputConfig(),
        stft_config=stft_config,
    )

    training_subjects = tuple(split["partitions"]["source_train"]["subjects"])
    source_normal_indices = {
        subject: np.asarray(
            [
                int(row)
                for row in train_indices
                if (
                    str(manifest["windows"][int(row)]["subject_id"]) == subject
                    and int(source_cache.labels[int(row)]) == 0
                )
            ][: settings.enrollment_baseline_windows],
            dtype=np.int64,
        )
        for subject in training_subjects
    }
    training_baselines, training_baseline_summary = compute_subject_log_spectral_baselines(
        model.e2_frontend,
        images=preprocessed.images,
        normal_indices_by_subject=source_normal_indices,
        device=selected_device,
    )
    validation_baselines, validation_baseline_summary = compute_subject_log_spectral_baselines(
        model.e2_frontend,
        images=validation_cache.images,
        normal_indices_by_subject=_normal_indices(
            validation_cache,
            PAPER_FOLDS[4],
            maximum=settings.enrollment_baseline_windows,
        ),
        device=selected_device,
    )
    training_dataset.subject_baselines = training_baselines
    validation_dataset.subject_baselines = validation_baselines

    def evaluate_epoch(current: EEGMambaBE2Classifier) -> dict[str, Any]:
        return evaluate_natural_fold(
            current,
            cache=validation_cache,
            dataset=validation_dataset,
            device=selected_device,
            batch_size=settings.prediction_batch_size,
            minimum_recall=settings.minimum_evaluation_recall,
        )

    training = train_mamba_b(
        model,
        training_dataset,
        evaluate_epoch=evaluate_epoch,
        device=selected_device,
        config=settings,
    )
    frozen_threshold = float(training["best_evaluation"]["threshold"])
    output_dir = runtime_path(output_dir).resolve()
    checkpoint_path, checkpoint_sha256 = _save_checkpoint(
        model,
        training=training,
        output_dir=output_dir / "checkpoints",
    )

    outer_baselines, outer_baseline_summary = compute_subject_log_spectral_baselines(
        model.e2_frontend,
        images=outer_cache.images,
        normal_indices_by_subject=_normal_indices(
            outer_cache,
            PAPER_FOLDS[0],
            maximum=settings.enrollment_baseline_windows,
        ),
        device=selected_device,
    )
    outer_dataset.subject_baselines = outer_baselines
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
        raise RuntimeError("Fold 0 unexpectedly selected its own threshold")

    selection_public = _public_evaluation(training["best_evaluation"])
    evaluation_public = _public_evaluation(outer_evaluation)
    public_training = {
        key: value
        for key, value in training.items()
        if key not in {"best_state_dict", "best_evaluation"}
    }
    body = {
        "schema_version": MAMBA_B_EXPERIMENT_SCHEMA_VERSION,
        "method": "Mamba-B: official EEGMamba + E2 + classification head",
        "fold": 0,
        "validation_fold": 4,
        "methodology": {
            "source_train_subjects": list(training_subjects),
            "checkpoint_selection_partition": "source_validation_fold4",
            "threshold_selection_partition": "source_validation_fold4",
            "outer_test_partition": "held_out_fold0",
            "outer_test_evaluations": 1,
            "directly_comparable_reference_artifact": str(reference_path),
            "training_augmentation": False,
        },
        "model_contract": model.contract(),
        "baseline_protocol": {
            "enrollment_windows_per_patient": settings.enrollment_baseline_windows,
            "selection": "earliest known-normal windows in partition order",
            "source_train": training_baseline_summary,
            "source_validation": validation_baseline_summary,
            "outer_test": outer_baseline_summary,
        },
        "training": public_training,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": int(checkpoint_path.stat().st_size),
        },
        "selection_evaluation": selection_public,
        "evaluation": evaluation_public,
        "source": {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": reference["artifact_sha256"],
            "split_sha256": split["split_sha256"],
            "window_manifest_sha256": manifest["window_manifest_sha256"],
            "official_checkpoint": str(official_checkpoint),
            "official_checkpoint_sha256": file_sha256(official_checkpoint),
            "raw_cache": str(source_cache.path),
            "preprocessed_cache": str(preprocessed.path),
            "validation_natural_cache": str(validation_cache.path),
            "outer_natural_cache": str(outer_cache.path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(selected_device),
        },
        "duration_seconds": time.perf_counter() - started,
    }
    artifact_sha256 = canonical_hash(body)
    artifact = {**body, "artifact_sha256": artifact_sha256}
    artifact_path = write_content_addressed_json(
        artifact,
        output_dir / f"fold0_mamba_b_{artifact_sha256[:12]}.json",
        hash_field="artifact_sha256",
    )
    result = {
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": int(training["best_epoch"]),
        "validation_auprc": selection_public["pooled_metrics"]["auprc"],
        "pooled_auroc": evaluation_public["pooled_metrics"]["auroc"],
        "pooled_auprc": evaluation_public["pooled_metrics"]["auprc"],
        "pooled_f1": evaluation_public["pooled_metrics"]["f1"],
        "duration_seconds": body["duration_seconds"],
    }
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/"
            "fold0_pretrain_c27817a49668.json"
        ),
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=Path("artifacts/chbmit/eegmamba/pretrained_EEGMamba.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/eegmamba_b_fold0"),
    )
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    result = run_mamba_b_fold0(
        reference_artifact_path=args.reference_artifact,
        official_checkpoint_path=args.official_checkpoint,
        output_dir=args.output_dir,
        config=MambaBTrainingConfig(
            max_epochs=args.max_epochs,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=128,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
