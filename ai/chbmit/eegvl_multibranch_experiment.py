"""Train E1+E2+E3+E4 on strict CHB-MIT Fold 0 with full-band caches."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.v2.lightweight_dataset import write_content_addressed_json

from .cache import ChbmitWindowCache
from .deep_timeline import DeepTargetTimeline, build_deep_target_timeline
from .eeg_continual_eval_cache import (
    NaturalFoldImageCache,
    NaturalFoldImageDataset,
    build_natural_fold_image_cache,
)
from .eeg_continual_pretrain import (
    PAPER_FOLDS,
    compute_subject_log_spectral_baselines,
    evaluate_natural_fold,
    partition_indices,
)
from .eeg_continual_pretrain_model import ServerSTFTConfig
from .eegmamba_b_experiment import (
    _autocast,
    _load_json,
    _normal_indices,
    _public_evaluation,
    _scheduler,
    _seed_everything,
    runtime_path,
)
from .eegvl_m9_model import LoRAConfig
from .eegvl_models import DEFAULT_QWEN_MODEL
from .eegvl_multibranch_model import (
    EEGVLE1E2E3E4Classifier,
    checkpoint_sha256,
    load_portable_multibranch_state_dict,
    portable_multibranch_state_dict,
)
from .eegvl_s1_data import (
    S1ImageDataset,
    S1PreprocessConfig,
    S1PreprocessedCache,
    build_s1_preprocessed_cache,
)
from .eegvl_training import forward_eeg_batch, select_precision
from .index import canonical_hash
from .windows import WindowConfig, load_chbmit_index


MULTIBRANCH_EXPERIMENT_SCHEMA_VERSION = "eegvl_multibranch_fold_v1"


@dataclass(frozen=True)
class MultibranchTrainingConfig:
    seed: int = 42
    max_epochs: int = 5
    micro_batch_size: int = 32
    effective_batch_size: int = 32
    prediction_batch_size: int = 128
    efficientnet_learning_rate: float = 1e-4
    head_learning_rate: float = 1e-4
    lora_learning_rate: float = 2e-5
    e2_learning_rate: float = 5e-3
    e3_learning_rate: float = 1e-4
    e4_learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-6
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    minimum_evaluation_recall: float = 0.6
    enrollment_baseline_windows: int = 128
    checkpoint_metric: str = "auprc"
    num_workers: int = 0

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.micro_batch_size < 1 or self.effective_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.prediction_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Prediction batch/workers are invalid")
        learning_rates = (
            self.efficientnet_learning_rate,
            self.head_learning_rate,
            self.lora_learning_rate,
            self.e2_learning_rate,
            self.e3_learning_rate,
            self.e4_learning_rate,
        )
        if min(learning_rates) <= 0:
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
        if self.checkpoint_metric not in {"auroc", "auprc"}:
            raise ValueError("checkpoint_metric must be auroc or auprc")

    @property
    def accumulation_steps(self) -> int:
        return max(1, math.ceil(self.effective_batch_size / self.micro_batch_size))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "accumulation_steps": self.accumulation_steps,
            "actual_effective_batch_size": (
                self.micro_batch_size * self.accumulation_steps
            ),
            "loss": "cross_entropy",
            "logit_adjustment_tau": 0.0,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup_then_cosine",
        }


def _optimizer_groups(
    model: EEGVLE1E2E3E4Classifier,
    *,
    config: MultibranchTrainingConfig,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[nn.Parameter]] = {
        "efficientnet": [],
        "head": [],
        "lora": [],
        "e2": [],
        "e3": [],
        "e4": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_encoder.features."):
            grouped["efficientnet"].append(parameter)
        elif name.startswith("language_model."):
            grouped["lora"].append(parameter)
        elif name.startswith("e2_proj."):
            grouped["e2"].append(parameter)
        elif name.startswith("e3_proj."):
            grouped["e3"].append(parameter)
        elif name.startswith("e4_proj."):
            grouped["e4"].append(parameter)
        else:
            grouped["head"].append(parameter)
    incomplete = sorted(name for name, parameters in grouped.items() if not parameters)
    if incomplete:
        raise ValueError(f"Multibranch optimizer groups are incomplete: {incomplete}")
    specifications = (
        ("efficientnet", config.efficientnet_learning_rate, config.weight_decay),
        ("head", config.head_learning_rate, config.weight_decay),
        ("lora", config.lora_learning_rate, 0.0),
        ("e2", config.e2_learning_rate, config.weight_decay),
        ("e3", config.e3_learning_rate, config.weight_decay),
        ("e4", config.e4_learning_rate, config.weight_decay),
    )
    return [
        {
            "params": grouped[name],
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "group_name": name,
        }
        for name, learning_rate, weight_decay in specifications
    ]


def train_multibranch(
    model: EEGVLE1E2E3E4Classifier,
    training_dataset: S1ImageDataset,
    *,
    evaluate_epoch: Callable[[EEGVLE1E2E3E4Classifier], dict[str, Any]],
    device: torch.device,
    config: MultibranchTrainingConfig,
) -> dict[str, Any]:
    config.validate()
    if training_dataset.augmentation is not None:
        raise ValueError("Matched multibranch experiment uses no augmentation")
    _seed_everything(config.seed)
    model.to(device)
    precision = select_precision(device)
    loader = DataLoader(
        training_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    secondary_metric = "auprc" if config.checkpoint_metric == "auroc" else "auroc"
    secondary_state: dict[str, torch.Tensor] | None = None
    secondary_evaluation: dict[str, Any] | None = None
    secondary_epoch = 0
    secondary_score = float("-inf")
    optimizer_steps = 0
    minibatches = 0
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
                torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            rows = int(labels.shape[0])
            epoch_loss += float(loss.detach()) * rows
            epoch_rows += rows
            minibatches += 1

        evaluation = evaluate_epoch(model)
        pooled = evaluation["pooled_metrics"]
        scores = {
            "auroc": pooled["auroc"],
            "auprc": pooled["auprc"],
        }
        if any(value is None for value in scores.values()):
            raise ValueError("Validation AUROC/AUPRC is undefined")
        score = float(scores[config.checkpoint_metric])
        current_secondary_score = float(scores[secondary_metric])
        improved = score > best_score + 1e-12
        secondary_improved = current_secondary_score > secondary_score + 1e-12
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = portable_multibranch_state_dict(model)
            best_evaluation = evaluation
        if secondary_improved:
            secondary_score = current_secondary_score
            secondary_epoch = epoch
            secondary_state = portable_multibranch_state_dict(model)
            secondary_evaluation = evaluation
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / max(1, epoch_rows),
                "minibatches_completed": minibatches,
                "optimizer_steps_completed": optimizer_steps,
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
                "selected_metrics": {
                    config.checkpoint_metric: improved,
                    secondary_metric: secondary_improved,
                },
                "learning_rates": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
            }
        )
        print(
            f"E1+E2+E3+E4 epoch {epoch}/{config.max_epochs}: "
            f"loss={history[-1]['training_loss']:.6f}, "
            f"val_auroc={float(pooled['auroc']):.6f}, "
            f"val_auprc={float(pooled['auprc']):.6f}, "
            f"selected_{config.checkpoint_metric}={improved}, "
            f"selected_{secondary_metric}={secondary_improved}",
            flush=True,
        )

    if best_state is None or best_evaluation is None:
        raise RuntimeError("Multibranch training produced no checkpoint")
    if secondary_state is None or secondary_evaluation is None:
        raise RuntimeError("Multibranch training produced no secondary checkpoint")
    load_portable_multibranch_state_dict(model, best_state)
    return {
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_evaluation": best_evaluation,
        "best_metric": config.checkpoint_metric,
        "secondary_state_dict": secondary_state,
        "secondary_epoch": secondary_epoch,
        "secondary_score": secondary_score,
        "secondary_evaluation": secondary_evaluation,
        "secondary_metric": secondary_metric,
        "history": history,
        "precision": precision,
        "minibatches": minibatches,
        "optimizer_steps": optimizer_steps,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "config": config.to_dict(),
    }


def _save_checkpoint(
    model: EEGVLE1E2E3E4Classifier,
    *,
    training: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "fold0_e1_e2_e3_e4_best.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": MULTIBRANCH_EXPERIMENT_SCHEMA_VERSION,
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
    return destination, checkpoint_sha256(destination)


def _build_fullband_natural_cache(
    *,
    index: Mapping[str, Any],
    subjects: Sequence[str],
    data_root: Path,
    shared_dir: Path,
    preprocess: S1PreprocessConfig,
    seed: int,
) -> tuple[NaturalFoldImageCache, NaturalFoldImageDataset]:
    timeline_config = WindowConfig(
        window_seconds=4.0,
        stride_seconds=4.0,
        ictal_overlap_fraction=0.5,
        seizure_guard_seconds=0.0,
        normal_to_ictal_ratio=0.0,
        sampling_frequency_hz=256,
        sampling_seed=seed,
    )
    timelines = {
        subject: DeepTargetTimeline(
            build_deep_target_timeline(
                index,
                target_subject=subject,
                output_dir=shared_dir / "timelines",
                window_config=timeline_config,
            )
        )
        for subject in subjects
    }
    path = build_natural_fold_image_cache(
        index,
        timelines,
        data_root=data_root,
        output_dir=shared_dir / "natural_evaluation_cache",
        preprocess=preprocess,
        batch_size=128,
        progress_every=10_000,
    )
    cache = NaturalFoldImageCache(path)
    return cache, NaturalFoldImageDataset(cache)


def run_multibranch_fold0(
    *,
    reference_artifact_path: Path,
    data_root: Path,
    output_dir: Path,
    qwen_model_name: str = DEFAULT_QWEN_MODEL,
    local_files_only: bool = True,
    config: MultibranchTrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    settings = config or MultibranchTrainingConfig()
    settings.validate()
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Multibranch experiment requires CUDA")
    started = time.perf_counter()
    reference_path = runtime_path(reference_artifact_path).resolve()
    reference = _load_json(reference_path)
    if int(reference.get("fold", -1)) != 0 or int(reference.get("validation_fold", -1)) != 4:
        raise ValueError("Multibranch experiment requires strict Fold 0 / Fold 4 reference")
    if not reference.get("methodology", {}).get("outer_fold_is_finally_held_out"):
        raise ValueError("Reference artifact does not keep Fold 0 held out")
    split = _load_json(runtime_path(reference["split"]["path"]))
    manifest = _load_json(runtime_path(reference["source"]["window_manifest"]))
    source_cache = ChbmitWindowCache(runtime_path(reference["source"]["raw_cache"]))
    index = load_chbmit_index(runtime_path(reference["source"]["index"]))
    output_dir = runtime_path(output_dir).resolve()
    data_root = runtime_path(data_root).resolve()

    preprocess = S1PreprocessConfig(recipe_id="p0_clip_scale")
    preprocessed_path = build_s1_preprocessed_cache(
        source_cache,
        output_dir=output_dir / "preprocessed",
        config=preprocess,
    )
    preprocessed = S1PreprocessedCache(preprocessed_path)
    train_indices = partition_indices(manifest, split, "source_train")
    training_dataset = S1ImageDataset(
        preprocessed,
        source_cache,
        manifest,
        train_indices,
        augmentation=None,
        seed=settings.seed,
    )
    validation_cache, validation_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=PAPER_FOLDS[4],
        data_root=data_root,
        shared_dir=output_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )

    _seed_everything(settings.seed)
    stft_config = ServerSTFTConfig(
        n_fft=256,
        win_length=128,
        hop_length=32,
        zscore_input=False,
    )
    model = EEGVLE1E2E3E4Classifier.from_pretrained(
        qwen_model_name=qwen_model_name,
        local_files_only=local_files_only,
        pretrained_visual_encoder=True,
        stft_config=stft_config,
        lora_config=LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        pooling="mean",
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
        model.visual_encoder,
        images=preprocessed.images,
        normal_indices_by_subject=source_normal_indices,
        device=selected_device,
    )
    validation_baselines, validation_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
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

    def evaluate_epoch(current: EEGVLE1E2E3E4Classifier) -> dict[str, Any]:
        return evaluate_natural_fold(
            current,
            cache=validation_cache,
            dataset=validation_dataset,
            device=selected_device,
            batch_size=settings.prediction_batch_size,
            minimum_recall=settings.minimum_evaluation_recall,
        )

    training = train_multibranch(
        model,
        training_dataset,
        evaluate_epoch=evaluate_epoch,
        device=selected_device,
        config=settings,
    )
    frozen_threshold = float(training["best_evaluation"]["threshold"])
    checkpoint_path, checkpoint_digest = _save_checkpoint(
        model,
        training=training,
        output_dir=output_dir / "checkpoints",
    )

    outer_cache, outer_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=PAPER_FOLDS[0],
        data_root=data_root,
        shared_dir=output_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )
    outer_baselines, outer_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
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
        if key not in {
            "best_state_dict",
            "best_evaluation",
            "secondary_state_dict",
            "secondary_evaluation",
        }
    }
    body = {
        "schema_version": MULTIBRANCH_EXPERIMENT_SCHEMA_VERSION,
        "method": "E1 EfficientNet-Qwen + E2/E3/E4 additive residuals",
        "fold": 0,
        "validation_fold": 4,
        "methodology": {
            "source_train_subjects": list(training_subjects),
            "checkpoint_selection_partition": "source_validation_fold4",
            "threshold_selection_partition": "source_validation_fold4",
            "outer_test_partition": "held_out_fold0",
            "outer_test_evaluations": 1,
            "training_augmentation": False,
            "preprocess": preprocess.to_dict(),
            "full_band_before_model_stft": True,
            "per_window_channel_zscore": False,
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
            "sha256": checkpoint_digest,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "portable": "frozen Qwen base weights excluded",
        },
        "selection_evaluation": selection_public,
        "evaluation": evaluation_public,
        "source": {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": reference["artifact_sha256"],
            "split_sha256": split["split_sha256"],
            "window_manifest_sha256": manifest["window_manifest_sha256"],
            "index_sha256": index["index_sha256"],
            "qwen_model": qwen_model_name,
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
    artifact_digest = canonical_hash(body)
    artifact = {**body, "artifact_sha256": artifact_digest}
    artifact_path = write_content_addressed_json(
        artifact,
        output_dir / f"fold0_e1_e2_e3_e4_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    result = {
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_digest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
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
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/1.0.0"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/eegvl_multibranch_fullband"),
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    result = run_multibranch_fold0(
        reference_artifact_path=args.reference_artifact,
        data_root=args.data_root,
        output_dir=args.output_dir,
        qwen_model_name=args.qwen_model,
        local_files_only=not args.allow_model_download,
        config=MultibranchTrainingConfig(
            max_epochs=args.max_epochs,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
