"""Train the curated four-branch model on Folds 0-2 and test Fold 4 once.

Fold 3 is a complete natural-timeline validation set used for checkpoint and
threshold selection. Fold 4 remains untouched until the best checkpoint and
decision threshold have both been frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.chbmit.cache import ChbmitWindowCache
from ai.chbmit.eeg_continual_pretrain import (
    PAPER_FOLDS,
    compute_subject_log_spectral_baselines,
    evaluate_natural_fold,
)
from ai.chbmit.eegmamba_b_experiment import (
    _load_json,
    _normal_indices,
    _public_evaluation,
    _seed_everything,
    runtime_path,
)
from ai.chbmit.eegvl_multibranch_experiment import (
    MultibranchTrainingConfig,
    _build_fullband_natural_cache,
    train_multibranch,
)
from ai.chbmit.eegvl_multibranch_model import checkpoint_sha256
from ai.chbmit.eegvl_s1_data import (
    S1ImageDataset,
    S1PreprocessConfig,
    S1PreprocessedCache,
    build_s1_preprocessed_cache,
)
from ai.chbmit.index import canonical_hash
from ai.chbmit.windows import load_chbmit_index

from .model import DEFAULT_QWEN_MODEL, build_model


SCHEMA_VERSION = "good_multibranch_fold012_val3_test4_v1"
TRAIN_FOLDS = (0, 1, 2)
VALIDATION_FOLD = 3
TEST_FOLD = 4


def build_partition(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build a complete patient-disjoint Fold 0-2 / Fold 3 / Fold 4 split."""
    windows = list(manifest["windows"])
    expected_subjects = {
        subject for fold_subjects in PAPER_FOLDS.values() for subject in fold_subjects
    }
    manifest_subjects = {str(row["subject_id"]) for row in windows}
    if manifest_subjects != expected_subjects:
        raise ValueError("Window manifest subjects do not match the paper folds")

    training_subjects = tuple(
        subject for fold in TRAIN_FOLDS for subject in PAPER_FOLDS[fold]
    )
    validation_subjects = tuple(PAPER_FOLDS[VALIDATION_FOLD])
    test_subjects = tuple(PAPER_FOLDS[TEST_FOLD])
    groups = [set(training_subjects), set(validation_subjects), set(test_subjects)]
    if any(
        groups[left] & groups[right]
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    ):
        raise ValueError("Training, validation, and test patients overlap")

    def summarize(subjects: tuple[str, ...]) -> dict[str, Any]:
        subject_set = set(subjects)
        indices = np.asarray(
            [
                index
                for index, row in enumerate(windows)
                if str(row["subject_id"]) in subject_set
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [int(windows[int(index)]["label"]) for index in indices],
            dtype=np.int64,
        )
        return {
            "subjects": list(subjects),
            "indices": indices,
            "window_count": int(indices.size),
            "normal_windows": int((labels == 0).sum()),
            "ictal_windows": int((labels == 1).sum()),
        }

    training = summarize(training_subjects)
    validation = summarize(validation_subjects)
    test = summarize(test_subjects)
    covered = sum(
        int(partition["window_count"]) for partition in (training, validation, test)
    )
    if covered != len(windows):
        raise ValueError("Fold partition does not cover the complete window manifest")
    return {
        "training_folds": list(TRAIN_FOLDS),
        "validation_fold": VALIDATION_FOLD,
        "test_fold": TEST_FOLD,
        "training": training,
        "validation_sampled_manifest": validation,
        "test_sampled_manifest": test,
    }


def _source_normal_indices(
    *,
    manifest: Mapping[str, Any],
    cache: ChbmitWindowCache,
    training_indices: np.ndarray,
    subjects: Sequence[str],
    maximum: int,
) -> dict[str, np.ndarray]:
    rows_by_subject: dict[str, list[int]] = {subject: [] for subject in subjects}
    for row in training_indices:
        index = int(row)
        subject = str(manifest["windows"][index]["subject_id"])
        if int(cache.labels[index]) == 0 and len(rows_by_subject[subject]) < maximum:
            rows_by_subject[subject].append(index)
    result = {
        subject: np.asarray(indices, dtype=np.int64)
        for subject, indices in rows_by_subject.items()
    }
    if any(indices.size == 0 for indices in result.values()):
        raise ValueError("Every training subject needs at least one normal baseline window")
    return result


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    training: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "fold012_val3_best_epoch8.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": str(model.model_version),
            "model_contract": model.contract(),
            "training_config": dict(training["config"]),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_auprc": float(training["best_score"]),
            "locked_threshold": float(training["best_evaluation"]["threshold"]),
            "state_dict": training["best_state_dict"],
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination, checkpoint_sha256(destination)


def _partition_public(partition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {
            inner_key: inner_value.tolist()
            if isinstance(inner_value, np.ndarray)
            else inner_value
            for inner_key, inner_value in value.items()
        }
        if isinstance(value, Mapping)
        else value
        for key, value in partition.items()
    }


def run_experiment(
    *,
    reference_artifact_path: Path,
    data_root: Path,
    output_dir: Path,
    shared_cache_dir: Path,
    qwen_model_name: str = DEFAULT_QWEN_MODEL,
    local_files_only: bool = True,
    config: MultibranchTrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    settings = config or MultibranchTrainingConfig(max_epochs=8)
    settings.validate()
    if settings.max_epochs != 8:
        raise ValueError("This comparison protocol is fixed to exactly 8 epochs")
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("The 8-epoch multibranch experiment requires CUDA")

    started = time.perf_counter()
    reference_path = runtime_path(reference_artifact_path).resolve()
    output_dir = runtime_path(output_dir).resolve()
    shared_cache_dir = runtime_path(shared_cache_dir).resolve()
    data_root = runtime_path(data_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_json(reference_path)
    manifest = _load_json(runtime_path(reference["source"]["window_manifest"]))
    source_cache = ChbmitWindowCache(runtime_path(reference["source"]["raw_cache"]))
    index = load_chbmit_index(runtime_path(reference["source"]["index"]))
    partition = build_partition(manifest)
    training_indices = np.asarray(partition["training"]["indices"], dtype=np.int64)
    training_subjects = tuple(partition["training"]["subjects"])
    validation_subjects = tuple(partition["validation_sampled_manifest"]["subjects"])
    test_subjects = tuple(partition["test_sampled_manifest"]["subjects"])

    preprocess = S1PreprocessConfig(recipe_id="p0_clip_scale")
    preprocessed_path = build_s1_preprocessed_cache(
        source_cache,
        output_dir=shared_cache_dir / "preprocessed",
        config=preprocess,
    )
    preprocessed = S1PreprocessedCache(preprocessed_path)
    training_dataset = S1ImageDataset(
        preprocessed,
        source_cache,
        manifest,
        training_indices,
        augmentation=None,
        seed=settings.seed,
    )
    validation_cache, validation_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=validation_subjects,
        data_root=data_root,
        shared_dir=shared_cache_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )

    _seed_everything(settings.seed)
    model = build_model(
        qwen_model_name=qwen_model_name,
        local_files_only=local_files_only,
        pretrained_visual_encoder=True,
    )
    training_baselines, training_baseline_summary = (
        compute_subject_log_spectral_baselines(
            model.visual_encoder,
            images=preprocessed.images,
            normal_indices_by_subject=_source_normal_indices(
                manifest=manifest,
                cache=source_cache,
                training_indices=training_indices,
                subjects=training_subjects,
                maximum=settings.enrollment_baseline_windows,
            ),
            device=selected_device,
        )
    )
    validation_baselines, validation_baseline_summary = (
        compute_subject_log_spectral_baselines(
            model.visual_encoder,
            images=validation_cache.images,
            normal_indices_by_subject=_normal_indices(
                validation_cache,
                validation_subjects,
                maximum=settings.enrollment_baseline_windows,
            ),
            device=selected_device,
        )
    )
    training_dataset.subject_baselines = training_baselines
    validation_dataset.subject_baselines = validation_baselines

    def evaluate_epoch(current: torch.nn.Module) -> dict[str, Any]:
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
    locked_threshold = float(training["best_evaluation"]["threshold"])
    checkpoint_path, checkpoint_digest = _save_checkpoint(
        model=model,
        training=training,
        output_dir=output_dir / "checkpoints",
    )

    # Fold 4 is materialized and evaluated only after model and threshold lock.
    test_cache, test_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=test_subjects,
        data_root=data_root,
        shared_dir=shared_cache_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )
    test_baselines, test_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=test_cache.images,
        normal_indices_by_subject=_normal_indices(
            test_cache,
            test_subjects,
            maximum=settings.enrollment_baseline_windows,
        ),
        device=selected_device,
    )
    test_dataset.subject_baselines = test_baselines
    test_evaluation = evaluate_natural_fold(
        model,
        cache=test_cache,
        dataset=test_dataset,
        device=selected_device,
        batch_size=settings.prediction_batch_size,
        minimum_recall=settings.minimum_evaluation_recall,
        threshold=locked_threshold,
    )
    if test_evaluation["threshold_selected_on_this_dataset"]:
        raise RuntimeError("Fold 4 unexpectedly selected its own threshold")

    validation_public = _public_evaluation(training["best_evaluation"])
    test_public = _public_evaluation(test_evaluation)
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
        "schema_version": SCHEMA_VERSION,
        "method": "E1 EfficientNet-Qwen + E2/E3/E4 direct additive residuals",
        "partition": _partition_public(partition),
        "methodology": {
            "checkpoint_selection": "maximum pooled AUPRC on natural Fold 3",
            "threshold_selection": (
                "maximum exact F1 on natural Fold 3 subject to minimum recall "
                f"{settings.minimum_evaluation_recall}"
            ),
            "test_partition": "complete natural Fold 4 timelines",
            "test_evaluations": 1,
            "test_labels_used_for_selection": False,
            "training_augmentation": False,
            "preprocess": preprocess.to_dict(),
            "per_window_channel_zscore": False,
        },
        "model_contract": model.contract(),
        "baseline_protocol": {
            "enrollment_windows_per_patient": settings.enrollment_baseline_windows,
            "selection": "earliest known-normal windows in partition order",
            "training": training_baseline_summary,
            "validation_fold3": validation_baseline_summary,
            "test_fold4": test_baseline_summary,
        },
        "training": public_training,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "portable": "frozen Qwen base weights excluded",
        },
        "validation": validation_public,
        "test": test_public,
        "source": {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": reference["artifact_sha256"],
            "window_manifest_sha256": manifest["window_manifest_sha256"],
            "index_sha256": index["index_sha256"],
            "qwen_model": qwen_model_name,
            "raw_cache": str(source_cache.path),
            "preprocessed_cache": str(preprocessed.path),
            "validation_natural_cache": str(validation_cache.path),
            "test_natural_cache": str(test_cache.path),
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
        output_dir / f"fold012_val3_test4_epoch8_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    result = {
        "artifact": str(artifact_path),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(training["best_epoch"]),
        "locked_threshold": locked_threshold,
        "validation_auroc": validation_public["pooled_metrics"]["auroc"],
        "validation_auprc": validation_public["pooled_metrics"]["auprc"],
        "test_auroc": test_public["pooled_metrics"]["auroc"],
        "test_auprc": test_public["pooled_metrics"]["auprc"],
        "test_f1": test_public["pooled_metrics"]["f1"],
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
        default=Path("artifacts/chbmit/good_multibranch_fold012_val3_test4_epoch8"),
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("artifacts/chbmit/eegvl_multibranch_fullband"),
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    result = run_experiment(
        reference_artifact_path=args.reference_artifact,
        data_root=args.data_root,
        output_dir=args.output_dir,
        shared_cache_dir=args.shared_cache_dir,
        qwen_model_name=args.qwen_model,
        local_files_only=not args.allow_model_download,
        config=MultibranchTrainingConfig(
            max_epochs=8,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
