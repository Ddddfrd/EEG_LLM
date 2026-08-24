"""Train five epochs on all CHB-MIT patients except chb10-chb14.

The same complete natural chb10-chb14 timeline is used for epoch selection,
threshold selection, and reported evaluation. This is a development experiment,
not an unbiased held-out test.
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
from .train_fold012_val3_test4 import _source_normal_indices


SCHEMA_VERSION = "good_multibranch_train19_chb10_14_epoch5_v1"
EXPECTED_SUBJECTS = tuple(f"chb{number:02d}" for number in range(1, 25))
VALIDATION_TEST_SUBJECTS = tuple(f"chb{number:02d}" for number in range(10, 15))
TRAINING_SUBJECTS = tuple(
    subject for subject in EXPECTED_SUBJECTS if subject not in VALIDATION_TEST_SUBJECTS
)


def build_partition(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the requested 19-patient train and five-patient development split."""
    windows = list(manifest["windows"])
    manifest_subjects = {str(row["subject_id"]) for row in windows}
    if manifest_subjects != set(EXPECTED_SUBJECTS):
        raise ValueError("Manifest must contain exactly chb01 through chb24")
    if set(TRAINING_SUBJECTS) & set(VALIDATION_TEST_SUBJECTS):
        raise ValueError("Training and chb10-chb14 patients overlap")

    def summarize(subjects: tuple[str, ...]) -> dict[str, Any]:
        selected = set(subjects)
        indices = np.asarray(
            [
                index
                for index, row in enumerate(windows)
                if str(row["subject_id"]) in selected
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

    training = summarize(TRAINING_SUBJECTS)
    validation_test = summarize(VALIDATION_TEST_SUBJECTS)
    if int(training["window_count"]) + int(validation_test["window_count"]) != len(windows):
        raise ValueError("Requested split does not cover the complete manifest")
    return {
        "training": training,
        "validation_test_sampled_manifest": validation_test,
        "validation_equals_test": True,
    }


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    training: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "train19_chb10_14_best_epoch5.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": str(model.model_version),
            "model_contract": model.contract(),
            "training_config": dict(training["config"]),
            "best_epoch": int(training["best_epoch"]),
            "best_development_auprc": float(training["best_score"]),
            "selected_threshold": float(training["best_evaluation"]["threshold"]),
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
    settings = config or MultibranchTrainingConfig(max_epochs=5)
    settings.validate()
    if settings.max_epochs != 5:
        raise ValueError("This comparison protocol is fixed to exactly 5 epochs")
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("The five-epoch full-band experiment requires CUDA")

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
    development_cache, development_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=VALIDATION_TEST_SUBJECTS,
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
                subjects=TRAINING_SUBJECTS,
                maximum=settings.enrollment_baseline_windows,
            ),
            device=selected_device,
        )
    )
    development_baselines, development_baseline_summary = (
        compute_subject_log_spectral_baselines(
            model.visual_encoder,
            images=development_cache.images,
            normal_indices_by_subject=_normal_indices(
                development_cache,
                VALIDATION_TEST_SUBJECTS,
                maximum=settings.enrollment_baseline_windows,
            ),
            device=selected_device,
        )
    )
    training_dataset.subject_baselines = training_baselines
    development_dataset.subject_baselines = development_baselines

    def evaluate_epoch(current: torch.nn.Module) -> dict[str, Any]:
        return evaluate_natural_fold(
            current,
            cache=development_cache,
            dataset=development_dataset,
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
    checkpoint_path, checkpoint_digest = _save_checkpoint(
        model=model,
        training=training,
        output_dir=output_dir / "checkpoints",
    )

    development_public = _public_evaluation(training["best_evaluation"])
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
            "training_subjects": list(TRAINING_SUBJECTS),
            "validation_subjects": list(VALIDATION_TEST_SUBJECTS),
            "test_subjects": list(VALIDATION_TEST_SUBJECTS),
            "validation_equals_test": True,
            "independent_test_available": False,
            "checkpoint_selection": "maximum pooled natural AUPRC on chb10-chb14",
            "threshold_selection": (
                "maximum exact F1 on chb10-chb14 subject to minimum recall "
                f"{settings.minimum_evaluation_recall}"
            ),
            "evaluation_label": "development_only",
            "training_augmentation": False,
            "preprocess": preprocess.to_dict(),
            "per_window_channel_zscore": False,
        },
        "model_contract": model.contract(),
        "baseline_protocol": {
            "enrollment_windows_per_patient": settings.enrollment_baseline_windows,
            "selection": "earliest known-normal windows in partition order",
            "training": training_baseline_summary,
            "chb10_chb14": development_baseline_summary,
        },
        "training": public_training,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "portable": "frozen Qwen base weights excluded",
        },
        "validation_test_development_evaluation": development_public,
        "source": {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": reference["artifact_sha256"],
            "window_manifest_sha256": manifest["window_manifest_sha256"],
            "index_sha256": index["index_sha256"],
            "qwen_model": qwen_model_name,
            "raw_cache": str(source_cache.path),
            "preprocessed_cache": str(preprocessed.path),
            "development_natural_cache": str(development_cache.path),
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
        output_dir / f"train19_chb10_14_epoch5_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    pooled = development_public["pooled_metrics"]
    result = {
        "artifact": str(artifact_path),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(training["best_epoch"]),
        "selected_threshold": float(development_public["threshold"]),
        "development_auroc": pooled["auroc"],
        "development_auprc": pooled["auprc"],
        "development_f1": pooled["f1"],
        "duration_seconds": body["duration_seconds"],
        "validation_equals_test": True,
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
        default=Path("artifacts/chbmit/good_multibranch_train19_chb10_14_epoch5"),
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
            max_epochs=5,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=args.micro_batch_size,
            prediction_batch_size=args.prediction_batch_size,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
