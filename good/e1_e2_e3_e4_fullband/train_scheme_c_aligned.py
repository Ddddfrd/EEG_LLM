"""Run the five-change Scheme C alignment on train-19 versus chb10-chb14."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.chbmit.cache import ChbmitWindowCache, build_window_caches
from ai.chbmit.direct20 import build_direct20_index
from ai.chbmit.eeg_continual_pretrain import (
    compute_subject_log_spectral_baselines,
    evaluate_natural_fold,
)
from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegmamba_b_experiment import (
    _load_json,
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
from ai.chbmit.windows import WindowConfig, build_window_manifest, load_chbmit_index

from .model import DEFAULT_QWEN_MODEL, build_model
from .train_19_vs_chb10_14 import (
    EXPECTED_SUBJECTS,
    TRAINING_SUBJECTS,
    VALIDATION_TEST_SUBJECTS,
    build_partition,
)


SCHEMA_VERSION = "good_multibranch_scheme_c_aligned_v1"
CALIBRATION_FRACTION = 0.2
MAX_CALIBRATION_WINDOWS = 4000
MAX_WINDOWS_PER_PATIENT = 20_000


def _cap_manifest_per_patient(
    manifest: Mapping[str, Any],
    *,
    maximum: int,
) -> dict[str, Any]:
    """Apply a deterministic patient cap while retaining the original ordering."""
    rows = list(manifest["windows"])
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        subject = str(row["subject_id"])
        if counts[subject] >= maximum:
            continue
        kept.append(dict(row))
        counts[subject] += 1
    body = {
        key: value
        for key, value in manifest.items()
        if key != "window_manifest_sha256"
    }
    body["windows"] = kept
    statistics = dict(body["statistics"])
    statistics["selected_windows_before_patient_cap"] = len(rows)
    statistics["selected_windows"] = len(kept)
    statistics["patient_cap"] = maximum
    statistics["selected_ictal"] = sum(int(row["label"]) == 1 for row in kept)
    statistics["selected_normal"] = sum(int(row["label"]) == 0 for row in kept)
    statistics["post_cap_by_subject"] = {
        subject: {
            "windows": sum(str(row["subject_id"]) == subject for row in kept),
            "ictal": sum(
                str(row["subject_id"]) == subject and int(row["label"]) == 1
                for row in kept
            ),
            "normal": sum(
                str(row["subject_id"]) == subject and int(row["label"]) == 0
                for row in kept
            ),
        }
        for subject in EXPECTED_SUBJECTS
    }
    body["statistics"] = statistics
    return {**body, "window_manifest_sha256": canonical_hash(body)}


def _calibration_count(normal_count: int) -> int:
    if normal_count < 1:
        raise ValueError("Calibration requires at least one normal window")
    return min(
        MAX_CALIBRATION_WINDOWS,
        max(1, int(math.ceil(normal_count * CALIBRATION_FRACTION))),
    )


def _sampled_calibration_indices(
    *,
    manifest: Mapping[str, Any],
    cache: ChbmitWindowCache,
    allowed_indices: np.ndarray,
    subjects: Sequence[str],
) -> dict[str, np.ndarray]:
    allowed = {int(index) for index in np.asarray(allowed_indices, dtype=np.int64)}
    result: dict[str, np.ndarray] = {}
    for subject in subjects:
        normal = [
            index
            for index, row in enumerate(manifest["windows"])
            if (
                index in allowed
                and str(row["subject_id"]) == subject
                and int(cache.labels[index]) == 0
            )
        ]
        result[subject] = np.asarray(
            normal[: _calibration_count(len(normal))],
            dtype=np.int64,
        )
    return result


def _natural_calibration_indices(
    cache: Any,
    subjects: Sequence[str],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for subject in subjects:
        subject_slice = cache.subject_slice(subject)
        rows = np.arange(subject_slice.start, subject_slice.stop, dtype=np.int64)
        normal = rows[np.asarray(cache.labels[subject_slice]) == 0]
        result[subject] = normal[: _calibration_count(len(normal))]
    return result


def _candidate(training: Mapping[str, Any], metric: str) -> dict[str, Any]:
    if str(training["best_metric"]) == metric:
        return {
            "metric": metric,
            "epoch": int(training["best_epoch"]),
            "score": float(training["best_score"]),
            "evaluation": training["best_evaluation"],
            "state_dict": training["best_state_dict"],
        }
    if str(training["secondary_metric"]) == metric:
        return {
            "metric": metric,
            "epoch": int(training["secondary_epoch"]),
            "score": float(training["secondary_score"]),
            "evaluation": training["secondary_evaluation"],
            "state_dict": training["secondary_state_dict"],
        }
    raise ValueError(f"Training did not track checkpoint metric {metric}")


def _save_candidate(
    *,
    model: torch.nn.Module,
    training: Mapping[str, Any],
    metric: str,
    output_dir: Path,
) -> tuple[Path, str]:
    candidate = _candidate(training, metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"train19_chb10_14_best_{metric}.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": str(model.model_version),
            "model_contract": model.contract(),
            "training_config": dict(training["config"]),
            "selection_metric": metric,
            "best_epoch": candidate["epoch"],
            "best_score": candidate["score"],
            "selected_threshold": float(candidate["evaluation"]["threshold"]),
            "state_dict": candidate["state_dict"],
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination, checkpoint_sha256(destination)


def _public_training(training: Mapping[str, Any]) -> dict[str, Any]:
    private = {
        "best_state_dict",
        "best_evaluation",
        "secondary_state_dict",
        "secondary_evaluation",
    }
    return {key: value for key, value in training.items() if key not in private}


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
    settings = config or MultibranchTrainingConfig(
        max_epochs=5,
        enrollment_baseline_windows=MAX_CALIBRATION_WINDOWS,
        checkpoint_metric="auroc",
    )
    settings.validate()
    if settings.max_epochs != 5:
        raise ValueError("Scheme C alignment is fixed to exactly 5 epochs")
    if settings.checkpoint_metric != "auroc":
        raise ValueError("Scheme C primary checkpoint must use AUROC")
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Scheme C alignment requires CUDA")

    started = time.perf_counter()
    reference_path = runtime_path(reference_artifact_path).resolve()
    output_dir = runtime_path(output_dir).resolve()
    shared_cache_dir = runtime_path(shared_cache_dir).resolve()
    data_root = runtime_path(data_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_json(reference_path)
    base_index = load_chbmit_index(runtime_path(reference["source"]["index"]))
    index = build_direct20_index(base_index)
    manifest = _cap_manifest_per_patient(
        build_window_manifest(
            index,
            config=WindowConfig(
                window_seconds=4.0,
                stride_seconds=1.0,
                ictal_overlap_fraction=0.5,
                seizure_guard_seconds=0.0,
                normal_to_ictal_ratio=7.0 / 3.0,
                sampling_frequency_hz=256,
                sampling_seed=settings.seed,
            ),
        ),
        maximum=MAX_WINDOWS_PER_PATIENT,
    )
    manifest_path = write_content_addressed_json(
        manifest,
        output_dir / "data" / (
            f"scheme_c_windows_{manifest['window_manifest_sha256'][:12]}.json"
        ),
        hash_field="window_manifest_sha256",
    )
    raw_cache_path = build_window_caches(
        index,
        manifest,
        data_root=data_root,
        output_dir=shared_cache_dir / "raw_windows",
        progress_every=2000,
    )
    source_cache = ChbmitWindowCache(raw_cache_path)
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
        stft_config_override=ServerSTFTConfig(
            source_channels=20,
            eeg_channels=20,
            n_fft=64,
            win_length=64,
            hop_length=32,
            zscore_input=False,
        ),
    )
    training_baselines, training_baseline_summary = (
        compute_subject_log_spectral_baselines(
            model.visual_encoder,
            images=preprocessed.images,
            normal_indices_by_subject=_sampled_calibration_indices(
                manifest=manifest,
                cache=source_cache,
                allowed_indices=training_indices,
                subjects=TRAINING_SUBJECTS,
            ),
            device=selected_device,
        )
    )
    development_baselines, development_baseline_summary = (
        compute_subject_log_spectral_baselines(
            model.visual_encoder,
            images=development_cache.images,
            normal_indices_by_subject=_natural_calibration_indices(
                development_cache,
                VALIDATION_TEST_SUBJECTS,
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
    checkpoints: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for metric in ("auroc", "auprc"):
        checkpoint_path, checkpoint_digest = _save_candidate(
            model=model,
            training=training,
            metric=metric,
            output_dir=output_dir / "checkpoints",
        )
        candidate = _candidate(training, metric)
        checkpoints[metric] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "epoch": candidate["epoch"],
            "score": candidate["score"],
        }
        evaluations[metric] = _public_evaluation(candidate["evaluation"])

    body = {
        "schema_version": SCHEMA_VERSION,
        "method": "Scheme C aligned E1+E2+E3+E4",
        "partition": {
            "training_subjects": list(TRAINING_SUBJECTS),
            "validation_test_subjects": list(VALIDATION_TEST_SUBJECTS),
            "training_windows": int(partition["training"]["window_count"]),
            "development_windows": int(development_cache.metadata["window_count"]),
            "validation_equals_test": True,
        },
        "five_changes": {
            "direct_20_channels": index["direct20_contract"],
            "stft": model.visual_encoder.config.to_dict(),
            "training_sampling": {
                "window_seconds": 4.0,
                "stride_seconds": 1.0,
                "positive_negative_ratio": "3:7 per patient",
                "maximum_windows_per_patient": MAX_WINDOWS_PER_PATIENT,
                "manifest_statistics": manifest["statistics"],
            },
            "e2_calibration": {
                "fraction": CALIBRATION_FRACTION,
                "maximum": MAX_CALIBRATION_WINDOWS,
                "selection": "earliest available known-normal windows",
                "training": training_baseline_summary,
                "development": development_baseline_summary,
            },
            "dual_checkpoint_selection": ["auroc", "auprc"],
        },
        "methodology": {
            "validation_equals_test": True,
            "evaluation": "complete natural chb10-chb14 pooled probabilities and labels",
            "training_augmentation": False,
            "preprocess": preprocess.to_dict(),
            "per_window_channel_zscore": False,
        },
        "model_contract": model.contract(),
        "training": _public_training(training),
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "source": {
            "reference_artifact": str(reference_path),
            "base_index_sha256": base_index["index_sha256"],
            "direct20_index_sha256": index["index_sha256"],
            "window_manifest": str(manifest_path),
            "window_manifest_sha256": manifest["window_manifest_sha256"],
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
        output_dir / f"scheme_c_aligned_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    primary = evaluations["auroc"]["pooled_metrics"]
    result = {
        "artifact": str(artifact_path),
        "checkpoints": checkpoints,
        "auroc_best_epoch": checkpoints["auroc"]["epoch"],
        "auprc_best_epoch": checkpoints["auprc"]["epoch"],
        "development_auroc": primary["auroc"],
        "development_auprc": primary["auprc"],
        "development_f1": primary["f1"],
        "training_windows": body["partition"]["training_windows"],
        "development_windows": body["partition"]["development_windows"],
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
        default=Path("artifacts/chbmit/good_multibranch_scheme_c_aligned"),
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("artifacts/chbmit/good_multibranch_scheme_c_aligned/cache"),
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
            enrollment_baseline_windows=MAX_CALIBRATION_WINDOWS,
            checkpoint_metric="auroc",
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
