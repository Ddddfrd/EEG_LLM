"""Run Scheme C on the patient split reported by the EEGMamba paper.

Only the patient partition changes from the retained Scheme C experiment:
chb01-chb19 train, chb20-chb21 validation, chb22-chb23 final test, and chb24
unused. Model architecture, preprocessing, sampling, E2 calibration and the
five-epoch training budget remain unchanged.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

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
from ai.chbmit.eegvl_multibranch_model import (
    EEGVLE1E2E3E4Classifier,
    checkpoint_sha256,
    load_portable_multibranch_state_dict,
)
from ai.chbmit.eegvl_s1_data import (
    S1ImageDataset,
    S1PreprocessConfig,
    S1PreprocessedCache,
    build_s1_preprocessed_cache,
)
from ai.chbmit.index import canonical_hash
from ai.chbmit.windows import WindowConfig, build_window_manifest, load_chbmit_index
from ai.v2.lightweight_dataset import write_content_addressed_json

from .model import DEFAULT_QWEN_MODEL, build_model
from .train_19_vs_chb10_14 import EXPECTED_SUBJECTS
from .train_scheme_c_aligned import (
    CALIBRATION_FRACTION,
    MAX_CALIBRATION_WINDOWS,
    MAX_WINDOWS_PER_PATIENT,
    _candidate,
    _cap_manifest_per_patient,
    _natural_calibration_indices,
    _public_training,
    _sampled_calibration_indices,
)


SCHEMA_VERSION = "scheme_c_eegmamba_patient_split_v1"
TRAINING_SUBJECTS = tuple(f"chb{number:02d}" for number in range(1, 20))
VALIDATION_SUBJECTS = ("chb20", "chb21")
TEST_SUBJECTS = ("chb22", "chb23")
UNUSED_SUBJECTS = ("chb24",)


def build_eegmamba_partition(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the paper's 19/2/2 patient split while explicitly excluding chb24."""
    windows = list(manifest["windows"])
    manifest_subjects = {str(row["subject_id"]) for row in windows}
    if manifest_subjects != set(EXPECTED_SUBJECTS):
        raise ValueError("Manifest must contain exactly chb01 through chb24")
    groups = {
        "training": TRAINING_SUBJECTS,
        "validation": VALIDATION_SUBJECTS,
        "test": TEST_SUBJECTS,
        "unused": UNUSED_SUBJECTS,
    }
    flattened = [subject for subjects in groups.values() for subject in subjects]
    if len(flattened) != len(set(flattened)):
        raise ValueError("EEGMamba patient split groups overlap")
    if set(flattened) != set(EXPECTED_SUBJECTS):
        raise ValueError("EEGMamba patient split does not cover chb01-chb24")

    def summarize(subjects: tuple[str, ...]) -> dict[str, Any]:
        selected = set(subjects)
        indices = np.asarray(
            [index for index, row in enumerate(windows) if str(row["subject_id"]) in selected],
            dtype=np.int64,
        )
        labels = np.asarray(
            [int(windows[int(index)]["label"]) for index in indices],
            dtype=np.int64,
        )
        return {
            "subjects": list(subjects),
            "indices": indices,
            "window_count": int(len(indices)),
            "normal_windows": int(np.sum(labels == 0)),
            "ictal_windows": int(np.sum(labels == 1)),
        }

    result = {name: summarize(subjects) for name, subjects in groups.items()}
    if sum(int(values["window_count"]) for values in result.values()) != len(windows):
        raise ValueError("EEGMamba split rows do not cover the sampled manifest")
    return result


def _save_candidate(
    *,
    model: torch.nn.Module,
    training: Mapping[str, Any],
    metric: str,
    output_dir: Path,
) -> tuple[Path, str]:
    candidate = _candidate(training, metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"train_chb01_19_best_{metric}.pt"
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": str(model.model_version),
            "model_contract": model.contract(),
            "training_config": dict(training["config"]),
            "selection_partition": list(VALIDATION_SUBJECTS),
            "selection_metric": metric,
            "best_epoch": candidate["epoch"],
            "best_validation_score": candidate["score"],
            "selected_validation_threshold": float(candidate["evaluation"]["threshold"]),
            "state_dict": candidate["state_dict"],
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination, checkpoint_sha256(destination)


def _write_report(result: Mapping[str, Any], destination: Path) -> None:
    lines = [
        "# Scheme C on the EEGMamba Patient Split",
        "",
        "Patient split: chb01-chb19 train, chb20-chb21 validation, "
        "chb22-chb23 final test, chb24 unused.",
        "",
        "Only the patient partition differs from the retained five-epoch Scheme C run.",
        "",
        "## Final chb22-chb23 results",
        "",
        "| Validation selection | Epoch | Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("auroc", "auprc"):
        checkpoint = result["checkpoints"][metric]
        validation = result["validation_evaluations"][metric]["pooled_metrics"]
        test = result["test_evaluations"][metric]["pooled_metrics"]
        lines.append(
            f"| {metric}-best | {checkpoint['epoch']} | {validation['auroc']:.4f} | "
            f"{validation['auprc']:.4f} | {test['auroc']:.4f} | "
            f"{test['auprc']:.4f} | {test['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- chb22-chb23 labels are not used for epoch or threshold selection.",
            "- The test threshold is selected on chb20-chb21 and then held fixed.",
            "- This matches the EEGMamba patient IDs, not its 16-channel/10-second preprocessing or foundation pretraining.",
            "- E2 still uses each patient's earliest known-normal windows, as in the retained Scheme C protocol.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


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
    model_builder: Callable[..., EEGVLE1E2E3E4Classifier] = build_model,
    stft_config_override: ServerSTFTConfig | None = None,
) -> dict[str, Any]:
    settings = config or MultibranchTrainingConfig(
        max_epochs=5,
        enrollment_baseline_windows=MAX_CALIBRATION_WINDOWS,
        checkpoint_metric="auroc",
    )
    settings.validate()
    if settings.max_epochs != 5:
        raise ValueError("EEGMamba-split Scheme C is fixed to exactly 5 epochs")
    if settings.checkpoint_metric != "auroc":
        raise ValueError("Primary checkpoint selection must use validation AUROC")
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("EEGMamba-split Scheme C requires CUDA")

    started = time.perf_counter()
    reference_path = runtime_path(reference_artifact_path).resolve()
    data_root = runtime_path(data_root).resolve()
    output_dir = runtime_path(output_dir).resolve()
    shared_cache_dir = runtime_path(shared_cache_dir).resolve()
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
        output_dir / "data" / f"scheme_c_windows_{manifest['window_manifest_sha256'][:12]}.json",
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
    partition = build_eegmamba_partition(manifest)
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
    validation_cache, validation_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=VALIDATION_SUBJECTS,
        data_root=data_root,
        shared_dir=shared_cache_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )
    test_cache, test_dataset = _build_fullband_natural_cache(
        index=index,
        subjects=TEST_SUBJECTS,
        data_root=data_root,
        shared_dir=shared_cache_dir,
        preprocess=preprocess,
        seed=settings.seed,
    )

    _seed_everything(settings.seed)
    stft_config = stft_config_override or ServerSTFTConfig(
        source_channels=20,
        eeg_channels=20,
        n_fft=64,
        win_length=64,
        hop_length=32,
        zscore_input=False,
    )
    model = model_builder(
        qwen_model_name=qwen_model_name,
        local_files_only=local_files_only,
        pretrained_visual_encoder=True,
        stft_config_override=stft_config,
    )
    training_baselines, training_baseline_summary = compute_subject_log_spectral_baselines(
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
    validation_baselines, validation_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=validation_cache.images,
        normal_indices_by_subject=_natural_calibration_indices(
            validation_cache,
            VALIDATION_SUBJECTS,
        ),
        device=selected_device,
    )
    test_baselines, test_baseline_summary = compute_subject_log_spectral_baselines(
        model.visual_encoder,
        images=test_cache.images,
        normal_indices_by_subject=_natural_calibration_indices(
            test_cache,
            TEST_SUBJECTS,
        ),
        device=selected_device,
    )
    training_dataset.subject_baselines = training_baselines
    validation_dataset.subject_baselines = validation_baselines
    test_dataset.subject_baselines = test_baselines

    def evaluate_validation(current: torch.nn.Module) -> dict[str, Any]:
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
        evaluate_epoch=evaluate_validation,
        device=selected_device,
        config=settings,
    )
    checkpoints: dict[str, Any] = {}
    validation_evaluations: dict[str, Any] = {}
    test_evaluations: dict[str, Any] = {}
    for metric in ("auroc", "auprc"):
        candidate = _candidate(training, metric)
        load_portable_multibranch_state_dict(model, candidate["state_dict"])
        checkpoint_path, checkpoint_digest = _save_candidate(
            model=model,
            training=training,
            metric=metric,
            output_dir=output_dir / "checkpoints",
        )
        test_evaluation = evaluate_natural_fold(
            model,
            cache=test_cache,
            dataset=test_dataset,
            device=selected_device,
            batch_size=settings.prediction_batch_size,
            minimum_recall=settings.minimum_evaluation_recall,
            threshold=float(candidate["evaluation"]["threshold"]),
        )
        checkpoints[metric] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "epoch": candidate["epoch"],
            "validation_score": candidate["score"],
            "validation_threshold": float(candidate["evaluation"]["threshold"]),
        }
        validation_evaluations[metric] = _public_evaluation(candidate["evaluation"])
        test_evaluations[metric] = _public_evaluation(test_evaluation)

    body = {
        "schema_version": SCHEMA_VERSION,
        "method": "Scheme C E1+E2+E3+E4 on EEGMamba patient split",
        "partition": {
            "training_subjects": list(TRAINING_SUBJECTS),
            "validation_subjects": list(VALIDATION_SUBJECTS),
            "test_subjects": list(TEST_SUBJECTS),
            "unused_subjects": list(UNUSED_SUBJECTS),
            "training_windows": int(partition["training"]["window_count"]),
            "validation_windows": int(validation_cache.metadata["window_count"]),
            "test_windows": int(test_cache.metadata["window_count"]),
            "validation_equals_test": False,
        },
        "scheme_c_contract": {
            "direct_20_channels": index["direct20_contract"],
            "stft": model.visual_encoder.config.to_dict(),
            "training_sampling": {
                "window_seconds": 4.0,
                "stride_seconds": 1.0,
                "positive_negative_ratio": "3:7 per patient",
                "maximum_windows_per_patient": MAX_WINDOWS_PER_PATIENT,
            },
            "natural_evaluation_stride_seconds": 4.0,
            "e2_calibration": {
                "fraction": CALIBRATION_FRACTION,
                "maximum": MAX_CALIBRATION_WINDOWS,
                "selection": "earliest available known-normal windows",
                "training": training_baseline_summary,
                "validation": validation_baseline_summary,
                "test": test_baseline_summary,
            },
            "training_augmentation": False,
            "preprocess": preprocess.to_dict(),
            "per_window_channel_zscore": False,
            "epochs": 5,
        },
        "methodology": {
            "checkpoint_selection": "chb20-chb21 only",
            "threshold_selection": "chb20-chb21 only",
            "final_test": "chb22-chb23 complete natural timeline, evaluated once per selected checkpoint",
            "test_labels_used_for_selection": False,
            "paper_alignment_scope": "patient IDs only",
        },
        "model_contract": model.contract(),
        "training": _public_training(training),
        "checkpoints": checkpoints,
        "validation_evaluations": validation_evaluations,
        "test_evaluations": test_evaluations,
        "source": {
            "reference_artifact": str(reference_path),
            "base_index_sha256": base_index["index_sha256"],
            "direct20_index_sha256": index["index_sha256"],
            "window_manifest": str(manifest_path),
            "window_manifest_sha256": manifest["window_manifest_sha256"],
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
            "duration_seconds": time.perf_counter() - started,
        },
    }
    artifact_digest = canonical_hash(body)
    artifact = {**body, "artifact_sha256": artifact_digest}
    artifact_path = write_content_addressed_json(
        artifact,
        output_dir / f"scheme_c_eegmamba_split_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    report_path = output_dir / "SCHEME_C_EEGMAMBA_SPLIT_RESULTS.md"
    _write_report(artifact, report_path)
    primary_test = test_evaluations["auroc"]["pooled_metrics"]
    result = {
        "artifact": str(artifact_path),
        "report": str(report_path),
        "checkpoints": checkpoints,
        "auroc_best_test_auroc": primary_test["auroc"],
        "auroc_best_test_auprc": primary_test["auprc"],
        "duration_seconds": body["runtime"]["duration_seconds"],
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
        default=Path("artifacts/chbmit/scheme_c_eegmamba_split"),
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
