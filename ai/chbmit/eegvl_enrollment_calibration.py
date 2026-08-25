"""Strict rest-enrollment calibration for frozen CHB-MIT EEG-VL models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ai.v2.lightweight_dataset import write_content_addressed_json
from ai.v2.metrics import evaluate, find_optimal_threshold_exact

from .cache import ChbmitWindowCache
from .deep_timeline import DeepTargetTimeline
from .eeg_continual_eval_cache import NaturalFoldImageCache, NaturalFoldImageDataset
from .eeg_continual_pretrain import (
    PAPER_FOLDS,
    compute_subject_log_spectral_baselines,
    load_server_checkpoint,
    partition_indices,
)
from .eeg_continual_pretrain_model import ServerSTFTConfig
from .eegmamba_b import file_sha256
from .eegvl_m9_model import LoRAConfig
from .eegvl_multibranch_model import (
    EEGVLE1E2E3E4Classifier,
    load_portable_multibranch_state_dict,
)
from .eegvl_s1_data import S1PreprocessedCache
from .eegvl_training import predict_dataset
from .index import canonical_hash


ENROLLMENT_CALIBRATION_SCHEMA_VERSION = "eegvl_enrollment_calibration_v1"
MODEL_E1_E2 = "e1_e2_stft64"
MODEL_MULTIBRANCH = "e1_e2_e3_e4_fullband"


@dataclass(frozen=True)
class EnrollmentCalibrationConfig:
    enrollment_windows: int = 128
    prediction_batch_size: int = 128
    inference_precision: str = "fp32"
    minimum_validation_recall: float = 0.6
    calibration_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975, 0.99, 1.0)
    calibration_margins: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)

    def validate(self) -> None:
        if self.enrollment_windows < 1:
            raise ValueError("enrollment_windows must be positive")
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        if self.inference_precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("inference_precision must be fp32, bf16, or fp16")
        if not 0.0 <= self.minimum_validation_recall <= 1.0:
            raise ValueError("minimum_validation_recall must be in [0, 1]")
        if not self.calibration_quantiles:
            raise ValueError("At least one calibration quantile is required")
        if any(not 0.0 <= value <= 1.0 for value in self.calibration_quantiles):
            raise ValueError("Calibration quantiles must be in [0, 1]")
        if any(value < 0.0 for value in self.calibration_margins):
            raise ValueError("Calibration margins must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def artifact_path(value: str | Path) -> Path:
    """Translate recorded WSL or Windows artifact paths for the current host."""
    text = str(value)
    if os.name == "nt":
        match = re.fullmatch(r"/mnt/([A-Za-z])/(.*)", text)
        if match:
            drive, suffix = match.groups()
            return Path(f"{drive.upper()}:/{suffix}")
    else:
        match = re.fullmatch(r"([A-Za-z]):\\(.*)", text)
        if match:
            drive, suffix = match.groups()
            return Path(f"/mnt/{drive.lower()}/{suffix.replace(chr(92), '/')}")
    return Path(text)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_matching_timelines(
    cache_path: Path,
    cache: NaturalFoldImageCache,
) -> dict[str, DeepTargetTimeline]:
    timeline_root = cache_path.parent.parent / "timelines"
    loaded: dict[str, DeepTargetTimeline] = {}
    for subject_value in cache.metadata["subject_order"]:
        subject = str(subject_value)
        expected = str(cache.metadata["subjects"][subject]["timeline_metadata_sha256"])
        matches: list[DeepTargetTimeline] = []
        for candidate in timeline_root.glob(f"deep_timeline_{subject}_*"):
            timeline = DeepTargetTimeline(candidate)
            if str(timeline.metadata["metadata_sha256"]) == expected:
                matches.append(timeline)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one matching timeline for {subject}, found {len(matches)}"
            )
        loaded[subject] = matches[0]
    return loaded


def _earliest_contiguous_normal_run(
    timeline: DeepTargetTimeline,
    *,
    window_count: int,
) -> np.ndarray:
    if window_count < 1:
        raise ValueError("window_count must be positive")
    stride_samples = int(round(
        float(timeline.metadata["window_config"]["stride_seconds"])
        * float(timeline.metadata["window_config"]["sampling_frequency_hz"])
    ))
    run_start = 0
    run_length = 0
    previous_record = -1
    previous_sample: int | None = None
    for row in range(len(timeline.labels)):
        record = int(timeline.record_indices[row])
        sample = int(timeline.start_samples[row])
        contiguous = (
            previous_sample is not None
            and record == previous_record
            and sample - previous_sample == stride_samples
        )
        if int(timeline.labels[row]) == 0:
            if not contiguous:
                run_start = row
                run_length = 1
            else:
                run_length += 1
            if run_length == window_count:
                return np.arange(run_start, row + 1, dtype=np.int64)
        else:
            run_start = row + 1
            run_length = 0
        previous_record = record
        previous_sample = sample
    raise ValueError(
        f"{timeline.metadata['target_subject']} has no contiguous normal run "
        f"of {window_count} windows"
    )


def build_strict_enrollment_partition(
    cache: NaturalFoldImageCache,
    timelines: Mapping[str, DeepTargetTimeline],
    *,
    enrollment_windows: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Select rest enrollment and return disjoint future-only score rows."""
    enrollment: dict[str, np.ndarray] = {}
    scoring: dict[str, np.ndarray] = {}
    subjects: dict[str, Any] = {}
    for subject_value in cache.metadata["subject_order"]:
        subject = str(subject_value)
        timeline = timelines[subject]
        rows = cache.subject_slice(subject)
        row_start = int(rows.start or 0)
        row_end = int(rows.stop or 0)
        if len(timeline.labels) != row_end - row_start:
            raise ValueError(f"Timeline/cache row mismatch for {subject}")
        cached_labels = np.asarray(cache.labels[rows], dtype=np.uint8)
        if not np.array_equal(cached_labels, np.asarray(timeline.labels)):
            raise ValueError(f"Timeline/cache label mismatch for {subject}")

        local_enrollment = _earliest_contiguous_normal_run(
            timeline,
            window_count=enrollment_windows,
        )
        global_enrollment = local_enrollment + row_start
        score_start = int(global_enrollment[-1]) + 1
        global_scoring = np.arange(score_start, row_end, dtype=np.int64)
        if not len(global_scoring):
            raise ValueError(f"No post-enrollment rows remain for {subject}")
        score_labels = np.asarray(cache.labels[global_scoring], dtype=np.uint8)
        if set(score_labels.tolist()) != {0, 1}:
            raise ValueError(f"Post-enrollment score rows need both classes for {subject}")
        enrollment[subject] = global_enrollment
        scoring[subject] = global_scoring

        local_start = int(local_enrollment[0])
        local_end = int(local_enrollment[-1])
        subjects[subject] = {
            "selection": "earliest_same_edf_contiguous_known_normal_run",
            "enrollment_global_rows": [
                int(global_enrollment[0]),
                int(global_enrollment[-1]) + 1,
            ],
            "enrollment_local_rows": [local_start, local_end + 1],
            "enrollment_window_count": int(len(global_enrollment)),
            "enrollment_duration_seconds": float(
                len(global_enrollment)
                * float(timeline.metadata["window_config"]["stride_seconds"])
            ),
            "record_id": timeline.record_id(local_start),
            "start_sample": int(timeline.start_samples[local_start]),
            "end_start_sample": int(timeline.start_samples[local_end]),
            "all_enrollment_labels_normal": bool(
                np.asarray(cache.labels[global_enrollment]).sum() == 0
            ),
            "discarded_pre_enrollment_windows": local_start,
            "scoring_global_rows": [score_start, row_end],
            "scoring_window_count": int(len(global_scoring)),
            "scoring_normal_windows": int(np.sum(score_labels == 0)),
            "scoring_ictal_windows": int(np.sum(score_labels == 1)),
            "enrollment_identity_sha256": canonical_hash({
                "subject": subject,
                "record_indices": np.asarray(
                    timeline.record_indices[local_enrollment], dtype=np.int64
                ).tolist(),
                "start_samples": np.asarray(
                    timeline.start_samples[local_enrollment], dtype=np.int64
                ).tolist(),
            }),
        }
    enrollment_rows = np.concatenate(list(enrollment.values()))
    scoring_rows = np.concatenate(list(scoring.values()))
    if np.intersect1d(enrollment_rows, scoring_rows).size:
        raise RuntimeError("Enrollment and scoring rows overlap")
    return enrollment, scoring, {
        "selection": "earliest_same_edf_contiguous_known_normal_run",
        "known_normal_source": "CHB-MIT labels used only to simulate a supervised rest session",
        "pre_enrollment_policy": "discarded_from_simulated_stream_and_not_scored",
        "enrollment_rows_excluded_from_scoring": True,
        "subjects": subjects,
        "enrollment_row_sha256": _sha256_array(enrollment_rows),
        "scoring_row_sha256": _sha256_array(scoring_rows),
    }


def _source_reference(result: Mapping[str, Any]) -> dict[str, Any]:
    reference_path = result["source"].get("reference_artifact")
    if reference_path is None:
        return dict(result)
    return _load_json(artifact_path(reference_path))


def compute_population_baseline(
    model: nn.Module,
    result: Mapping[str, Any],
    *,
    device: torch.device,
    enrollment_windows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference = _source_reference(result)
    source = result["source"]
    reference_source = reference["source"]
    split = _load_json(artifact_path(reference["split"]["path"]))
    manifest = _load_json(artifact_path(reference_source["window_manifest"]))
    raw_cache = ChbmitWindowCache(artifact_path(source["raw_cache"]))
    preprocessed = S1PreprocessedCache(artifact_path(source["preprocessed_cache"]))
    train_indices = partition_indices(manifest, split, "source_train")
    training_subjects = tuple(split["partitions"]["source_train"]["subjects"])
    normal_indices = {
        subject: np.asarray([
            int(row)
            for row in train_indices
            if (
                str(manifest["windows"][int(row)]["subject_id"]) == subject
                and int(raw_cache.labels[int(row)]) == 0
            )
        ][:enrollment_windows], dtype=np.int64)
        for subject in training_subjects
    }
    if any(not len(indices) for indices in normal_indices.values()):
        raise ValueError("A source subject has no normal baseline windows")
    baselines, summary = compute_subject_log_spectral_baselines(
        getattr(model, "visual_encoder"),
        images=preprocessed.images,
        normal_indices_by_subject=normal_indices,
        device=device,
    )
    population = np.mean(
        np.stack([baselines[subject] for subject in training_subjects]),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    return population, {
        "definition": "unweighted mean of per-source-patient enrollment baselines",
        "source_subjects": list(training_subjects),
        "source_baselines": summary,
        "population_baseline_sha256": _sha256_array(population),
    }


def compute_enrollment_baselines(
    model: nn.Module,
    cache: NaturalFoldImageCache,
    enrollment: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return compute_subject_log_spectral_baselines(
        getattr(model, "visual_encoder"),
        images=cache.images,
        normal_indices_by_subject=enrollment,
        device=device,
    )


def _load_multibranch_model(
    result: Mapping[str, Any],
) -> tuple[EEGVLE1E2E3E4Classifier, dict[str, Any]]:
    checkpoint = artifact_path(result["checkpoint"]["path"])
    if file_sha256(checkpoint) != str(result["checkpoint"]["sha256"]):
        raise ValueError("Multibranch checkpoint SHA256 mismatch")
    contract = result["model_contract"]
    stft = contract["e1"]["stft"]
    model = EEGVLE1E2E3E4Classifier.from_pretrained(
        qwen_model_name=str(contract["qwen"]["model_name"]),
        local_files_only=True,
        pretrained_visual_encoder=True,
        stft_config=ServerSTFTConfig(**{
            key: stft[key] for key in ServerSTFTConfig.__dataclass_fields__
        }),
        lora_config=LoRAConfig(
            rank=int(contract["lora"]["rank"]),
            alpha=float(contract["lora"]["alpha"]),
            dropout=float(contract["lora"]["dropout"]),
            target_modules=tuple(contract["lora"]["target_modules"]),
        ),
        pooling=str(contract["qwen"]["pooling"]),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_version") != model.model_version:
        raise ValueError("Multibranch checkpoint model version mismatch")
    if model.contract() != dict(payload["model_contract"]):
        raise ValueError("Multibranch checkpoint model contract changed")
    load_portable_multibranch_state_dict(model, payload["state_dict"])
    return model, payload


def load_experiment_model(
    model_name: str,
    result: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    if model_name == MODEL_E1_E2:
        checkpoint = artifact_path(result["checkpoint"]["path"])
        return load_server_checkpoint(
            checkpoint,
            expected_sha256=str(result["checkpoint"]["sha256"]),
        )
    if model_name == MODEL_MULTIBRANCH:
        return _load_multibranch_model(result)
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_with_cache(
    model: nn.Module,
    cache: NaturalFoldImageCache,
    baselines: Mapping[str, np.ndarray],
    *,
    model_name: str,
    checkpoint_sha256: str,
    partition_name: str,
    condition: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    precision: str,
) -> np.ndarray:
    contract = {
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "natural_cache_metadata_sha256": cache.metadata["metadata_sha256"],
        "partition": partition_name,
        "condition": condition,
        "inference_precision": precision,
        "baseline_sha256": {
            subject: _sha256_array(value) for subject, value in baselines.items()
        },
    }
    cache_key = canonical_hash(contract)
    destination = output_dir / "prediction_cache" / f"{cache_key}.npz"
    if destination.is_file():
        payload = np.load(destination)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        if probabilities.shape != (len(cache.labels),):
            raise ValueError("Cached probability shape mismatch")
        if not np.array_equal(payload["labels"], np.asarray(cache.labels)):
            raise ValueError("Cached prediction labels changed")
        return probabilities
    dataset = NaturalFoldImageDataset(cache, subject_baselines=baselines)
    probabilities, labels = predict_dataset(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )
    if not np.array_equal(labels, np.asarray(cache.labels)):
        raise RuntimeError("Prediction labels changed from natural cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez(
        temporary,
        probabilities=probabilities.astype(np.float32, copy=False),
        labels=labels.astype(np.uint8, copy=False),
        contract_json=np.asarray(json.dumps(contract, sort_keys=True)),
    )
    os.replace(temporary, destination)
    return probabilities


def _scoring_arrays(
    cache: NaturalFoldImageCache,
    probabilities: np.ndarray,
    scoring: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate([scoring[str(subject)] for subject in cache.metadata["subject_order"]])
    return (
        np.asarray(cache.labels[rows], dtype=np.int64),
        np.asarray(probabilities[rows], dtype=np.float32),
    )


def evaluate_scoring_partition(
    cache: NaturalFoldImageCache,
    probabilities: np.ndarray,
    scoring: Mapping[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    labels, scores = _scoring_arrays(cache, probabilities, scoring)
    pooled = evaluate(
        labels,
        scores,
        threshold=threshold,
        print_report=False,
        sample_duration_seconds=4.0,
    )
    patients: dict[str, Any] = {}
    for subject_value in cache.metadata["subject_order"]:
        subject = str(subject_value)
        rows = scoring[subject]
        patients[subject] = evaluate(
            np.asarray(cache.labels[rows], dtype=np.int64),
            np.asarray(probabilities[rows], dtype=np.float32),
            threshold=threshold,
            print_report=False,
            sample_duration_seconds=4.0,
        )
    macro = {
        metric: float(np.mean([
            float(values[metric])
            for values in patients.values()
            if values[metric] is not None
        ]))
        for metric in ("auroc", "auprc", "f1", "recall", "false_alarms_per_hour")
    }
    return {
        "threshold": float(threshold),
        "pooled_metrics": pooled,
        "macro_patient_metrics": macro,
        "patient_metrics": patients,
        "scoring_label_sha256": _sha256_array(labels),
        "scoring_probability_sha256": _sha256_array(scores),
        "scoring_window_count": int(len(labels)),
    }


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def apply_patient_score_calibration(
    cache: NaturalFoldImageCache,
    probabilities: np.ndarray,
    enrollment: Mapping[str, np.ndarray],
    *,
    global_threshold: float,
    quantile: float | None,
    margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if quantile is not None and not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    calibrated = np.asarray(probabilities, dtype=np.float64).copy()
    patient_details: dict[str, Any] = {}
    for subject_value in cache.metadata["subject_order"]:
        subject = str(subject_value)
        rows = cache.subject_slice(subject)
        enrollment_scores = np.asarray(probabilities[enrollment[subject]], dtype=np.float64)
        if quantile is None:
            patient_threshold = float(global_threshold)
        else:
            patient_threshold = max(
                float(global_threshold),
                float(np.quantile(enrollment_scores, quantile)) + margin,
            )
            patient_threshold = min(patient_threshold, 1.0 - 1e-6)
        adjusted_logits = (
            _logit(np.asarray(probabilities[rows]))
            - float(_logit(np.asarray([patient_threshold]))[0])
            + float(_logit(np.asarray([global_threshold]))[0])
        )
        calibrated[rows] = 1.0 / (1.0 + np.exp(-adjusted_logits))
        patient_details[subject] = {
            "enrollment_score_min": float(enrollment_scores.min()),
            "enrollment_score_median": float(np.median(enrollment_scores)),
            "enrollment_score_max": float(enrollment_scores.max()),
            "patient_threshold": patient_threshold,
        }
    return calibrated.astype(np.float32), {
        "quantile": quantile,
        "probability_margin": margin,
        "global_threshold": global_threshold,
        "transform": "patient logit offset preserving the selected patient threshold",
        "patients": patient_details,
    }


def select_patient_calibration_rule(
    cache: NaturalFoldImageCache,
    probabilities: np.ndarray,
    enrollment: Mapping[str, np.ndarray],
    scoring: Mapping[str, np.ndarray],
    *,
    global_threshold: float,
    quantiles: Sequence[float],
    margins: Sequence[float],
    minimum_recall: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    rules: list[tuple[float | None, float]] = [(None, 0.0)]
    rules.extend((float(q), float(margin)) for q in quantiles for margin in margins)
    best_key: tuple[float, float, float, float] | None = None
    best_rule: dict[str, Any] | None = None
    for quantile, margin in rules:
        adjusted, calibration = apply_patient_score_calibration(
            cache,
            probabilities,
            enrollment,
            global_threshold=global_threshold,
            quantile=quantile,
            margin=margin,
        )
        evaluation = evaluate_scoring_partition(
            cache,
            adjusted,
            scoring,
            threshold=global_threshold,
        )
        metrics = evaluation["pooled_metrics"]
        recall = float(metrics["recall"] or 0.0)
        f1 = float(metrics["f1"] or 0.0)
        false_alarms = float(metrics["false_alarms_per_hour"] or float("inf"))
        auprc = float(metrics["auprc"] or 0.0)
        eligible = recall >= minimum_recall
        candidate = {
            "quantile": quantile,
            "probability_margin": margin,
            "eligible": eligible,
            "pooled_metrics": metrics,
            "patient_thresholds": {
                subject: values["patient_threshold"]
                for subject, values in calibration["patients"].items()
            },
        }
        candidates.append(candidate)
        key = (f1, -false_alarms, auprc, recall)
        if eligible and (best_key is None or key > best_key):
            best_key = key
            best_rule = candidate
    if best_rule is None:
        raise RuntimeError("No patient calibration rule meets validation recall")
    return {
        "quantile": best_rule["quantile"],
        "probability_margin": best_rule["probability_margin"],
        "selection_partition": "fold4_post_enrollment_only",
        "selection_objective": "max_f1_then_min_fa_per_hour_then_auprc_then_recall",
        "minimum_recall": minimum_recall,
    }, candidates


def _condition_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_metrics = candidate["pooled_metrics"]
    baseline_metrics = baseline["pooled_metrics"]
    return {
        metric: (
            None
            if candidate_metrics[metric] is None or baseline_metrics[metric] is None
            else float(candidate_metrics[metric]) - float(baseline_metrics[metric])
        )
        for metric in ("auroc", "auprc", "f1", "recall", "precision", "false_alarms_per_hour")
    }


def run_model_calibration(
    model_name: str,
    result_path: Path,
    *,
    output_dir: Path,
    config: EnrollmentCalibrationConfig,
    device: torch.device,
) -> dict[str, Any]:
    result_path = artifact_path(result_path).resolve()
    result = _load_json(result_path)
    model, _ = load_experiment_model(model_name, result)
    model.to(device)
    checkpoint_digest = str(result["checkpoint"]["sha256"])
    validation_cache_path = artifact_path(result["source"]["validation_natural_cache"])
    outer_cache_path = artifact_path(result["source"]["outer_natural_cache"])
    validation_cache = NaturalFoldImageCache(validation_cache_path)
    outer_cache = NaturalFoldImageCache(outer_cache_path)
    validation_timelines = _load_matching_timelines(validation_cache_path, validation_cache)
    outer_timelines = _load_matching_timelines(outer_cache_path, outer_cache)
    val_enrollment, val_scoring, val_protocol = build_strict_enrollment_partition(
        validation_cache,
        validation_timelines,
        enrollment_windows=config.enrollment_windows,
    )
    outer_enrollment, outer_scoring, outer_protocol = build_strict_enrollment_partition(
        outer_cache,
        outer_timelines,
        enrollment_windows=config.enrollment_windows,
    )

    population, population_summary = compute_population_baseline(
        model,
        result,
        device=device,
        enrollment_windows=config.enrollment_windows,
    )
    validation_patient_baselines, validation_baseline_summary = compute_enrollment_baselines(
        model,
        validation_cache,
        val_enrollment,
        device=device,
    )
    outer_patient_baselines, outer_baseline_summary = compute_enrollment_baselines(
        model,
        outer_cache,
        outer_enrollment,
        device=device,
    )
    validation_population = {
        str(subject): population for subject in validation_cache.metadata["subject_order"]
    }
    outer_population = {
        str(subject): population for subject in outer_cache.metadata["subject_order"]
    }
    prediction_args = {
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_digest,
        "output_dir": output_dir,
        "device": device,
        "batch_size": config.prediction_batch_size,
        "precision": config.inference_precision,
    }
    val_b0_probabilities = _predict_with_cache(
        model,
        validation_cache,
        validation_population,
        partition_name="fold4",
        condition="b0_population_baseline",
        **prediction_args,
    )
    val_b1_probabilities = _predict_with_cache(
        model,
        validation_cache,
        validation_patient_baselines,
        partition_name="fold4",
        condition="b1_patient_rest_baseline",
        **prediction_args,
    )
    b0_val_labels, b0_val_scores = _scoring_arrays(
        validation_cache, val_b0_probabilities, val_scoring
    )
    b1_val_labels, b1_val_scores = _scoring_arrays(
        validation_cache, val_b1_probabilities, val_scoring
    )
    b0_threshold = find_optimal_threshold_exact(
        b0_val_labels,
        b0_val_scores,
        min_recall=config.minimum_validation_recall,
    )
    b1_threshold = find_optimal_threshold_exact(
        b1_val_labels,
        b1_val_scores,
        min_recall=config.minimum_validation_recall,
    )
    val_b0 = evaluate_scoring_partition(
        validation_cache, val_b0_probabilities, val_scoring, threshold=b0_threshold
    )
    val_b1 = evaluate_scoring_partition(
        validation_cache, val_b1_probabilities, val_scoring, threshold=b1_threshold
    )
    selected_rule, candidate_rules = select_patient_calibration_rule(
        validation_cache,
        val_b1_probabilities,
        val_enrollment,
        val_scoring,
        global_threshold=b1_threshold,
        quantiles=config.calibration_quantiles,
        margins=config.calibration_margins,
        minimum_recall=config.minimum_validation_recall,
    )
    val_b2_probabilities, val_b2_calibration = apply_patient_score_calibration(
        validation_cache,
        val_b1_probabilities,
        val_enrollment,
        global_threshold=b1_threshold,
        quantile=selected_rule["quantile"],
        margin=float(selected_rule["probability_margin"]),
    )
    val_b2 = evaluate_scoring_partition(
        validation_cache, val_b2_probabilities, val_scoring, threshold=b1_threshold
    )

    outer_b0_probabilities = _predict_with_cache(
        model,
        outer_cache,
        outer_population,
        partition_name="fold0",
        condition="b0_population_baseline",
        **prediction_args,
    )
    outer_b1_probabilities = _predict_with_cache(
        model,
        outer_cache,
        outer_patient_baselines,
        partition_name="fold0",
        condition="b1_patient_rest_baseline",
        **prediction_args,
    )
    outer_b2_probabilities, outer_b2_calibration = apply_patient_score_calibration(
        outer_cache,
        outer_b1_probabilities,
        outer_enrollment,
        global_threshold=b1_threshold,
        quantile=selected_rule["quantile"],
        margin=float(selected_rule["probability_margin"]),
    )
    outer_b0 = evaluate_scoring_partition(
        outer_cache, outer_b0_probabilities, outer_scoring, threshold=b0_threshold
    )
    outer_b1 = evaluate_scoring_partition(
        outer_cache, outer_b1_probabilities, outer_scoring, threshold=b1_threshold
    )
    outer_b2 = evaluate_scoring_partition(
        outer_cache, outer_b2_probabilities, outer_scoring, threshold=b1_threshold
    )
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "source_result": str(result_path),
        "source_result_sha256": file_sha256(result_path),
        "checkpoint": {
            "path": str(artifact_path(result["checkpoint"]["path"]).resolve()),
            "sha256": checkpoint_digest,
        },
        "protocol": {
            "validation": val_protocol,
            "outer_test": outer_protocol,
        },
        "baseline": {
            "b0_population": population_summary,
            "b1_validation_enrollment": validation_baseline_summary,
            "b1_outer_enrollment": outer_baseline_summary,
        },
        "threshold_selection": {
            "partition": "Fold 4 post-enrollment rows only",
            "minimum_recall": config.minimum_validation_recall,
            "b0_population_threshold": float(b0_threshold),
            "b1_patient_baseline_threshold": float(b1_threshold),
            "b2_rule": selected_rule,
            "b2_candidates": candidate_rules,
        },
        "validation": {
            "b0_population_baseline": val_b0,
            "b1_patient_rest_baseline": val_b1,
            "b2_patient_score_calibration": val_b2,
            "b2_calibration": val_b2_calibration,
        },
        "outer_test": {
            "b0_population_baseline": outer_b0,
            "b1_patient_rest_baseline": outer_b1,
            "b2_patient_score_calibration": outer_b2,
            "b2_calibration": outer_b2_calibration,
            "deltas_from_b0": {
                "b1": _condition_delta(outer_b1, outer_b0),
                "b2": _condition_delta(outer_b2, outer_b0),
            },
            "b2_increment_over_b1": _condition_delta(outer_b2, outer_b1),
        },
    }


def run_enrollment_calibration(
    *,
    e1_e2_result: Path,
    multibranch_result: Path,
    output_dir: Path,
    config: EnrollmentCalibrationConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    settings = config or EnrollmentCalibrationConfig()
    settings.validate()
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.type != "cuda":
        raise RuntimeError("Enrollment calibration inference requires CUDA")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_results = {}
    for model_name, result_path in (
        (MODEL_E1_E2, e1_e2_result),
        (MODEL_MULTIBRANCH, multibranch_result),
    ):
        model_results[model_name] = run_model_calibration(
            model_name,
            result_path,
            output_dir=output_dir,
            config=settings,
            device=selected_device,
        )
    body = {
        "schema_version": ENROLLMENT_CALIBRATION_SCHEMA_VERSION,
        "objective": "Phase A strict enrollment protocol and Phase B B0/B1/B2 calibration",
        "config": settings.to_dict(),
        "folds": {
            "selection": {"fold": 4, "subjects": list(PAPER_FOLDS[4])},
            "outer_test": {"fold": 0, "subjects": list(PAPER_FOLDS[0])},
        },
        "conditions": {
            "b0": "frozen model plus source-population E2 baseline",
            "b1": "frozen model plus new-patient contiguous rest E2 baseline",
            "b2": "B1 plus Fold-4-selected patient score calibration",
        },
        "models": model_results,
        "runtime": {
            "device": str(selected_device),
            "gpu_name": torch.cuda.get_device_name(selected_device),
            "torch_version": torch.__version__,
            "inference_precision": settings.inference_precision,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    artifact_digest = canonical_hash(body)
    artifact = {**body, "artifact_sha256": artifact_digest}
    path = write_content_addressed_json(
        artifact,
        output_dir / f"enrollment_calibration_{artifact_digest[:12]}.json",
        hash_field="artifact_sha256",
    )
    return {**artifact, "artifact": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e1-e2-result",
        type=Path,
        default=Path(
            "artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/"
            "fold0_pretrain_c27817a49668.json"
        ),
    )
    parser.add_argument(
        "--multibranch-result",
        type=Path,
        default=Path(
            "artifacts/chbmit/eegvl_multibranch_fullband/"
            "fold0_e1_e2_e3_e4_f1660457394b.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/eegvl_enrollment_calibration"),
    )
    parser.add_argument("--enrollment-windows", type=int, default=128)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument(
        "--inference-precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    args = parser.parse_args()
    result = run_enrollment_calibration(
        e1_e2_result=args.e1_e2_result,
        multibranch_result=args.multibranch_result,
        output_dir=args.output_dir,
        config=EnrollmentCalibrationConfig(
            enrollment_windows=args.enrollment_windows,
            prediction_batch_size=args.prediction_batch_size,
            inference_precision=args.inference_precision,
        ),
    )
    print(json.dumps({
        "artifact": result["artifact"],
        "artifact_sha256": result["artifact_sha256"],
        "duration_seconds": result["duration_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
