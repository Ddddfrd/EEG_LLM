"""Leakage-resistant metrics for the frozen Phase 2 evaluation protocol."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation_protocol import EVALUATION_PROTOCOL_VERSION


BOOTSTRAP_METRICS = (
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "auprc",
    "brier",
    "ece",
    "false_alarms_per_hour",
)


def _validated_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Metrics require at least one observation")
    if labels.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob must contain only finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("y_prob must contain probabilities in [0, 1]")
    return labels, probabilities


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    min_recall: float = 0.80,
) -> float:
    """Select a threshold on calibration data only."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")

    best: tuple[float, float] | None = None
    fallback: tuple[float, float] = (-1.0, 0.5)
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if recall > fallback[0]:
            fallback = (recall, float(threshold))
        candidate = (f1, -float(threshold))
        if recall >= min_recall and (best is None or candidate > best):
            best = candidate
    return -best[1] if best is not None else fallback[1]


def find_optimal_threshold_exact(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    min_recall: float = 0.80,
) -> float:
    """Select from observed probability boundaries without a coarse grid."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")
    candidates = np.unique(np.concatenate(([0.0], probabilities)))
    best: tuple[float, float] | None = None
    for threshold in candidates:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        recall = tp / (tp + fn)
        if recall < min_recall:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        candidate = (f1, -float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No threshold satisfies the requested recall")
    return -best[1]


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    if bins < 2:
        raise ValueError("ece_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        observed = float(labels[selected].mean())
        error += float(selected.mean()) * abs(confidence - observed)
    return error


def evaluate(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float | None = None,
    print_report: bool = True,
    *,
    allow_threshold_selection: bool = False,
    sample_duration_seconds: float = 1.0,
    observation_durations_seconds: Sequence[float] | np.ndarray | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate fixed probabilities without hiding undefined metrics.

    ``threshold=None`` is rejected by default so final test labels cannot be
    used accidentally. Calibration callers must opt in explicitly.
    """
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if threshold is None:
        if not allow_threshold_selection:
            raise ValueError(
                "A fixed threshold is required; select it on calibration data"
            )
        threshold = find_optimal_threshold(labels, probabilities)
        threshold_source = "selected_from_current_calibration_data"
    else:
        threshold = float(threshold)
        threshold_source = "provided_fixed_threshold"
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(sample_duration_seconds) or sample_duration_seconds <= 0:
        raise ValueError("sample_duration_seconds must be positive and finite")
    if observation_durations_seconds is None:
        durations = np.full(labels.size, sample_duration_seconds, dtype=np.float64)
    else:
        durations = np.asarray(
            observation_durations_seconds, dtype=np.float64
        ).reshape(-1)
        if durations.shape != labels.shape:
            raise ValueError(
                "observation_durations_seconds must match y_true shape"
            )
        if not np.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError("observation durations must be positive and finite")

    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & negative))

    statuses: dict[str, str] = {}

    def ratio(name: str, numerator: int, denominator: int, reason: str) -> float | None:
        if denominator == 0:
            statuses[name] = f"uncomputable:{reason}"
            return None
        statuses[name] = "computed"
        return numerator / denominator

    sensitivity = ratio("sensitivity", tp, tp + fn, "no_positive_labels")
    specificity = ratio("specificity", tn, tn + fp, "no_negative_labels")
    precision = ratio("precision", tp, tp + fp, "no_predicted_positives")
    if sensitivity is None or precision is None or sensitivity + precision == 0:
        f1: float | None = None
        statuses["f1"] = (
            "uncomputable:missing_precision_or_sensitivity"
            if sensitivity is None or precision is None
            else "uncomputable:precision_and_sensitivity_are_zero"
        )
    else:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
        statuses["f1"] = "computed"

    if positive.any() and negative.any():
        auroc: float | None = float(roc_auc_score(labels, probabilities))
        auprc: float | None = float(average_precision_score(labels, probabilities))
        statuses["auroc"] = "computed"
        statuses["auprc"] = "computed"
    else:
        auroc = None
        auprc = None
        reason = "single_class_labels"
        statuses["auroc"] = f"uncomputable:{reason}"
        statuses["auprc"] = f"uncomputable:{reason}"

    brier = float(np.mean((probabilities - labels) ** 2))
    ece = _expected_calibration_error(labels, probabilities, ece_bins)
    statuses["brier"] = "computed"
    statuses["ece"] = "computed"
    negative_hours = float(durations[negative].sum()) / 3600.0
    if negative_hours == 0.0:
        false_alarms_per_hour: float | None = None
        statuses["false_alarms_per_hour"] = (
            "uncomputable:no_negative_monitoring_time"
        )
    else:
        false_alarms_per_hour = fp / negative_hours
        statuses["false_alarms_per_hour"] = "computed"

    metrics: dict[str, Any] = {
        "status": (
            "complete"
            if all(value == "computed" for value in statuses.values())
            else "partial"
        ),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "observation_count": int(labels.size),
        "sample_duration_seconds": float(sample_duration_seconds),
        "negative_observation_hours": negative_hours,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "seizure_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "auc_roc": auroc,
        "auprc": auprc,
        "auc_pr": auprc,
        "brier": brier,
        "ece": ece,
        "false_alarms_per_hour": false_alarms_per_hour,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "metric_status": statuses,
    }
    if print_report:
        printable = lambda value: "uncomputable" if value is None else f"{value:.4f}"
        print(f"\nThreshold: {threshold:.2f} ({threshold_source})")
        print(
            f"Sensitivity: {printable(sensitivity)}  |  "
            f"Specificity: {printable(specificity)}"
        )
        print(f"Precision: {printable(precision)}  |  F1: {printable(f1)}")
        print(f"AUROC: {printable(auroc)}  |  AUPRC: {printable(auprc)}")
        print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    return metrics


def _aggregate_homogeneous_groups(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_ids: Sequence[str],
    sample_duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[str(group_id)].append(index)
    group_labels: list[int] = []
    group_probabilities: list[float] = []
    group_durations: list[float] = []
    for group_id in sorted(grouped):
        indices = np.asarray(grouped[group_id], dtype=np.int64)
        unique_labels = set(labels[indices].tolist())
        if len(unique_labels) != 1:
            raise ValueError(f"Group {group_id!r} contains mixed labels")
        group_labels.append(unique_labels.pop())
        group_probabilities.append(float(probabilities[indices].max()))
        group_durations.append(float(indices.size) * sample_duration_seconds)
    return (
        np.asarray(group_labels, dtype=np.int64),
        np.asarray(group_probabilities, dtype=np.float64),
        np.asarray(group_durations, dtype=np.float64),
    )


def _bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
    *,
    threshold: float,
    sample_duration_seconds: float,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if resamples <= 0:
        return {
            name: {"status": "not_requested", "lower": None, "upper": None}
            for name in BOOTSTRAP_METRICS
        }
    unique_groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    if unique_groups.size < 2:
        return {
            name: {
                "status": "uncomputable:fewer_than_two_groups",
                "lower": None,
                "upper": None,
            }
            for name in BOOTSTRAP_METRICS
        }
    group_indices = {
        group: np.flatnonzero(group_ids == group) for group in unique_groups
    }
    samples: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        selected_groups = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        indices = np.concatenate([group_indices[group] for group in selected_groups])
        result = evaluate(
            labels[indices],
            probabilities[indices],
            threshold=threshold,
            print_report=False,
            sample_duration_seconds=sample_duration_seconds,
        )
        for name in BOOTSTRAP_METRICS:
            value = result[name]
            if value is not None:
                samples[name].append(float(value))

    intervals: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        if not values:
            intervals[name] = {
                "status": "uncomputable:no_valid_bootstrap_resamples",
                "lower": None,
                "upper": None,
                "valid_resamples": 0,
                "requested_resamples": resamples,
            }
            continue
        intervals[name] = {
            "status": "computed" if len(values) == resamples else "partial",
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "valid_resamples": len(values),
            "requested_resamples": resamples,
        }
    return intervals


def evaluate_protocol(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    group_ids: Sequence[str],
    patient_ids: Sequence[int] | None = None,
    sample_duration_seconds: float = 1.0,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Report segment, episode/group, patient, and clustered-bootstrap results."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if len(group_ids) != labels.size:
        raise ValueError("group_ids must have one value per observation")
    group_array = np.asarray(list(map(str, group_ids)), dtype=object)
    segment = evaluate(
        labels,
        probabilities,
        threshold=threshold,
        print_report=False,
        sample_duration_seconds=sample_duration_seconds,
    )
    episode_labels, episode_probabilities, episode_durations = (
        _aggregate_homogeneous_groups(
            labels,
            probabilities,
            group_ids,
            sample_duration_seconds,
        )
    )
    episode = evaluate(
        episode_labels,
        episode_probabilities,
        threshold=threshold,
        print_report=False,
        observation_durations_seconds=episode_durations,
    )

    per_patient: dict[str, dict[str, Any]] = {}
    if patient_ids is not None:
        if len(patient_ids) != labels.size:
            raise ValueError("patient_ids must have one value per observation")
        patient_array = np.asarray(patient_ids, dtype=np.int64)
        for patient_id in sorted(set(patient_array.tolist())):
            selected = patient_array == patient_id
            per_patient[str(patient_id)] = evaluate(
                labels[selected],
                probabilities[selected],
                threshold=threshold,
                print_report=False,
                sample_duration_seconds=sample_duration_seconds,
            )

    return {
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "threshold": float(threshold),
        "threshold_source": "external_calibration_scope",
        "segment_level": segment,
        "episode_level": episode,
        "patient_level": per_patient,
        "bootstrap_95_ci": _bootstrap_confidence_intervals(
            labels,
            probabilities,
            group_array,
            threshold=threshold,
            sample_duration_seconds=sample_duration_seconds,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
