"""Cross-subject aggregation of alarm evaluation results.

Pooled event metrics sum event and alarm counts over monitoring hours, so a
patient with a long clean recording contributes its false alarms fully. The
selection objective is declared before any search runs and frozen before the
final test (EEG_RL_ALARM_POLICY_PLAN.md section L1-A.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import ProbabilityTimeline
from .evaluator import evaluate_alarm_actions

EVENT_KEYS = (
    "event_count",
    "detected_events",
    "missed_events",
    "false_alarm_episodes",
    "candidate_alarm_episode_count",
    "accepted_alarm_episode_count",
)


@dataclass(frozen=True)
class SelectionObjective:
    """Declares the operating point used to compare alarm policies.

    ``J = event_sensitivity - lambda_fa * false_alarms_per_hour
          - lambda_latency * mean_latency / normalizer``.
    """

    lambda_fa: float
    lambda_latency: float
    latency_normalizer_seconds: float = 60.0
    minimum_event_sensitivity: float = 0.8
    minimum_patient_event_sensitivity: float | None = None

    def validate(self) -> None:
        if self.lambda_fa < 0 or self.lambda_latency < 0:
            raise ValueError("objective coefficients must be non-negative")
        if self.latency_normalizer_seconds <= 0:
            raise ValueError("latency normalizer must be positive")
        if not 0 <= self.minimum_event_sensitivity <= 1:
            raise ValueError("minimum event sensitivity must be in [0, 1]")
        if self.minimum_patient_event_sensitivity is not None and not (
            0 <= self.minimum_patient_event_sensitivity <= 1
        ):
            raise ValueError("minimum patient event sensitivity must be in [0, 1]")

    def score(self, pooled: Mapping[str, Any]) -> float | None:
        """Return J, or ``None`` when sensitivity or false-alarm rate is undefined."""
        sensitivity = pooled["event_sensitivity"]
        fa_per_hour = pooled["false_alarms_per_hour"]
        if sensitivity is None or fa_per_hour is None:
            return None
        latency = pooled["mean_detection_latency_seconds"] or 0.0
        return float(
            sensitivity
            - self.lambda_fa * fa_per_hour
            - self.lambda_latency * latency / self.latency_normalizer_seconds
        )

    def passes_guardrail(
        self,
        pooled: Mapping[str, Any],
        patient_summary: Mapping[str, Any] | None = None,
    ) -> bool:
        sensitivity = pooled["event_sensitivity"]
        if sensitivity is None or sensitivity < self.minimum_event_sensitivity:
            return False
        if self.minimum_patient_event_sensitivity is None:
            return True
        if patient_summary is None:
            raise ValueError("patient summary is required by the patient guardrail")
        worst = patient_summary["worst_event_sensitivity"]
        return worst is not None and worst >= self.minimum_patient_event_sensitivity


def _sum_confusion(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for result in results:
        matrix = result["action_metrics"]["confusion_matrix"]
        for key in totals:
            totals[key] += int(matrix[key])
    return totals


def _ratios(matrix: Mapping[str, int]) -> dict[str, float | None]:
    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "sensitivity": ratio(matrix["tp"], matrix["tp"] + matrix["fn"]),
        "specificity": ratio(matrix["tn"], matrix["tn"] + matrix["fp"]),
        "precision": ratio(matrix["tp"], matrix["tp"] + matrix["fp"]),
        "f1": ratio(2 * matrix["tp"], 2 * matrix["tp"] + matrix["fp"] + matrix["fn"]),
    }


def _pooled_event_metrics(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    totals = {key: 0 for key in EVENT_KEYS}
    latencies: list[float] = []
    total_hours = 0.0
    for result in results:
        events = result["event_metrics"]
        for key in EVENT_KEYS:
            totals[key] += int(events[key])
        hours = events["normal_monitoring_hours"]
        total_hours += float(hours) if hours is not None else 0.0
        latencies.extend(float(value) for value in events["latencies_seconds"])
    detected = totals["detected_events"]
    return {
        "event_count": totals["event_count"],
        "detected_events": detected,
        "missed_events": totals["missed_events"],
        "event_sensitivity": detected / totals["event_count"]
        if totals["event_count"]
        else None,
        "detected_event_ids": sorted(
            event_id
            for result in results
            for event_id in result["event_metrics"]["detected_event_ids"]
        ),
        "latencies_seconds": sorted(latencies),
        "false_alarm_episodes": totals["false_alarm_episodes"],
        "normal_monitoring_hours": total_hours,
        "false_alarms_per_hour": (
            totals["false_alarm_episodes"] / total_hours if total_hours > 0 else None
        ),
        "mean_detection_latency_seconds": (
            float(np.mean(latencies)) if latencies else None
        ),
        "median_detection_latency_seconds": (
            float(np.median(latencies)) if latencies else None
        ),
        "candidate_alarm_episode_count": totals["candidate_alarm_episode_count"],
        "accepted_alarm_episode_count": totals["accepted_alarm_episode_count"],
    }


def _pooled_action_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = _sum_confusion(results)
    observation_count = sum(
        int(result["action_metrics"]["observation_count"]) for result in results
    )
    return {"observation_count": observation_count, "confusion_matrix": matrix,
            **_ratios(matrix)}


def _patient_summary(per_subject: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and worst per-patient values over subjects that have events."""
    sensitivities = [
        float(result["event_metrics"]["event_sensitivity"])
        for result in per_subject.values()
        if result["event_metrics"]["event_sensitivity"] is not None
    ]
    fa_rates = [
        float(result["event_metrics"]["false_alarms_per_hour"])
        for result in per_subject.values()
        if result["event_metrics"]["false_alarms_per_hour"] is not None
    ]
    latencies = [
        float(result["event_metrics"]["mean_detection_latency_seconds"])
        for result in per_subject.values()
        if result["event_metrics"]["mean_detection_latency_seconds"] is not None
    ]
    return {
        "subjects": sorted(per_subject),
        "mean_event_sensitivity": (
            float(np.mean(sensitivities)) if sensitivities else None
        ),
        "worst_event_sensitivity": (
            float(np.min(sensitivities)) if sensitivities else None
        ),
        "mean_false_alarms_per_hour": float(np.mean(fa_rates)) if fa_rates else None,
        "worst_false_alarms_per_hour": float(np.max(fa_rates)) if fa_rates else None,
        "mean_detection_latency_seconds": (
            float(np.mean(latencies)) if latencies else None
        ),
    }


def evaluate_cohort(
    timelines: Mapping[str, ProbabilityTimeline],
    actions_by_subject: Mapping[str, Sequence[int] | np.ndarray],
    *,
    refractory_seconds: float,
) -> dict[str, Any]:
    """Evaluate explicit alarm actions per subject and aggregate the cohort."""
    if set(timelines) != set(actions_by_subject):
        raise ValueError("actions must be provided for exactly the cohort subjects")
    per_subject = {
        subject: evaluate_alarm_actions(
            timelines[subject],
            actions_by_subject[subject],
            refractory_seconds=refractory_seconds,
        )
        for subject in sorted(timelines)
    }
    results = [per_subject[subject] for subject in sorted(per_subject)]
    return {
        "subjects": sorted(per_subject),
        "per_subject": per_subject,
        "pooled": {
            **_pooled_event_metrics(results),
            "action_metrics": _pooled_action_metrics(results),
        },
        "patient_summary": _patient_summary(per_subject),
    }
