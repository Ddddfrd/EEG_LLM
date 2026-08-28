"""Deterministic probability and explicit-action alarm evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .contracts import EventInterval, ProbabilityTimeline


@dataclass(frozen=True)
class AlarmConfig:
    """Temporal voting and alarm refractory configuration."""

    vote_k: int = 2
    vote_n: int = 3
    refractory_seconds: float = 60.0

    def validate(self) -> None:
        if self.vote_n < 1 or not 1 <= self.vote_k <= self.vote_n:
            raise ValueError("alarm vote must satisfy 1 <= vote_k <= vote_n")
        if self.refractory_seconds < 0:
            raise ValueError("refractory_seconds must be non-negative")


@dataclass
class _AlarmEpisode:
    record_index: int
    start_seconds: float
    end_seconds: float


def _binary_actions(actions: Sequence[int] | np.ndarray, *, rows: int) -> np.ndarray:
    values = np.asarray(actions).reshape(-1)
    if values.shape != (rows,):
        raise ValueError("alarm action count does not match timeline")
    if not np.isin(values, (0, 1, False, True)).all():
        raise ValueError("alarm actions must be binary")
    return values.astype(bool, copy=False)


def voted_actions(
    timeline: ProbabilityTimeline,
    raw_predictions: Sequence[int] | np.ndarray,
    *,
    vote_k: int,
    vote_n: int,
) -> np.ndarray:
    """Apply k-of-n voting to raw boolean predictions with record-local history."""
    timeline.validate()
    raw = _binary_actions(raw_predictions, rows=timeline.row_count)
    if vote_n < 1 or not 1 <= vote_k <= vote_n:
        raise ValueError("alarm vote must satisfy 1 <= vote_k <= vote_n")
    voted = np.zeros(timeline.row_count, dtype=bool)
    stride_samples = int(round(timeline.stride_seconds * timeline.sampling_frequency_hz))
    history: list[bool] = []
    previous_record = -1
    previous_start: int | None = None
    for row, prediction in enumerate(raw):
        record = int(timeline.record_indices[row])
        start = int(timeline.start_samples[row])
        contiguous = (
            record == previous_record
            and previous_start is not None
            and start - previous_start == stride_samples
        )
        if not contiguous:
            history.clear()
        history.append(bool(prediction))
        history = history[-vote_n:]
        voted[row] = len(history) == vote_n and sum(history) >= vote_k
        previous_record = record
        previous_start = start
    return voted


def actions_from_probabilities(
    timeline: ProbabilityTimeline,
    *,
    threshold: float,
    vote_k: int = 2,
    vote_n: int = 3,
) -> np.ndarray:
    """Convert frozen probabilities to voted actions with record-local history."""
    timeline.validate()
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    config = AlarmConfig(vote_k=vote_k, vote_n=vote_n, refractory_seconds=0)
    config.validate()
    raw = timeline.probabilities >= threshold
    return voted_actions(timeline, raw, vote_k=vote_k, vote_n=vote_n)


def _alarm_episodes(
    timeline: ProbabilityTimeline,
    actions: np.ndarray,
    *,
    refractory_seconds: float,
) -> tuple[list[_AlarmEpisode], list[_AlarmEpisode]]:
    candidates: list[_AlarmEpisode] = []
    for row in np.flatnonzero(actions):
        record_index = int(timeline.record_indices[row])
        start_seconds = float(timeline.start_samples[row]) / timeline.sampling_frequency_hz
        end_seconds = start_seconds + timeline.window_seconds
        if (
            candidates
            and candidates[-1].record_index == record_index
            and start_seconds <= candidates[-1].end_seconds
        ):
            candidates[-1].end_seconds = max(candidates[-1].end_seconds, end_seconds)
            continue
        candidates.append(_AlarmEpisode(record_index, start_seconds, end_seconds))

    accepted: list[_AlarmEpisode] = []
    previous_start_by_record: dict[int, float] = {}
    for episode in candidates:
        previous = previous_start_by_record.get(episode.record_index)
        if previous is not None and episode.start_seconds - previous < refractory_seconds:
            continue
        accepted.append(episode)
        previous_start_by_record[episode.record_index] = episode.start_seconds
    return candidates, accepted


def _overlaps(alarm: _AlarmEpisode, event: EventInterval) -> bool:
    return (
        alarm.record_index == event.record_index
        and min(alarm.end_seconds, event.end_seconds)
        > max(alarm.start_seconds, event.start_seconds)
    )


def _normal_monitoring_hours(timeline: ProbabilityTimeline) -> float:
    total_seconds = 0.0
    current_record = -1
    interval_start = 0.0
    interval_end = 0.0
    for row in np.flatnonzero(timeline.labels == 0):
        record = int(timeline.record_indices[row])
        start = float(timeline.start_samples[row]) / timeline.sampling_frequency_hz
        end = start + timeline.window_seconds
        if record != current_record or start > interval_end:
            if current_record >= 0:
                total_seconds += interval_end - interval_start
            current_record = record
            interval_start = start
            interval_end = end
        else:
            interval_end = max(interval_end, end)
    if current_record >= 0:
        total_seconds += interval_end - interval_start
    return total_seconds / 3600.0


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    positive = labels == 1
    negative = ~positive
    predicted = predictions.astype(bool)
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    return {
        "observation_count": int(labels.size),
        "sensitivity": recall,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "precision": precision,
        "f1": 2 * tp / f1_denominator if f1_denominator else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_alarm_actions(
    timeline: ProbabilityTimeline,
    alarm_actions: Sequence[int] | np.ndarray,
    *,
    refractory_seconds: float = 60.0,
) -> dict[str, Any]:
    """Evaluate explicit causal alarm actions without thresholding probabilities."""
    timeline.validate()
    if refractory_seconds < 0:
        raise ValueError("refractory_seconds must be non-negative")
    actions = _binary_actions(alarm_actions, rows=timeline.row_count)
    candidates, accepted = _alarm_episodes(
        timeline,
        actions,
        refractory_seconds=refractory_seconds,
    )
    detected: set[str] = set()
    latencies: list[float] = []
    false_alarm_count = 0
    for alarm in accepted:
        matching = [event for event in timeline.events if _overlaps(alarm, event)]
        if not matching:
            false_alarm_count += 1
            continue
        for event in matching:
            if event.event_id in detected:
                continue
            detected.add(event.event_id)
            latencies.append(max(0.0, alarm.start_seconds - event.start_seconds))

    normal_hours = _normal_monitoring_hours(timeline)
    event_count = len(timeline.events)
    detected_count = len(detected)
    return {
        "subject_id": timeline.subject_id,
        "refractory_seconds": refractory_seconds,
        "action_metrics": _binary_metrics(timeline.labels, actions),
        "event_metrics": {
            "event_count": event_count,
            "detected_events": detected_count,
            "missed_events": event_count - detected_count,
            "event_sensitivity": detected_count / event_count if event_count else None,
            "detected_event_ids": sorted(detected),
            "latencies_seconds": sorted(latencies),
            "false_alarm_episodes": false_alarm_count,
            "normal_monitoring_hours": normal_hours,
            "false_alarms_per_hour": (
                false_alarm_count / normal_hours if normal_hours else None
            ),
            "mean_detection_latency_seconds": (
                float(np.mean(latencies)) if latencies else None
            ),
            "median_detection_latency_seconds": (
                float(np.median(latencies)) if latencies else None
            ),
            "candidate_alarm_episode_count": len(candidates),
            "accepted_alarm_episode_count": len(accepted),
        },
    }


def evaluate_probability_policy(
    timeline: ProbabilityTimeline,
    *,
    threshold: float,
    alarm_config: AlarmConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the existing threshold, k-of-n vote, and refractory policy."""
    config = alarm_config or AlarmConfig()
    config.validate()
    actions = actions_from_probabilities(
        timeline,
        threshold=threshold,
        vote_k=config.vote_k,
        vote_n=config.vote_n,
    )
    action_result = evaluate_alarm_actions(
        timeline,
        actions,
        refractory_seconds=config.refractory_seconds,
    )
    raw_predictions = timeline.probabilities >= threshold
    window_metrics = _binary_metrics(timeline.labels, raw_predictions)
    if np.unique(timeline.labels).size == 2:
        window_metrics["auroc"] = float(
            roc_auc_score(timeline.labels, timeline.probabilities)
        )
        window_metrics["auprc"] = float(
            average_precision_score(timeline.labels, timeline.probabilities)
        )
    else:
        window_metrics["auroc"] = None
        window_metrics["auprc"] = None
    return {
        "subject_id": timeline.subject_id,
        "threshold": threshold,
        "alarm_config": {
            "vote_k": config.vote_k,
            "vote_n": config.vote_n,
            "refractory_seconds": config.refractory_seconds,
        },
        "window_metrics": window_metrics,
        "event_metrics": action_result["event_metrics"],
    }
