"""Window- and event-level evaluation on full CHB-MIT target timelines."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ai.v2.metrics import evaluate

from .timeline_cache import TargetTimelineCache


@dataclass(frozen=True)
class AlarmConfig:
    vote_k: int = 2
    vote_n: int = 3
    refractory_seconds: float = 60.0

    def validate(self) -> None:
        if self.vote_n < 1 or not 1 <= self.vote_k <= self.vote_n:
            raise ValueError("Alarm vote must satisfy 1 <= k <= n")
        if self.refractory_seconds < 0:
            raise ValueError("Refractory duration must be non-negative")


def _validate_probabilities(
    timeline: TargetTimelineCache,
    probabilities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.shape != timeline.labels.shape:
        raise ValueError("Probability count does not match target timeline")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return values


def _voted_predictions(
    timeline: TargetTimelineCache,
    raw_predictions: np.ndarray,
    *,
    config: AlarmConfig,
) -> np.ndarray:
    config.validate()
    voted = np.zeros_like(raw_predictions, dtype=bool)
    stride_samples = int(
        round(
            float(timeline.metadata["window_config"]["stride_seconds"])
            * float(timeline.metadata["window_config"]["sampling_frequency_hz"])
        )
    )
    history: deque[bool] = deque(maxlen=config.vote_n)
    previous_record = -1
    previous_start: int | None = None
    for row in range(len(raw_predictions)):
        record = int(timeline.record_indices[row])
        start = int(timeline.start_samples[row])
        contiguous = (
            record == previous_record
            and previous_start is not None
            and start - previous_start == stride_samples
        )
        if not contiguous:
            history.clear()
        history.append(bool(raw_predictions[row]))
        voted[row] = (
            len(history) == config.vote_n
            and sum(history) >= config.vote_k
        )
        previous_record = record
        previous_start = start
    return voted


def _alarm_episodes(
    timeline: TargetTimelineCache,
    voted: np.ndarray,
    *,
    config: AlarmConfig,
) -> list[dict[str, Any]]:
    sampling_frequency = float(
        timeline.metadata["window_config"]["sampling_frequency_hz"]
    )
    window_seconds = float(
        timeline.metadata["window_config"]["window_seconds"]
    )
    episodes: list[dict[str, Any]] = []
    for row in np.flatnonzero(voted):
        record_index = int(timeline.record_indices[row])
        start_seconds = float(timeline.start_samples[row]) / sampling_frequency
        end_seconds = start_seconds + window_seconds
        event_index = int(timeline.event_indices[row])
        if (
            episodes
            and episodes[-1]["record_index"] == record_index
            and start_seconds <= float(episodes[-1]["end_seconds"])
        ):
            episodes[-1]["end_seconds"] = max(
                float(episodes[-1]["end_seconds"]), end_seconds
            )
            if event_index:
                episodes[-1]["event_indices"].add(event_index)
            continue
        episodes.append({
            "record_index": record_index,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "event_indices": {event_index} if event_index else set(),
        })

    accepted: list[dict[str, Any]] = []
    last_alarm_start_by_record: dict[int, float] = {}
    for episode in episodes:
        record_index = int(episode["record_index"])
        start_seconds = float(episode["start_seconds"])
        previous = last_alarm_start_by_record.get(record_index)
        if (
            previous is not None
            and start_seconds - previous < config.refractory_seconds
        ):
            continue
        accepted.append(episode)
        last_alarm_start_by_record[record_index] = start_seconds
    return accepted


def _intervals_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    return min(first_end, second_end) > max(first_start, second_start)


def _normal_monitoring_hours(timeline: TargetTimelineCache) -> float:
    sampling_frequency = float(
        timeline.metadata["window_config"]["sampling_frequency_hz"]
    )
    window_seconds = float(
        timeline.metadata["window_config"]["window_seconds"]
    )
    total_seconds = 0.0
    current_record = -1
    interval_start = 0.0
    interval_end = 0.0
    for row in np.flatnonzero(np.asarray(timeline.labels) == 0):
        record = int(timeline.record_indices[row])
        start = float(timeline.start_samples[row]) / sampling_frequency
        end = start + window_seconds
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


def evaluate_target_timeline(
    timeline: TargetTimelineCache,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    alarm_config: AlarmConfig | None = None,
) -> dict[str, Any]:
    values = _validate_probabilities(timeline, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1]")
    config = alarm_config or AlarmConfig()
    raw_predictions = values >= threshold
    voted = _voted_predictions(timeline, raw_predictions, config=config)
    alarms = _alarm_episodes(timeline, voted, config=config)
    events = list(timeline.metadata["events"])
    detected_event_ids: list[str] = []
    latencies: list[float] = []
    false_alarm_count = 0
    for alarm in alarms:
        matching_events = [
            event
            for event in events
            if str(event["record_id"])
            == str(
                timeline.metadata["records"][
                    int(alarm["record_index"])
                ]["record_id"]
            )
            and _intervals_overlap(
                float(alarm["start_seconds"]),
                float(alarm["end_seconds"]),
                float(event["start_seconds"]),
                float(event["end_seconds"]),
            )
        ]
        if not matching_events:
            false_alarm_count += 1
            continue
        for event in matching_events:
            event_id = str(event["event_id"])
            if event_id in detected_event_ids:
                continue
            detected_event_ids.append(event_id)
            latencies.append(
                max(
                    0.0,
                    float(alarm["start_seconds"])
                    - float(event["start_seconds"]),
                )
            )
    normal_hours = _normal_monitoring_hours(timeline)
    event_count = len(events)
    detected_count = len(detected_event_ids)
    window_metrics = evaluate(
        np.asarray(timeline.labels, dtype=np.int64),
        values,
        threshold=threshold,
        print_report=False,
        sample_duration_seconds=float(
            timeline.metadata["window_config"]["stride_seconds"]
        ),
    )
    return {
        "threshold": threshold,
        "alarm_config": {
            "vote_k": config.vote_k,
            "vote_n": config.vote_n,
            "refractory_seconds": config.refractory_seconds,
        },
        "window_metrics": window_metrics,
        "event_metrics": {
            "event_count": event_count,
            "detected_events": detected_count,
            "missed_events": event_count - detected_count,
            "event_sensitivity": (
                detected_count / event_count if event_count else None
            ),
            "detected_event_ids": sorted(detected_event_ids),
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
            "alarm_episode_count": len(alarms),
        },
    }
