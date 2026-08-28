"""Framework-independent contracts for cached EEG probability timelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventInterval:
    """One labeled seizure interval within an EDF recording."""

    event_index: int
    event_id: str
    record_index: int
    start_seconds: float
    end_seconds: float

    def validate(self, *, record_count: int) -> None:
        if self.event_index < 1:
            raise ValueError("event_index must be positive")
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not 0 <= self.record_index < record_count:
            raise ValueError("event record_index is out of range")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("event interval is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_id": self.event_id,
            "record_index": self.record_index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventInterval:
        return cls(
            event_index=int(payload["event_index"]),
            event_id=str(payload["event_id"]),
            record_index=int(payload["record_index"]),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
        )


@dataclass(frozen=True)
class ProbabilityTimeline:
    """One subject's frozen probabilities in exact natural-timeline order."""

    subject_id: str
    probabilities: np.ndarray
    labels: np.ndarray
    record_indices: np.ndarray
    start_samples: np.ndarray
    event_indices: np.ndarray
    records: tuple[str, ...]
    events: tuple[EventInterval, ...]
    sampling_frequency_hz: float
    window_seconds: float
    stride_seconds: float

    @classmethod
    def create(
        cls,
        *,
        subject_id: str,
        probabilities: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        record_indices: Sequence[int] | np.ndarray,
        start_samples: Sequence[int] | np.ndarray,
        event_indices: Sequence[int] | np.ndarray,
        records: Sequence[str],
        events: Sequence[EventInterval],
        sampling_frequency_hz: float,
        window_seconds: float,
        stride_seconds: float,
    ) -> ProbabilityTimeline:
        timeline = cls(
            subject_id=str(subject_id),
            probabilities=np.asarray(probabilities, dtype=np.float32).reshape(-1),
            labels=np.asarray(labels, dtype=np.uint8).reshape(-1),
            record_indices=np.asarray(record_indices, dtype=np.int32).reshape(-1),
            start_samples=np.asarray(start_samples, dtype=np.int64).reshape(-1),
            event_indices=np.asarray(event_indices, dtype=np.int32).reshape(-1),
            records=tuple(str(value) for value in records),
            events=tuple(events),
            sampling_frequency_hz=float(sampling_frequency_hz),
            window_seconds=float(window_seconds),
            stride_seconds=float(stride_seconds),
        )
        timeline.validate()
        return timeline

    @property
    def row_count(self) -> int:
        return int(self.probabilities.size)

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window and stride durations must be positive")
        if not self.records:
            raise ValueError("records must not be empty")

        arrays = (
            self.probabilities,
            self.labels,
            self.record_indices,
            self.start_samples,
            self.event_indices,
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("timeline arrays must be one-dimensional")
        if not self.row_count or any(len(value) != self.row_count for value in arrays):
            raise ValueError("timeline arrays must have one non-zero common length")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("probabilities must be finite")
        if np.any((self.probabilities < 0) | (self.probabilities > 1)):
            raise ValueError("probabilities must be in [0, 1]")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be binary")
        if np.any(self.record_indices < 0) or np.any(self.record_indices >= len(self.records)):
            raise ValueError("record_indices contain an out-of-range value")
        if np.any(self.start_samples < 0):
            raise ValueError("start_samples must be non-negative")
        if np.any(np.diff(self.record_indices) < 0):
            raise ValueError("record_indices must be in non-decreasing timeline order")

        same_record = self.record_indices[1:] == self.record_indices[:-1]
        if np.any(np.diff(self.start_samples)[same_record] <= 0):
            raise ValueError("start_samples must increase strictly within each record")

        event_by_index: dict[int, EventInterval] = {}
        event_ids: set[str] = set()
        for event in self.events:
            event.validate(record_count=len(self.records))
            if event.event_index in event_by_index:
                raise ValueError("event_index values must be unique")
            if event.event_id in event_ids:
                raise ValueError("event_id values must be unique")
            event_by_index[event.event_index] = event
            event_ids.add(event.event_id)

        observed = set(int(value) for value in np.unique(self.event_indices))
        if observed - ({0} | set(event_by_index)):
            raise ValueError("event_indices reference an unknown event")
        if not np.array_equal(self.labels == 1, self.event_indices > 0):
            raise ValueError("positive labels and event_indices must agree")
        for row in np.flatnonzero(self.event_indices > 0):
            event = event_by_index[int(self.event_indices[row])]
            if int(self.record_indices[row]) != event.record_index:
                raise ValueError("event row is assigned to the wrong record")
