"""Causal, low-dimensional observations for temporal alarm policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import ProbabilityTimeline

MAD_NORMALIZATION = 1.4826
DEFAULT_HISTORY_LENGTH = 8


@dataclass(frozen=True)
class EnrollmentStatistics:
    """Probability distribution estimated from permitted normal enrollment rows."""

    median: float
    scaled_mad: float
    quantile_95: float
    count: int

    def validate(self) -> None:
        values = np.asarray(
            [self.median, self.scaled_mad, self.quantile_95], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError("enrollment statistics must be finite")
        if not 0 <= self.median <= 1 or not 0 <= self.quantile_95 <= 1:
            raise ValueError("enrollment probability statistics must be in [0, 1]")
        if self.scaled_mad < 0 or self.count < 1:
            raise ValueError("enrollment scale and count are invalid")


def compute_enrollment_statistics(
    normal_probabilities: Sequence[float] | np.ndarray,
) -> EnrollmentStatistics:
    """Compute robust enrollment summaries without using future labels."""
    values = np.asarray(normal_probabilities, dtype=np.float64).reshape(-1)
    if not values.size:
        raise ValueError("normal_probabilities must not be empty")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("normal_probabilities must be finite and in [0, 1]")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    statistics = EnrollmentStatistics(
        median=median,
        scaled_mad=MAD_NORMALIZATION * mad,
        quantile_95=float(np.quantile(values, 0.95)),
        count=int(values.size),
    )
    statistics.validate()
    return statistics


def causal_probability_histories(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> np.ndarray:
    """Build causal rolling histories and reset them at every recording boundary."""
    timeline.validate()
    enrollment.validate()
    if history_length < 1:
        raise ValueError("history_length must be positive")
    histories = np.empty((timeline.row_count, history_length), dtype=np.float32)
    history: list[float] = []
    previous_record = -1
    previous_start: int | None = None
    stride_samples = int(round(timeline.stride_seconds * timeline.sampling_frequency_hz))
    for row, probability in enumerate(timeline.probabilities):
        record = int(timeline.record_indices[row])
        start = int(timeline.start_samples[row])
        contiguous = (
            record == previous_record
            and previous_start is not None
            and start - previous_start == stride_samples
        )
        if not contiguous:
            history.clear()
        history.append(float(probability))
        history = history[-history_length:]
        padding = [enrollment.median] * (history_length - len(history))
        histories[row] = np.asarray(padding + history, dtype=np.float32)
        previous_record = record
        previous_start = start
    return histories


def build_policy_observation(
    probability_history: Sequence[float] | np.ndarray,
    enrollment: EnrollmentStatistics,
    *,
    seconds_since_alarm: float,
    refractory_remaining_seconds: float,
    record_start: bool,
) -> np.ndarray:
    """Construct the fixed 14-dimensional L1 policy observation."""
    enrollment.validate()
    history = np.asarray(probability_history, dtype=np.float32).reshape(-1)
    if history.shape != (DEFAULT_HISTORY_LENGTH,):
        raise ValueError(f"probability_history must have {DEFAULT_HISTORY_LENGTH} values")
    if not np.isfinite(history).all() or np.any((history < 0) | (history > 1)):
        raise ValueError("probability_history must be finite and in [0, 1]")
    if seconds_since_alarm < 0 or refractory_remaining_seconds < 0:
        raise ValueError("alarm timing values must be non-negative")
    timing_scale = 300.0
    observation = np.concatenate(
        [
            history,
            np.asarray(
                [
                    enrollment.median,
                    enrollment.scaled_mad,
                    enrollment.quantile_95,
                    min(seconds_since_alarm, timing_scale) / timing_scale,
                    min(refractory_remaining_seconds, timing_scale) / timing_scale,
                    float(record_start),
                ],
                dtype=np.float32,
            ),
        ]
    )
    if observation.shape != (14,) or not np.isfinite(observation).all():
        raise RuntimeError("policy observation contract is invalid")
    return observation
