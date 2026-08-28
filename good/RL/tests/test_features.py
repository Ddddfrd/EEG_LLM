from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.features import (
    build_policy_observation,
    causal_probability_histories,
    compute_enrollment_statistics,
)


def test_enrollment_and_observation_contract(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.05, 0.1, 0.2, 0.3])
    histories = causal_probability_histories(sample_timeline, enrollment)
    observation = build_policy_observation(
        histories[2],
        enrollment,
        seconds_since_alarm=12.0,
        refractory_remaining_seconds=48.0,
        record_start=False,
    )

    assert histories.shape == (sample_timeline.row_count, 8)
    assert observation.shape == (14,)
    assert observation.dtype == np.float32
    assert observation[-1] == 0.0


def test_probability_history_does_not_cross_record_boundary(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2])
    histories = causal_probability_histories(sample_timeline, enrollment)

    assert histories[4, :-1].tolist() == pytest.approx([enrollment.median] * 7)
    assert histories[4, -1] == pytest.approx(sample_timeline.probabilities[4])


def test_zero_mad_enrollment_is_supported(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.2, 0.2, 0.2])
    histories = causal_probability_histories(sample_timeline, enrollment)

    assert enrollment.scaled_mad == 0.0
    assert np.isfinite(histories).all()
