from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.contracts import EventInterval, ProbabilityTimeline


@pytest.fixture
def sample_timeline() -> ProbabilityTimeline:
    return ProbabilityTimeline.create(
        subject_id="chb99",
        probabilities=np.asarray(
            [0.1, 0.8, 0.9, 0.1, 0.9, 0.2, 0.8, 0.9], dtype=np.float32
        ),
        labels=np.asarray([0, 0, 1, 0, 0, 0, 1, 1], dtype=np.uint8),
        record_indices=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        start_samples=np.asarray([0, 4, 8, 12, 0, 4, 8, 12], dtype=np.int64),
        event_indices=np.asarray([0, 0, 1, 0, 0, 0, 2, 2], dtype=np.int32),
        records=("record_01.edf", "record_02.edf"),
        events=(
            EventInterval(1, "event-1", 0, 8.0, 12.0),
            EventInterval(2, "event-2", 1, 8.0, 16.0),
        ),
        sampling_frequency_hz=1.0,
        window_seconds=4.0,
        stride_seconds=4.0,
    )
