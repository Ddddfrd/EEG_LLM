from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from good.e1_e2_e3_e4_fullband.continual_personalization import (
    ContinualPersonalizationConfig,
    _balanced_epoch_indices,
    _classifier_probabilities,
    build_patient_continual_partition,
    train_replay_classifier,
)


class _FakeCache:
    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self.metadata = {"subject_order": ["chb99"]}

    @staticmethod
    def subject_slice(subject: str) -> slice:
        assert subject == "chb99"
        return slice(0, 40)


def _timeline() -> SimpleNamespace:
    labels = np.zeros(40, dtype=np.uint8)
    events = np.zeros(40, dtype=np.int16)
    for event, start in enumerate((10, 20, 30), start=1):
        labels[start : start + 2] = 1
        events[start : start + 2] = event
    return SimpleNamespace(
        labels=labels,
        event_indices=events,
        record_indices=np.zeros(40, dtype=np.int16),
        start_samples=np.arange(40, dtype=np.int32) * 256,
        metadata={
            "target_subject": "chb99",
            "window_config": {
                "window_seconds": 4.0,
                "stride_seconds": 1.0,
                "sampling_frequency_hz": 256,
            },
        },
    )


def test_patient_partition_is_chronological_and_event_disjoint() -> None:
    timeline = _timeline()
    cache = _FakeCache(timeline.labels)
    result = build_patient_continual_partition(
        cache,  # type: ignore[arg-type]
        {"chb99": timeline},
        config=ContinualPersonalizationConfig(
            enrollment_windows=4,
            adaptation_events=2,
        ),
    )
    patient = result["patients"]["chb99"]
    adaptation = np.asarray(patient["adaptation_local"])
    holdout = np.asarray(patient["holdout_local"])

    assert patient["selected_event_indices"] == [1, 2]
    assert patient["future_event_indices"] == [3]
    assert int(holdout[0]) == 25
    assert not np.intersect1d(patient["enrollment_local"], adaptation).size
    assert not np.intersect1d(adaptation, holdout).size
    assert set(timeline.event_indices[holdout].tolist()).isdisjoint({1, 2})


def test_balanced_epoch_indices_replicate_only_the_minority() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    selected = _balanced_epoch_indices(labels, np.random.default_rng(7))

    assert len(selected) == 6
    assert np.sum(labels[selected] == 0) == 3
    assert np.sum(labels[selected] == 1) == 3


def test_replay_classifier_learns_separable_frozen_features() -> None:
    features = np.asarray(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [-1.0, -2.0],
            [1.0, 1.0],
            [1.5, 0.5],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    classifier = nn.Linear(2, 2)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    trained, history, replay = train_replay_classifier(
        classifier,
        features,
        labels,
        [np.asarray([0, 1, 3, 4]), np.asarray([2, 5])],
        config=ContinualPersonalizationConfig(
            epochs_per_experience=20,
            replay_batch_size=4,
            classifier_learning_rate=0.05,
        ),
        device=torch.device("cpu"),
    )
    probabilities = _classifier_probabilities(trained, features)

    assert probabilities[labels == 1].min() > probabilities[labels == 0].max()
    assert len(history) == 2
    assert history[1]["replay_rows_before"] == 4
    assert len(replay) == 6
