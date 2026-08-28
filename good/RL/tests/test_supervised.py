from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.supervised import (
    SUPERVISED_FEATURE_NAMES,
    build_supervised_features,
    fit_logistic_control,
    fit_mlp_control,
)


def test_supervised_features_are_causal_and_fixed_width(sample_timeline) -> None:
    features = build_supervised_features(sample_timeline)
    assert features.shape == (sample_timeline.row_count, 14)
    assert len(SUPERVISED_FEATURE_NAMES) == 14
    assert features[4, :7].tolist() == pytest.approx([features[4, 8]] * 7)
    assert features[4, -1] == 1.0


def test_compact_controls_fit_and_respect_parameter_budget() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, 14)).astype(np.float32)
    labels = np.asarray([0] * 160 + [1] * 40, dtype=np.uint8)
    logistic = fit_logistic_control(features, labels)
    mlp = fit_mlp_control(features, labels)
    assert logistic.predict_probabilities(features).shape == (200,)
    assert mlp.predict_probabilities(features).shape == (200,)
    assert logistic.parameter_count == 15
    assert mlp.parameter_count <= 10_000
