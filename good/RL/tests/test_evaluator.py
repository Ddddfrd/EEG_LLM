from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.evaluator import (
    AlarmConfig,
    actions_from_probabilities,
    evaluate_alarm_actions,
    evaluate_probability_policy,
)


def test_voting_resets_at_record_boundaries(sample_timeline) -> None:
    actions = actions_from_probabilities(
        sample_timeline,
        threshold=0.5,
        vote_k=2,
        vote_n=3,
    )

    assert actions.tolist() == [False, False, True, True, False, False, True, True]


def test_probability_and_explicit_action_event_metrics_are_identical(
    sample_timeline,
) -> None:
    config = AlarmConfig(vote_k=2, vote_n=3, refractory_seconds=60.0)
    actions = actions_from_probabilities(
        sample_timeline,
        threshold=0.5,
        vote_k=config.vote_k,
        vote_n=config.vote_n,
    )
    probability_result = evaluate_probability_policy(
        sample_timeline,
        threshold=0.5,
        alarm_config=config,
    )
    action_result = evaluate_alarm_actions(
        sample_timeline,
        actions,
        refractory_seconds=config.refractory_seconds,
    )

    assert probability_result["event_metrics"] == action_result["event_metrics"]
    assert action_result["event_metrics"]["detected_events"] == 2
    assert action_result["event_metrics"]["false_alarm_episodes"] == 0


def test_false_alarm_and_miss_are_counted(sample_timeline) -> None:
    actions = np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    result = evaluate_alarm_actions(
        sample_timeline,
        actions,
        refractory_seconds=0.0,
    )

    assert result["event_metrics"]["detected_events"] == 0
    assert result["event_metrics"]["missed_events"] == 2
    assert result["event_metrics"]["false_alarm_episodes"] == 1


def test_silent_policy_has_zero_f1(sample_timeline) -> None:
    result = evaluate_alarm_actions(
        sample_timeline,
        np.zeros(sample_timeline.row_count, dtype=np.uint8),
    )

    assert result["action_metrics"]["f1"] == 0.0


def test_non_binary_actions_are_rejected(sample_timeline) -> None:
    with pytest.raises(ValueError, match="binary"):
        evaluate_alarm_actions(sample_timeline, [0, 0, 2, 0, 0, 0, 0, 0])
