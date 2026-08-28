from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.cohort import SelectionObjective, evaluate_cohort


@pytest.fixture
def two_subject_timelines(sample_timeline):
    """sample_timeline plus a second subject with one event on [0, 4)."""
    from eeg_alarm_policy.contracts import EventInterval

    second = sample_timeline.__class__.create(
        subject_id="chb98",
        probabilities=np.asarray([0.9, 0.1, 0.1, 0.1], dtype=np.float32),
        labels=np.asarray([1, 0, 0, 0], dtype=np.uint8),
        record_indices=np.asarray([0, 0, 0, 0], dtype=np.int32),
        start_samples=np.asarray([0, 4, 8, 12], dtype=np.int64),
        event_indices=np.asarray([1, 0, 0, 0], dtype=np.int32),
        records=("record_a.edf",),
        events=(EventInterval(1, "event-a", 0, 0.0, 4.0),),
        sampling_frequency_hz=1.0,
        window_seconds=4.0,
        stride_seconds=4.0,
    )
    return {"chb99": sample_timeline, "chb98": second}


def test_cohort_pools_event_counts_and_hours(two_subject_timelines) -> None:
    # chb99 normal spans: [0,8), [12,16), [16,24) = 20s; chb98: [4,16) = 12s.
    actions = {
        "chb99": np.asarray([0, 0, 1, 0, 0, 0, 1, 1], dtype=np.uint8),
        "chb98": np.asarray([1, 0, 0, 0], dtype=np.uint8),
    }
    cohort = evaluate_cohort(
        two_subject_timelines, actions, refractory_seconds=0.0
    )

    pooled = cohort["pooled"]
    assert pooled["event_count"] == 3
    assert pooled["detected_events"] == 3
    assert pooled["false_alarm_episodes"] == 0
    assert pooled["event_sensitivity"] == 1.0
    assert pooled["normal_monitoring_hours"] == pytest.approx(32 / 3600)
    assert pooled["action_metrics"]["confusion_matrix"]["tp"] == 4


def test_pooled_false_alarm_rate_uses_total_hours(two_subject_timelines) -> None:
    # One false alarm on chb98 only; pooled rate must divide by both subjects' hours.
    actions = {
        "chb99": np.asarray([0, 0, 1, 0, 0, 0, 1, 1], dtype=np.uint8),
        "chb98": np.asarray([0, 0, 1, 0], dtype=np.uint8),
    }
    cohort = evaluate_cohort(
        two_subject_timelines, actions, refractory_seconds=0.0
    )

    pooled = cohort["pooled"]
    assert pooled["false_alarm_episodes"] == 1
    assert pooled["normal_monitoring_hours"] == pytest.approx(32 / 3600)
    assert pooled["false_alarms_per_hour"] == pytest.approx(1 / (32 / 3600))
    assert cohort["patient_summary"]["worst_false_alarms_per_hour"] == pytest.approx(
        1 / (12 / 3600)
    )
    assert cohort["patient_summary"]["mean_false_alarms_per_hour"] == pytest.approx(
        0.5 / (12 / 3600)
    )


def test_pooled_latency_uses_every_detection(two_subject_timelines) -> None:
    actions = {
        "chb99": np.asarray([0, 0, 1, 0, 0, 0, 1, 1], dtype=np.uint8),
        "chb98": np.asarray([1, 0, 0, 0], dtype=np.uint8),
    }
    cohort = evaluate_cohort(
        two_subject_timelines, actions, refractory_seconds=0.0
    )

    assert cohort["pooled"]["latencies_seconds"] == [0.0, 0.0, 0.0]
    assert cohort["pooled"]["mean_detection_latency_seconds"] == 0.0


def test_cohort_rejects_missing_subject_actions(two_subject_timelines) -> None:
    with pytest.raises(ValueError, match="exactly"):
        evaluate_cohort(
            two_subject_timelines, {"chb99": [0]}, refractory_seconds=0.0
        )


def test_selection_objective_score_and_guardrail() -> None:
    objective = SelectionObjective(
        lambda_fa=0.02, lambda_latency=0.001, minimum_event_sensitivity=0.9
    )
    objective.validate()
    good = {
        "event_sensitivity": 1.0,
        "false_alarms_per_hour": 5.0,
        "mean_detection_latency_seconds": 12.0,
    }
    score = objective.score(good)
    assert score == pytest.approx(1.0 - 0.02 * 5.0 - 0.001 * 12.0 / 60.0)
    assert objective.passes_guardrail(good)

    silent = {
        "event_sensitivity": 0.5,
        "false_alarms_per_hour": 0.0,
        "mean_detection_latency_seconds": None,
    }
    assert objective.score(silent) == pytest.approx(0.5)
    assert not objective.passes_guardrail(silent)

    undefined = {"event_sensitivity": None, "false_alarms_per_hour": None}
    assert objective.score(undefined) is None

    with pytest.raises(ValueError):
        SelectionObjective(lambda_fa=-1.0, lambda_latency=0.0).validate()


def test_patient_guardrail_rejects_pooled_success_with_weak_patient() -> None:
    objective = SelectionObjective(
        lambda_fa=0.02,
        lambda_latency=0.001,
        minimum_event_sensitivity=0.8,
        minimum_patient_event_sensitivity=0.8,
    )
    pooled = {
        "event_sensitivity": 0.9,
        "false_alarms_per_hour": 0.2,
        "mean_detection_latency_seconds": 10.0,
    }
    assert not objective.passes_guardrail(
        pooled,
        {"worst_event_sensitivity": 0.75},
    )
    assert objective.passes_guardrail(
        pooled,
        {"worst_event_sensitivity": 0.8},
    )
