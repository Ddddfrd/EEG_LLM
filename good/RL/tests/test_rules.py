from __future__ import annotations

import numpy as np
import pytest

from eeg_alarm_policy.cohort import SelectionObjective
from eeg_alarm_policy.evaluator import actions_from_probabilities
from eeg_alarm_policy.rules import (
    FixedRuleGrid,
    TemporalRule,
    apply_rule,
    default_threshold_grid,
    ema_smooth,
    enrollment_probabilities,
    hysteresis_states,
    pareto_frontier,
    search_fixed_rules,
    select_operating_point,
)


def test_plain_rule_matches_threshold_vote_path(sample_timeline) -> None:
    rule = TemporalRule(
        threshold=0.5, vote_k=2, vote_n=3, refractory_seconds=60.0
    )
    assert apply_rule(sample_timeline, rule).tolist() == actions_from_probabilities(
        sample_timeline, threshold=0.5, vote_k=2, vote_n=3
    ).tolist()


def test_ema_is_causal_and_resets_at_record_boundary(sample_timeline) -> None:
    smoothed = ema_smooth(sample_timeline, alpha=0.5)
    # Record 0: 0.1, 0.45, 0.675, 0.3875; record 1 restarts at its own first value.
    assert smoothed[:4].tolist() == pytest.approx([0.1, 0.45, 0.675, 0.3875])
    assert smoothed[4] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        ema_smooth(sample_timeline, alpha=0.0)


def test_hysteresis_state_machine_and_boundary_reset(sample_timeline) -> None:
    states = hysteresis_states(
        sample_timeline, on_threshold=0.85, off_threshold=0.3
    )
    assert states.tolist() == [
        False, False, True, False,
        True, False, False, True,
    ]
    with pytest.raises(ValueError):
        hysteresis_states(sample_timeline, on_threshold=0.3, off_threshold=0.3)


def test_rule_application_is_deterministic(sample_timeline) -> None:
    rule = TemporalRule(
        threshold=0.7,
        vote_k=1,
        vote_n=2,
        refractory_seconds=30.0,
        ema_alpha=0.6,
    )
    first = apply_rule(sample_timeline, rule)
    second = apply_rule(sample_timeline, rule)
    assert np.array_equal(first, second)


def test_default_threshold_grid_is_bounded_and_sorted(sample_timeline) -> None:
    grid = default_threshold_grid({"chb99": sample_timeline})
    assert grid == tuple(sorted(set(grid)))
    assert all(0 < value < 1 for value in grid)
    enrollment = enrollment_probabilities(sample_timeline)
    q95 = float(np.quantile(enrollment, 0.95))
    # Grid candidates are rounded to six decimals by contract.
    assert any(abs(value - q95) < 1e-6 for value in grid)


def test_enrollment_probabilities_use_earliest_normal_rows(sample_timeline) -> None:
    enrollment = enrollment_probabilities(sample_timeline, fraction=0.5)
    normal_rows = np.flatnonzero(sample_timeline.labels == 0)
    # ceil(0.5 * 5 normal rows) = 3 earliest known-normal windows.
    expected = sample_timeline.probabilities[normal_rows[:3]]
    assert enrollment.tolist() == [float(value) for value in expected]


def test_grid_materialization_count() -> None:
    grid = FixedRuleGrid(
        thresholds=(0.3, 0.5),
        vote_ns=(1, 2),
        refractory_seconds=(0.0, 30.0),
    )
    assert len(grid.rules()) == 2 * (1 + 2) * 2

    hysteresis_grid = FixedRuleGrid(
        thresholds=(0.3, 0.5),
        vote_ns=(1,),
        refractory_seconds=(0.0,),
        include_hysteresis=True,
        ema_alphas=(0.5,),
    )
    # (2 ema x 3 hysteresis variants) x 2 thresholds x 1 vote x 1 refractory.
    assert len(hysteresis_grid.rules()) == 12


def test_search_returns_objective_sorted_rows(sample_timeline) -> None:
    timelines = {"chb99": sample_timeline}
    grid = FixedRuleGrid(
        thresholds=(0.5,),
        vote_ns=(1, 2, 3),
        refractory_seconds=(0.0,),
    )
    objective = SelectionObjective(
        lambda_fa=0.02,
        lambda_latency=0.001,
        minimum_event_sensitivity=1.0,
    )
    rows = search_fixed_rules(timelines, grid, objective, progress_every=0)
    assert len(rows) == 6

    scores = [row["j_score"] for row in rows]
    defined = [score for score in scores if score is not None]
    assert defined == sorted(defined, reverse=True)
    best = select_operating_point(rows, objective)
    assert best is not None
    # k=2, n=3 detects both events immediately with zero false alarms.
    assert best["j_score"] == pytest.approx(1.0)
    assert best["pooled_event_metrics"]["false_alarm_episodes"] == 0
    assert best["pooled_event_metrics"]["detected_events"] == 2

    rerun = search_fixed_rules(timelines, grid, objective, progress_every=0)
    assert [row["rule_label"] for row in rerun] == [
        row["rule_label"] for row in rows
    ]


def test_pareto_frontier_removes_dominated_rows() -> None:
    def row(sensitivity: float, false_alarms: float, latency: float):
        return {
            "guardrail_pass": True,
            "pooled_event_metrics": {
                "event_sensitivity": sensitivity,
                "false_alarms_per_hour": false_alarms,
                "mean_detection_latency_seconds": latency,
            },
        }

    strong = row(1.0, 0.2, 10.0)
    dominated = row(1.0, 0.3, 12.0)
    tradeoff = row(0.9, 0.1, 8.0)
    assert pareto_frontier((strong, dominated, tradeoff)) == [strong, tradeoff]
