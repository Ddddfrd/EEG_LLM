"""Deterministic temporal alarm rules and the fixed-rule grid search.

The grid covers threshold, k-of-n voting, and refractory period, with optional
record-local hysteresis and exponential moving-average smoothing. Every rule is
evaluated through the same evaluator used for RL actions, so rule and policy
results stay directly comparable (EEG_RL_ALARM_POLICY_PLAN.md section L1-A).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cohort import SelectionObjective, evaluate_cohort
from .contracts import ProbabilityTimeline
from .evaluator import voted_actions

ENROLLMENT_FRACTION = 0.2
ENROLLMENT_MAXIMUM_WINDOWS = 4000
ENROLLMENT_QUANTILES = (0.5, 0.9, 0.95, 0.99)
FIXED_GRID_LOW = 0.02
FIXED_GRID_POINTS = 24
DEFAULT_VOTE_NS = (1, 2, 3, 4, 5, 8)
DEFAULT_REFRACTORY_SECONDS = (0.0, 30.0, 60.0, 120.0, 300.0)
HYSTERESIS_OFF_FACTORS = (0.5, 0.8)


def _contiguous_segments(timeline: ProbabilityTimeline) -> list[list[int]]:
    """Split row indices into maximal same-record, stride-contiguous runs."""
    stride_samples = int(
        round(timeline.stride_seconds * timeline.sampling_frequency_hz)
    )
    segments: list[list[int]] = []
    current: list[int] = []
    previous_record = -1
    previous_start: int | None = None
    for row in range(timeline.row_count):
        record = int(timeline.record_indices[row])
        start = int(timeline.start_samples[row])
        contiguous = (
            record == previous_record
            and previous_start is not None
            and start - previous_start == stride_samples
        )
        if not contiguous and current:
            segments.append(current)
            current = []
        current.append(row)
        previous_record = record
        previous_start = start
    if current:
        segments.append(current)
    return segments


def ema_smooth(
    timeline: ProbabilityTimeline,
    *,
    alpha: float,
) -> np.ndarray:
    """Causal exponential moving average reset at every recording boundary."""
    if not 0 < alpha <= 1:
        raise ValueError("ema alpha must be in (0, 1]")
    smoothed = np.empty(timeline.row_count, dtype=np.float64)
    for segment in _contiguous_segments(timeline):
        state = 0.0
        for position, row in enumerate(segment):
            value = float(timeline.probabilities[row])
            state = value if position == 0 else alpha * value + (1.0 - alpha) * state
            smoothed[row] = state
    return smoothed


def hysteresis_states(
    timeline: ProbabilityTimeline,
    *,
    on_threshold: float,
    off_threshold: float,
) -> np.ndarray:
    """Two-threshold state machine reset at every recording boundary."""
    if not 0 <= off_threshold < on_threshold <= 1:
        raise ValueError("hysteresis requires 0 <= off < on <= 1")
    states = np.zeros(timeline.row_count, dtype=bool)
    for segment in _contiguous_segments(timeline):
        active = False
        for row in segment:
            value = float(timeline.probabilities[row])
            if not active and value >= on_threshold:
                active = True
            elif active and value < off_threshold:
                active = False
            states[row] = active
    return states


@dataclass(frozen=True)
class TemporalRule:
    """One deterministic alarm rule over frozen probabilities."""

    threshold: float
    vote_k: int
    vote_n: int
    refractory_seconds: float
    ema_alpha: float | None = None
    hysteresis_off_threshold: float | None = None

    def validate(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.vote_n < 1 or not 1 <= self.vote_k <= self.vote_n:
            raise ValueError("vote must satisfy 1 <= vote_k <= vote_n")
        if self.refractory_seconds < 0:
            raise ValueError("refractory_seconds must be non-negative")
        if self.ema_alpha is not None and not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1] or None")
        if self.hysteresis_off_threshold is not None and not (
            0 <= self.hysteresis_off_threshold < self.threshold
        ):
            raise ValueError("hysteresis off threshold must be below the on threshold")

    def label(self) -> str:
        parts = [
            f"t={self.threshold:.4f}",
            f"k={self.vote_k}",
            f"n={self.vote_n}",
            f"ref={self.refractory_seconds:g}s",
        ]
        if self.ema_alpha is not None:
            parts.append(f"ema={self.ema_alpha:g}")
        if self.hysteresis_off_threshold is not None:
            parts.append(f"hys_off={self.hysteresis_off_threshold:.4f}")
        return ",".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "vote_k": self.vote_k,
            "vote_n": self.vote_n,
            "refractory_seconds": self.refractory_seconds,
            "ema_alpha": self.ema_alpha,
            "hysteresis_off_threshold": self.hysteresis_off_threshold,
        }


def apply_rule(timeline: ProbabilityTimeline, rule: TemporalRule) -> np.ndarray:
    """Return the voted alarm actions produced by the rule on one timeline."""
    rule.validate()
    timeline.validate()
    if rule.ema_alpha is not None:
        signal = ema_smooth(timeline, alpha=rule.ema_alpha)
    else:
        signal = np.asarray(timeline.probabilities, dtype=np.float64)
    if rule.hysteresis_off_threshold is None:
        raw = signal >= rule.threshold
    else:
        raw = hysteresis_states(
            timeline,
            on_threshold=rule.threshold,
            off_threshold=rule.hysteresis_off_threshold,
        )
    return voted_actions(timeline, raw, vote_k=rule.vote_k, vote_n=rule.vote_n)


def enrollment_probabilities(
    timeline: ProbabilityTimeline,
    *,
    fraction: float = ENROLLMENT_FRACTION,
    maximum_windows: int = ENROLLMENT_MAXIMUM_WINDOWS,
) -> np.ndarray:
    """Earliest known-normal probabilities, mirroring the E2 enrollment rule."""
    timeline.validate()
    if not 0 < fraction <= 1:
        raise ValueError("enrollment fraction must be in (0, 1]")
    normal_rows = np.flatnonzero(timeline.labels == 0)
    if not normal_rows.size:
        raise ValueError(f"{timeline.subject_id} has no known-normal windows")
    count = min(int(maximum_windows), int(np.ceil(fraction * normal_rows.size)))
    return np.asarray(timeline.probabilities[normal_rows[:count]], dtype=np.float64)


def default_threshold_grid(
    timelines: Mapping[str, ProbabilityTimeline],
) -> tuple[float, ...]:
    """Bounded fixed grid plus per-subject enrollment quantile candidates."""
    candidates = np.linspace(
        FIXED_GRID_LOW, 0.94, FIXED_GRID_POINTS, endpoint=True
    ).tolist()
    for timeline in timelines.values():
        enrollment = enrollment_probabilities(timeline)
        candidates.extend(float(np.quantile(enrollment, q)) for q in ENROLLMENT_QUANTILES)
    unique = sorted({round(float(value), 6) for value in candidates if 0 < value < 1})
    return tuple(unique)


@dataclass(frozen=True)
class FixedRuleGrid:
    """Search space for deterministic rules."""

    thresholds: tuple[float, ...]
    vote_ns: tuple[int, ...] = DEFAULT_VOTE_NS
    refractory_seconds: tuple[float, ...] = DEFAULT_REFRACTORY_SECONDS
    include_hysteresis: bool = False
    ema_alphas: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.thresholds:
            raise ValueError("threshold grid must not be empty")
        if any(not 0 < value < 1 for value in self.thresholds):
            raise ValueError("thresholds must lie in (0, 1)")
        if not self.vote_ns or not self.refractory_seconds:
            raise ValueError("vote and refractory grids must not be empty")
        for alpha in self.ema_alphas:
            if not 0 < alpha <= 1:
                raise ValueError("ema alphas must lie in (0, 1]")

    def rules(self) -> list[TemporalRule]:
        """Materialize every rule in the grid, deterministic in iteration order."""
        self.validate()
        rule_list: list[TemporalRule] = []
        for ema_alpha in (None, *self.ema_alphas):
            hysteresis_variants: tuple[float | None, ...]
            if self.include_hysteresis:
                hysteresis_variants = (None, *HYSTERESIS_OFF_FACTORS)
            else:
                hysteresis_variants = (None,)
            for hysteresis in hysteresis_variants:
                for threshold in self.thresholds:
                    if hysteresis is not None and threshold * hysteresis >= threshold:
                        continue
                    off = (
                        None
                        if hysteresis is None
                        else round(threshold * hysteresis, 6)
                    )
                    for vote_n in self.vote_ns:
                        for vote_k in range(1, vote_n + 1):
                            for refractory in self.refractory_seconds:
                                rule_list.append(
                                    TemporalRule(
                                        threshold=float(threshold),
                                        vote_k=vote_k,
                                        vote_n=vote_n,
                                        refractory_seconds=float(refractory),
                                        ema_alpha=ema_alpha,
                                        hysteresis_off_threshold=off,
                                    )
                                )
        return rule_list


def evaluate_rule(
    timelines: Mapping[str, ProbabilityTimeline],
    rule: TemporalRule,
) -> dict[str, Any]:
    """Evaluate one rule over the whole cohort through the shared evaluator."""
    actions_by_subject = {
        subject: apply_rule(timeline, rule)
        for subject, timeline in timelines.items()
    }
    cohort = evaluate_cohort(
        timelines,
        actions_by_subject,
        refractory_seconds=rule.refractory_seconds,
    )
    return cohort


_RULE_SORT_FIELDS = (
    "threshold",
    "vote_n",
    "vote_k",
    "refractory_seconds",
    "ema_alpha",
    "hysteresis_off_threshold",
)


def _rule_sort_key(rule: TemporalRule) -> tuple:
    return tuple(
        float("-inf") if getattr(rule, field) is None else getattr(rule, field)
        for field in _RULE_SORT_FIELDS
    )


def search_fixed_rules(
    timelines: Mapping[str, ProbabilityTimeline],
    grid: FixedRuleGrid,
    objective: SelectionObjective,
    *,
    progress_every: int = 250,
) -> list[dict[str, Any]]:
    """Evaluate the full grid and return rows sorted by the declared objective."""
    objective.validate()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    rule_list = grid.rules()
    for index, rule in enumerate(rule_list, start=1):
        cohort = evaluate_rule(timelines, rule)
        pooled = cohort["pooled"]
        score = objective.score(pooled)
        rows.append(
            {
                "rule": rule.to_dict(),
                "rule_label": rule.label(),
                "j_score": score,
                "guardrail_pass": objective.passes_guardrail(
                    pooled,
                    cohort["patient_summary"],
                ),
                "pooled_event_metrics": {
                    key: pooled[key]
                    for key in (
                        "event_count",
                        "detected_events",
                        "missed_events",
                        "event_sensitivity",
                        "false_alarm_episodes",
                        "normal_monitoring_hours",
                        "false_alarms_per_hour",
                        "mean_detection_latency_seconds",
                        "median_detection_latency_seconds",
                        "accepted_alarm_episode_count",
                    )
                },
                "pooled_action_metrics": {
                    key: pooled["action_metrics"][key]
                    for key in ("sensitivity", "specificity", "precision", "f1")
                },
                "patient_summary": cohort["patient_summary"],
            }
        )
        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"rule search {index}/{len(rule_list)} "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            float("inf") if row["j_score"] is None else -row["j_score"],
            row["pooled_event_metrics"]["false_alarms_per_hour"]
            if row["pooled_event_metrics"]["false_alarms_per_hour"] is not None
            else float("inf"),
            _rule_sort_key(TemporalRule(**row["rule"])),
        )
    )
    return rows


def pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return unique guardrail-passing, non-dominated event operating points."""
    eligible = [
        row
        for row in rows
        if row["guardrail_pass"]
        and row["pooled_event_metrics"]["event_sensitivity"] is not None
        and row["pooled_event_metrics"]["false_alarms_per_hour"] is not None
    ]

    def values(row: Mapping[str, Any]) -> tuple[float, float, float]:
        metrics = row["pooled_event_metrics"]
        latency = metrics["mean_detection_latency_seconds"]
        return (
            float(metrics["event_sensitivity"]),
            float(metrics["false_alarms_per_hour"]),
            float(latency) if latency is not None else float("inf"),
        )

    frontier: list[Mapping[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in eligible:
        candidate_values = values(candidate)
        if candidate_values in seen:
            continue
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            other_values = values(other)
            no_worse = (
                other_values[0] >= candidate_values[0]
                and other_values[1] <= candidate_values[1]
                and other_values[2] <= candidate_values[2]
            )
            strictly_better = (
                other_values[0] > candidate_values[0]
                or other_values[1] < candidate_values[1]
                or other_values[2] < candidate_values[2]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            seen.add(candidate_values)
            frontier.append(candidate)
    return frontier


def select_operating_point(
    rows: Sequence[Mapping[str, Any]],
    objective: SelectionObjective,
) -> Mapping[str, Any] | None:
    """Best guardrail-passing row; grid rows arrive pre-sorted by objective."""
    for row in rows:
        if row["guardrail_pass"] and row["j_score"] is not None:
            return row
    return None


def reference_rule_rows(
    timelines: Mapping[str, ProbabilityTimeline],
    rules: Sequence[TemporalRule],
) -> list[dict[str, Any]]:
    """Evaluate named reference rules (for example the inherited default)."""
    rows = []
    for rule in rules:
        cohort = evaluate_rule(timelines, rule)
        rows.append(
            {
                "rule": rule.to_dict(),
                "rule_label": rule.label(),
                "pooled": {key: value for key, value in cohort["pooled"].items()},
            }
        )
    return rows


def iter_rule_labels(rules: Sequence[TemporalRule]) -> Iterator[str]:
    for rule in rules:
        yield rule.label()
