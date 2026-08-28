"""Command-line fixed-rule grid search over exported prediction artifacts.

Consumes content-addressed prediction artifacts written by the integration
exporter, runs the deterministic rule grid on the requested policy-selection
subjects, and writes one content-addressed JSON result plus a markdown table.
Final-test artifacts stay unread unless ``--final-evaluation`` is passed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    load_prediction_artifact,
    save_content_addressed_json,
)
from .cohort import SelectionObjective
from .contracts import ProbabilityTimeline
from .rules import (
    DEFAULT_REFRACTORY_SECONDS,
    DEFAULT_VOTE_NS,
    FixedRuleGrid,
    TemporalRule,
    default_threshold_grid,
    evaluate_rule,
    pareto_frontier,
    search_fixed_rules,
    select_operating_point,
)
from .splits import DevelopmentGate

RULE_SEARCH_SCHEMA_VERSION = "eeg_rl_rule_search_v1"
TOP_TABLE_ROWS = 15
REFERENCE_VOTE_K = 2
REFERENCE_VOTE_N = 3
REFERENCE_REFRACTORY_SECONDS = 60.0


def _parse_floats(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    return tuple(float(value) for value in text.split(",") if value.strip())


def _parse_subjects(text: str) -> tuple[str, ...]:
    subjects = tuple(subject.strip() for subject in text.split(",") if subject.strip())
    if not subjects:
        raise ValueError("--subjects must name at least one subject")
    return subjects


def _load_cohort(artifact_dir: Path, subjects: Sequence[str], gate: DevelopmentGate):
    timelines = {}
    artifact_ids = {}
    for metadata_path in sorted(Path(artifact_dir).glob("predictions_*.json")):
        artifact = load_prediction_artifact(metadata_path)
        gate.require_role_allowed(artifact.metadata["partition_role"])
        subject_id = str(artifact.metadata["subject_id"])
        if subject_id in subjects:
            if subject_id in timelines:
                raise ValueError(f"Multiple artifacts found for {subject_id}")
            timelines[subject_id] = artifact.timeline
            artifact_ids[subject_id] = artifact.metadata["artifact_id"]
    missing = sorted(set(subjects) - set(timelines))
    if missing:
        raise FileNotFoundError(f"No prediction artifact for subjects: {missing}")
    return timelines, artifact_ids


def _markdown_report(
    *,
    selected: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    subjects: Sequence[str],
    objective: SelectionObjective,
    threshold_count: int,
    rule_count: int,
    duration_seconds: float,
) -> str:
    def event_row(row: dict[str, Any]) -> str:
        events = row.get("pooled_event_metrics", {})
        return (
            f"| {row['rule_label']} | {events.get('event_sensitivity')} "
            f"| {events.get('false_alarm_episodes')} "
            f"| {events.get('false_alarms_per_hour')} "
            f"| {events.get('mean_detection_latency_seconds')} "
            f"| {row.get('j_score')} |"
        )

    lines = [
        "# Fixed-Rule Alarm Grid Search",
        "",
        f"- Subjects: {', '.join(subjects)}",
        f"- Objective: J = sensitivity - {objective.lambda_fa} * FA/h "
        f"- {objective.lambda_latency} * latency/{objective.latency_normalizer_seconds:.0f}s",
        f"- Guardrail: event sensitivity >= {objective.minimum_event_sensitivity}",
        f"- Threshold candidates: {threshold_count}; rules evaluated: {rule_count}",
        f"- Wall-clock duration: {duration_seconds:.1f}s",
        "",
        "| Rule | Sensitivity | FA episodes | FA/hour | Mean latency (s) | J |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if selected is not None:
        lines.append(event_row(selected) + " <- selected")
    for row in rows[:TOP_TABLE_ROWS]:
        if row is selected:
            continue
        lines.append(event_row(row))
    if references:
        lines += ["", "## Reference rules", ""]
        lines += [event_row(reference) for reference in references]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-fa", type=float, default=0.02)
    parser.add_argument("--lambda-latency", type=float, default=0.001)
    parser.add_argument("--latency-normalizer-seconds", type=float, default=60.0)
    parser.add_argument("--min-event-sensitivity", type=float, default=0.8)
    parser.add_argument("--min-patient-event-sensitivity", type=float, default=0.8)
    parser.add_argument("--include-hysteresis", action="store_true")
    parser.add_argument("--ema-alphas", type=str, default="")
    parser.add_argument(
        "--vote-ns",
        type=str,
        default=",".join(str(value) for value in DEFAULT_VOTE_NS),
    )
    parser.add_argument(
        "--refractory-seconds",
        type=str,
        default=",".join(str(value) for value in DEFAULT_REFRACTORY_SECONDS),
    )
    parser.add_argument(
        "--reference-thresholds",
        type=str,
        default="0.5987815260887146",
        help="Comma-separated thresholds evaluated with the inherited 2-of-3/60s rule",
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Unlock reading final-test artifacts; the gate refuses them otherwise",
    )
    args = parser.parse_args(argv)

    gate = DevelopmentGate(unlocked=args.final_evaluation)
    subjects = _parse_subjects(args.subjects)
    timelines, artifact_ids = _load_cohort(args.artifact_dir, subjects, gate)

    thresholds = default_threshold_grid(timelines)
    vote_ns = tuple(int(value) for value in args.vote_ns.split(",") if value.strip())
    refractory = _parse_floats(args.refractory_seconds)
    grid = FixedRuleGrid(
        thresholds=thresholds,
        vote_ns=vote_ns,
        refractory_seconds=refractory,
        include_hysteresis=args.include_hysteresis,
        ema_alphas=_parse_floats(args.ema_alphas),
    )
    objective = SelectionObjective(
        lambda_fa=args.lambda_fa,
        lambda_latency=args.lambda_latency,
        latency_normalizer_seconds=args.latency_normalizer_seconds,
        minimum_event_sensitivity=args.min_event_sensitivity,
        minimum_patient_event_sensitivity=args.min_patient_event_sensitivity,
    )
    objective.validate()

    started = time.perf_counter()
    rows = search_fixed_rules(timelines, grid, objective)
    selected = select_operating_point(rows, objective)
    frontier = pareto_frontier(rows)

    reference_rules = [
        TemporalRule(
            threshold=threshold,
            vote_k=REFERENCE_VOTE_K,
            vote_n=REFERENCE_VOTE_N,
            refractory_seconds=REFERENCE_REFRACTORY_SECONDS,
        )
        for threshold in _parse_floats(args.reference_thresholds)
    ]
    references = [_reference_payload(rule, timelines, objective) for rule in reference_rules]

    payload: dict[str, Any] = {
        "schema_version": RULE_SEARCH_SCHEMA_VERSION,
        "method": "deterministic fixed-rule grid search over frozen probabilities",
        "final_evaluation_unlocked": bool(args.final_evaluation),
        "subjects": list(subjects),
        "artifact_ids": artifact_ids,
        "objective": {
            "lambda_fa": objective.lambda_fa,
            "lambda_latency": objective.lambda_latency,
            "latency_normalizer_seconds": objective.latency_normalizer_seconds,
            "minimum_event_sensitivity": objective.minimum_event_sensitivity,
            "minimum_patient_event_sensitivity": (
                objective.minimum_patient_event_sensitivity
            ),
        },
        "grid": {
            "threshold_count": len(thresholds),
            "thresholds": list(thresholds),
            "vote_ns": list(grid.vote_ns),
            "refractory_seconds": list(grid.refractory_seconds),
            "include_hysteresis": grid.include_hysteresis,
            "ema_alphas": list(grid.ema_alphas),
            "rule_count": len(rows),
        },
        "selected": selected,
        "pareto_frontier": frontier,
        "references": references,
        "rows": rows,
        # Runtime versions stay in the hashed payload; wall-clock duration is
        # deliberately excluded so identical searches keep one content address.
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    duration_seconds = time.perf_counter() - started
    result_path = save_content_addressed_json(
        payload,
        args.output_dir,
        hash_field="result_sha256",
        stem="rule_search",
    )
    digest = json.loads(result_path.read_text(encoding="utf-8"))["result_sha256"]
    report_path = args.output_dir / f"rule_search_{digest[:12]}.md"
    report_path.write_text(
        _markdown_report(
            selected=selected,
            rows=rows,
            references=references,
            subjects=subjects,
            objective=objective,
            threshold_count=len(thresholds),
            rule_count=len(rows),
            duration_seconds=duration_seconds,
        ),
        encoding="utf-8",
    )
    print(f"rule search rows: {len(rows)} ({duration_seconds:.1f}s)")
    print(f"selected: {selected['rule_label'] if selected else 'none passes guardrail'}")
    print(f"result: {result_path}")
    return 0


def _reference_payload(
    rule: TemporalRule,
    timelines: dict[str, ProbabilityTimeline],
    objective: SelectionObjective,
) -> dict[str, Any]:
    cohort = evaluate_rule(timelines, rule)
    pooled = cohort["pooled"]
    return {
        "rule": rule.to_dict(),
        "rule_label": rule.label(),
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
        "j_score": objective.score(pooled),
    }


if __name__ == "__main__":
    raise SystemExit(main())
