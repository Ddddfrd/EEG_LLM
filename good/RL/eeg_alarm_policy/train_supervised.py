"""Train R2 logistic/MLP controls on chb20 and select policies on chb21."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import (
    load_prediction_artifact,
    save_content_addressed_json,
    save_prediction_artifact,
)
from .cohort import SelectionObjective
from .contracts import ProbabilityTimeline
from .rules import (
    FixedRuleGrid,
    default_threshold_grid,
    pareto_frontier,
    search_fixed_rules,
    select_operating_point,
)
from .supervised import (
    SupervisedControl,
    build_supervised_features,
    fit_logistic_control,
    fit_mlp_control,
)

SCHEMA_VERSION = "eeg_rl_supervised_controls_v1"


def _subject_artifact(directory: Path, subject: str):
    matches = sorted(directory.glob(f"predictions_{subject}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one base prediction artifact for {subject}, found {matches}"
        )
    return load_prediction_artifact(matches[0])


def _replace_probabilities(
    timeline: ProbabilityTimeline,
    probabilities: np.ndarray,
) -> ProbabilityTimeline:
    return ProbabilityTimeline.create(
        subject_id=timeline.subject_id,
        probabilities=probabilities,
        labels=timeline.labels,
        record_indices=timeline.record_indices,
        start_samples=timeline.start_samples,
        event_indices=timeline.event_indices,
        records=timeline.records,
        events=timeline.events,
        sampling_frequency_hz=timeline.sampling_frequency_hz,
        window_seconds=timeline.window_seconds,
        stride_seconds=timeline.stride_seconds,
    )


def _save_estimator(control: SupervisedControl, output_dir: Path) -> tuple[Path, str]:
    payload = pickle.dumps(control.estimator, protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_dir / f"{control.name}_{digest[:12]}.pkl"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("Existing supervised checkpoint hash mismatch")
        return destination, digest
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination, digest


def _ranking_metrics(timeline: ProbabilityTimeline) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(timeline.labels, timeline.probabilities)),
        "auprc": float(average_precision_score(timeline.labels, timeline.probabilities)),
    }


def _fit_and_select(
    control: SupervisedControl,
    *,
    training_artifact: Any,
    selection_artifact: Any,
    selection_features: np.ndarray,
    output_dir: Path,
    objective: SelectionObjective,
) -> dict[str, Any]:
    checkpoint, checkpoint_hash = _save_estimator(control, output_dir / "checkpoints")
    selection_probabilities = control.predict_probabilities(selection_features)
    selection_timeline = _replace_probabilities(
        selection_artifact.timeline,
        selection_probabilities,
    )
    score_artifact = save_prediction_artifact(
        selection_timeline,
        output_dir / "selection_predictions",
        partition_role="audit",
        model_metadata={
            **control.contract(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "model_contract_sha256": hashlib.sha256(
                json.dumps(control.contract(), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        source_metadata={
            "training_subject": training_artifact.timeline.subject_id,
            "training_artifact_id": training_artifact.metadata["artifact_id"],
            "selection_subject": selection_artifact.timeline.subject_id,
            "selection_artifact_id": selection_artifact.metadata["artifact_id"],
        },
    )
    timelines = {selection_timeline.subject_id: selection_timeline}
    grid = FixedRuleGrid(thresholds=default_threshold_grid(timelines))
    rows = search_fixed_rules(timelines, grid, objective, progress_every=500)
    selected = select_operating_point(rows, objective)
    return {
        "control": control.contract(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "selection_prediction_artifact": str(score_artifact.metadata_path),
        "selection_prediction_artifact_id": score_artifact.metadata["artifact_id"],
        "selection_ranking_metrics": _ranking_metrics(selection_timeline),
        "grid_rule_count": len(rows),
        "selected": selected,
        "pareto_frontier": pareto_frontier(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-fa", type=float, default=0.02)
    parser.add_argument("--lambda-latency", type=float, default=0.001)
    parser.add_argument("--min-event-sensitivity", type=float, default=0.8)
    args = parser.parse_args(argv)

    training = _subject_artifact(args.artifact_dir, "chb20")
    selection = _subject_artifact(args.artifact_dir, "chb21")
    training_features = build_supervised_features(training.timeline)
    selection_features = build_supervised_features(selection.timeline)
    controls = (
        fit_logistic_control(training_features, training.timeline.labels, seed=args.seed),
        fit_mlp_control(training_features, training.timeline.labels, seed=args.seed),
    )
    objective = SelectionObjective(
        lambda_fa=args.lambda_fa,
        lambda_latency=args.lambda_latency,
        minimum_event_sensitivity=args.min_event_sensitivity,
        minimum_patient_event_sensitivity=args.min_event_sensitivity,
    )
    results = {
        control.name: _fit_and_select(
            control,
            training_artifact=training,
            selection_artifact=selection,
            selection_features=selection_features,
            output_dir=args.output_dir,
            objective=objective,
        )
        for control in controls
    }
    eligible = [
        (name, payload["selected"])
        for name, payload in results.items()
        if payload["selected"] is not None
    ]
    selected_name, selected_payload = (
        max(eligible, key=lambda item: item[1]["j_score"])
        if eligible
        else (None, None)
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "chb20 supervised fit with chb21 temporal-policy selection",
        "training_artifact_id": training.metadata["artifact_id"],
        "selection_artifact_id": selection.metadata["artifact_id"],
        "seed": args.seed,
        "objective": {
            "lambda_fa": objective.lambda_fa,
            "lambda_latency": objective.lambda_latency,
            "minimum_event_sensitivity": objective.minimum_event_sensitivity,
        },
        "results": results,
        "selected_control": selected_name,
        "selected_operating_point": selected_payload,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    path = save_content_addressed_json(
        payload,
        args.output_dir,
        hash_field="result_sha256",
        stem="supervised_controls",
    )
    print(json.dumps({"result": str(path), "selected_control": selected_name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
