"""Apply only pre-frozen R4 methods to the held-out chb22-23 artifacts."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import load_prediction_artifact, save_content_addressed_json
from .cohort import SelectionObjective, evaluate_cohort
from .rules import TemporalRule, apply_rule
from .supervised import build_supervised_features

SCHEMA_VERSION = "eeg_rl_r4_final_evaluation_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_subject(directory: Path, subject: str):
    matches = sorted(directory.glob(f"predictions_{subject}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one held-out artifact for {subject}")
    return load_prediction_artifact(matches[0])


def _evaluate(
    timelines: dict[str, Any],
    actions: dict[str, np.ndarray],
    *,
    refractory_seconds: float,
    objective: SelectionObjective,
) -> dict[str, Any]:
    cohort = evaluate_cohort(
        timelines,
        actions,
        refractory_seconds=refractory_seconds,
    )
    pooled = cohort["pooled"]
    return {
        "j_score": objective.score(pooled),
        "guardrail_pass": objective.passes_guardrail(
            pooled,
            cohort["patient_summary"],
        ),
        **cohort,
    }


def _save_actions(
    actions: dict[str, np.ndarray],
    output_dir: Path,
    *,
    method: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for subject, values_input in actions.items():
        values = np.asarray(values_input, dtype=np.uint8)
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        destination = output_dir / f"{method}_{subject}_{digest[:12]}.npy"
        if not destination.exists():
            temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            os.replace(temporary, destination)
        result[subject] = {"path": str(destination), "sha256": digest}
    return result


def _load_supervised_estimator(r2: dict[str, Any], name: str):
    payload = r2["results"][name]
    checkpoint = Path(payload["checkpoint"]).resolve()
    if _sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError(f"{name} checkpoint SHA256 mismatch")
    return pickle.loads(checkpoint.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-artifact-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_final_test_export":
        raise ValueError("Final evaluation requires a valid pre-test freeze")
    subjects = tuple(freeze["final_test_subjects"])
    artifacts = {
        subject: _load_subject(args.test_artifact_dir, subject)
        for subject in subjects
    }
    timelines = {
        subject: artifact.timeline for subject, artifact in artifacts.items()
    }
    objective = SelectionObjective(
        lambda_fa=float(freeze["objective"]["lambda_fa"]),
        lambda_latency=float(freeze["objective"]["lambda_latency"]),
        latency_normalizer_seconds=float(
            freeze["objective"]["latency_normalizer_seconds"]
        ),
        minimum_event_sensitivity=float(
            freeze["objective"]["minimum_event_sensitivity"]
        ),
        minimum_patient_event_sensitivity=float(
            freeze["objective"]["minimum_event_sensitivity"]
        ),
    )
    methods: dict[str, Any] = {}
    frozen_rules = {
        "inherited_rule": TemporalRule(
            **freeze["final_comparators"]["inherited_rule"]
        ),
        "robust_fixed_rule": TemporalRule(
            **freeze["primary_method"]["rule"]
        ),
    }
    for name, rule in frozen_rules.items():
        actions = {
            subject: apply_rule(timeline, rule)
            for subject, timeline in timelines.items()
        }
        methods[name] = {
            "rule": rule.to_dict(),
            "evaluation": _evaluate(
                timelines,
                actions,
                refractory_seconds=rule.refractory_seconds,
                objective=objective,
            ),
            "actions": _save_actions(
                actions,
                args.output_dir / "actions",
                method=name,
            ),
        }

    r2_source = Path(freeze["sources"]["r2"]["path"])
    if _sha256(r2_source) != freeze["sources"]["r2"]["sha256"]:
        raise ValueError("Frozen R2 result SHA256 mismatch")
    r2 = json.loads(r2_source.read_text(encoding="utf-8"))
    for name in ("logistic_regression", "mlp_32x32"):
        estimator = _load_supervised_estimator(r2, name)
        score_timelines = {}
        for subject, timeline in timelines.items():
            scores = estimator.predict_proba(build_supervised_features(timeline))[:, 1]
            score_timelines[subject] = type(timeline).create(
                subject_id=timeline.subject_id,
                probabilities=np.asarray(scores, dtype=np.float32),
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
        rule = TemporalRule(**freeze["final_comparators"][name])
        actions = {
            subject: apply_rule(timeline, rule)
            for subject, timeline in score_timelines.items()
        }
        methods[name] = {
            "rule": rule.to_dict(),
            "ranking_metrics": {
                "auroc": float(
                    roc_auc_score(
                        np.concatenate([value.labels for value in score_timelines.values()]),
                        np.concatenate(
                            [value.probabilities for value in score_timelines.values()]
                        ),
                    )
                ),
                "auprc": float(
                    average_precision_score(
                        np.concatenate([value.labels for value in score_timelines.values()]),
                        np.concatenate(
                            [value.probabilities for value in score_timelines.values()]
                        ),
                    )
                ),
            },
            "evaluation": _evaluate(
                score_timelines,
                actions,
                refractory_seconds=rule.refractory_seconds,
                objective=objective,
            ),
            "actions": _save_actions(
                actions,
                args.output_dir / "actions",
                method=name,
            ),
        }

    base_labels = np.concatenate([timeline.labels for timeline in timelines.values()])
    base_probabilities = np.concatenate(
        [timeline.probabilities for timeline in timelines.values()]
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "single frozen evaluation on chb22-chb23",
        "protocol_freeze": {
            "path": str(args.freeze.resolve()),
            "sha256": _sha256(args.freeze),
            "freeze_sha256": freeze["freeze_sha256"],
        },
        "subjects": list(subjects),
        "prediction_artifact_ids": {
            subject: artifact.metadata["artifact_id"]
            for subject, artifact in artifacts.items()
        },
        "base_probability_metrics": {
            "auroc": float(roc_auc_score(base_labels, base_probabilities)),
            "auprc": float(average_precision_score(base_labels, base_probabilities)),
        },
        "primary_method": "robust_fixed_rule",
        "methods": methods,
        "ppo_evaluated": False,
        "ppo_exclusion_reason": freeze["excluded_from_final_test"]["ppo"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    path = save_content_addressed_json(
        payload,
        args.output_dir,
        hash_field="result_sha256",
        stem="r4_final_evaluation",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
