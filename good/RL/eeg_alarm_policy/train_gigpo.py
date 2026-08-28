"""Run G3 GiGPO (episode + step-level grouped advantages) on chb20.

Selects the median chb21 seed under the frozen protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_prediction_artifact, save_content_addressed_json
from .cohort import SelectionObjective, evaluate_cohort
from .environment import AlarmRewardConfig
from .features import compute_enrollment_statistics
from .gigpo_training import GiGPOConfig, train_gigpo
from .grpo_training import actor_actions
from .rules import enrollment_probabilities

SCHEMA_VERSION = "eeg_rl_g3_gigpo_v1"
GIGPO_SEEDS = (11, 22, 33, 44, 55)


def _load_subject(directory: Path, subject: str):
    matches = sorted(directory.glob(f"predictions_{subject}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one prediction artifact for {subject}")
    return load_prediction_artifact(matches[0])


def _metrics(
    timeline,
    actions: np.ndarray,
    *,
    objective: SelectionObjective,
    refractory_seconds: float,
) -> dict[str, Any]:
    cohort = evaluate_cohort(
        {timeline.subject_id: timeline},
        {timeline.subject_id: actions},
        refractory_seconds=refractory_seconds,
    )
    pooled = cohort["pooled"]
    return {
        "j_score": objective.score(pooled),
        "guardrail_pass": objective.passes_guardrail(
            pooled, cohort["patient_summary"]
        ),
        "pooled": pooled,
        "patient_summary": cohort["patient_summary"],
    }


def _save_checkpoint(
    model: torch.nn.Module,
    destination_dir: Path,
    *,
    seed: int,
    config: GiGPOConfig,
) -> tuple[Path, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination_dir / f".gigpo_seed_{seed}.{uuid.uuid4().hex}.tmp"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    destination = destination_dir / f"gigpo_seed_{seed}_{digest[:12]}.pt"
    if destination.exists():
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return destination, digest


def _save_actions(
    actions: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
) -> tuple[Path, str]:
    values = np.asarray(actions, dtype=np.uint8)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"gigpo_seed_{seed}_chb21_{digest[:12]}.npy"
    if not destination.exists():
        temporary = output_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, destination)
    return destination, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--phase-r3-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rollouts-per-group", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--step-advantage-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mode", choices=("mean_norm", "mean_std_norm"), default="mean_std_norm")
    args = parser.parse_args(argv)

    training_artifact = _load_subject(args.artifact_dir, "chb20")
    selection_artifact = _load_subject(args.artifact_dir, "chb21")
    training_timeline = training_artifact.timeline
    selection_timeline = selection_artifact.timeline
    training_enrollment = compute_enrollment_statistics(
        enrollment_probabilities(training_timeline)
    )
    selection_enrollment = compute_enrollment_statistics(
        enrollment_probabilities(selection_timeline)
    )
    reward_config = AlarmRewardConfig()
    objective = SelectionObjective(
        lambda_fa=0.02,
        lambda_latency=0.001,
        minimum_event_sensitivity=0.8,
        minimum_patient_event_sensitivity=0.8,
    )
    config = GiGPOConfig(
        epochs=args.epochs,
        rollouts_per_group=args.rollouts_per_group,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        step_advantage_weight=args.step_advantage_weight,
        gamma=args.gamma,
        mode=args.mode,
    )
    config.validate()

    results: list[dict[str, Any]] = []
    for seed in GIGPO_SEEDS:
        model, history = train_gigpo(
            training_timeline,
            training_enrollment,
            reward_config=reward_config,
            config=config,
            seed=seed,
        )
        actions, rewards = actor_actions(
            model,
            selection_timeline,
            selection_enrollment,
            reward_config=reward_config,
        )
        checkpoint, checkpoint_hash = _save_checkpoint(
            model,
            args.output_dir / "checkpoints",
            seed=seed,
            config=config,
        )
        action_path, action_hash = _save_actions(
            actions,
            args.output_dir / "actions",
            seed=seed,
        )
        result = {
            "seed": seed,
            **_metrics(
                selection_timeline,
                actions,
                objective=objective,
                refractory_seconds=reward_config.refractory_seconds,
            ),
            "reward_components": rewards,
            "history": history,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "action_path": str(action_path),
            "action_sha256": action_hash,
        }
        results.append(result)
        print(
            f"gigpo seed {seed}: J={result['j_score']} "
            f"guardrail={result['guardrail_pass']}",
            flush=True,
        )

    finite = sorted(
        (result for result in results if result["j_score"] is not None),
        key=lambda result: result["j_score"],
    )
    selected = finite[len(finite) // 2] if finite else None
    r3_payload = json.loads(args.phase_r3_result.read_text(encoding="utf-8"))
    comparators = r3_payload["development_comparators"]
    robust = comparators["robust_fixed_rule"]
    supervised = comparators["best_supervised"]
    comparison_score = max(float(robust["j_score"]), float(supervised["j_score"]))
    promoted = bool(
        selected
        and selected["guardrail_pass"]
        and float(selected["j_score"]) > comparison_score
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "GiGPO episode + step-level grouped advantages on cached probabilities",
        "protocol": {
            "training_subject": "chb20",
            "selection_subject": "chb21",
            "final_test_subjects_read": [],
            "development_only": True,
            "group_definition": (
                "episode: one EDF record initial state; "
                "step: (record_group, row) anchor across rollouts of the same record"
            ),
            "advantage": (
                "episode return normalized across same-record rollouts plus "
                "step_advantage_weight x return-to-go normalized within (record, row) "
                "step groups (GiGPO Eq. 3/5/7/8; exact-anchor deviation documented)"
            ),
            "anchor_deviation": (
                "verl-agent hashes exact observations; here the observation embeds this "
                "rollout's alarm history (seconds-since-alarm, refractory remaining), so "
                "exact matching would fragment at divergent alarm rows. The (record, row) "
                "anchor keeps every step group at rollouts_per_group members."
            ),
            "selection_rule": "median chb21 J seed; never best seed",
            "exploration": (
                "sparse alarm init (GRPOConfig.init_logit_bias) so record-group "
                "rollout returns stay distinct under refractory saturation"
            ),
        },
        "training_artifact_id": training_artifact.metadata["artifact_id"],
        "selection_artifact_id": selection_artifact.metadata["artifact_id"],
        "reward_config": vars(reward_config),
        "objective": {
            "lambda_fa": objective.lambda_fa,
            "lambda_latency": objective.lambda_latency,
            "minimum_event_sensitivity": objective.minimum_event_sensitivity,
            "minimum_patient_event_sensitivity": (
                objective.minimum_patient_event_sensitivity
            ),
        },
        "gigpo": {
            "config": config.to_dict(),
            "seeds": list(GIGPO_SEEDS),
            "results": results,
            "selected": selected,
            "promoted": promoted,
        },
        "development_comparators": {
            "robust_fixed_rule": robust,
            "best_supervised_name": comparators["best_supervised_name"],
            "best_supervised": supervised,
            "ppo_selected": r3_payload["ppo"]["selected"],
        },
        "base_probability_invariance": {
            "status": "unchanged",
            "reason": "G3 reads immutable cached probability artifacts only",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    result_path = save_content_addressed_json(
        payload,
        args.output_dir,
        hash_field="result_sha256",
        stem="g3_gigpo",
    )
    print(json.dumps({"result": str(result_path), "gigpo_promoted": promoted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
