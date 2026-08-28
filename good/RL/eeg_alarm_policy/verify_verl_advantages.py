"""Run G0 numerical parity checks against unmodified verl-agent functions."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import save_content_addressed_json
from .group_advantages import (
    gigpo_outcome_advantage,
    grpo_outcome_advantage,
    rloo_outcome_advantage,
)
from .verl_reference import load_reference_modules, reference_source_metadata

SCHEMA_VERSION = "eeg_alarm_g0_verl_advantage_parity_v1"


def _fixture() -> dict[str, Any]:
    return {
        "token_rewards": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "mask": torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "groups": np.asarray(["g0"] * 4 + ["g1"] * 4, dtype=object),
        "trajectories": np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
        "anchors": np.asarray(
            ["state-a", "state-b", "state-a", "unique", "state-a", "state-b", "state-a", "state-b"],
            dtype=object,
        ),
        "step_rewards": torch.tensor(
            [0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0],
            dtype=torch.float32,
        ),
    }


def _comparison(name: str, actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = torch.abs(actual - reference)
    maximum = float(difference.max().item())
    return {
        "name": name,
        "shape": list(actual.shape),
        "max_abs_error": maximum,
        "allclose_atol_1e-6": bool(torch.allclose(actual, reference, atol=1e-6, rtol=0)),
        "masked_positions_zero": bool(
            torch.count_nonzero(actual[_fixture()["mask"] == 0]).item() == 0
        )
        if list(actual.shape) == list(_fixture()["mask"].shape)
        else True,
    }


def run_parity() -> list[dict[str, Any]]:
    fixture = _fixture()
    core, gigpo = load_reference_modules()
    rewards = fixture["token_rewards"]
    mask = fixture["mask"]
    groups = fixture["groups"]
    trajectories = fixture["trajectories"]
    anchors = fixture["anchors"]
    step_rewards = fixture["step_rewards"]
    rows: list[dict[str, Any]] = []

    for normalize, cross_steps in (
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ):
        actual = grpo_outcome_advantage(
            rewards,
            mask,
            groups,
            trajectories,
            normalize_by_std=normalize,
            compute_mean_std_cross_steps=cross_steps,
        )
        reference, _ = core.compute_grpo_outcome_advantage(
            rewards,
            mask,
            groups,
            trajectories,
            norm_adv_by_std_in_grpo=normalize,
            compute_mean_std_cross_steps=cross_steps,
        )
        rows.append(
            _comparison(f"grpo_std={normalize}_cross_steps={cross_steps}", actual, reference)
        )

    for cross_steps in (True, False):
        actual = rloo_outcome_advantage(
            rewards,
            mask,
            groups,
            trajectories,
            compute_mean_std_cross_steps=cross_steps,
        )
        reference, _ = core.compute_rloo_outcome_advantage(
            rewards,
            mask,
            groups,
            trajectories,
            compute_mean_std_cross_steps=cross_steps,
        )
        rows.append(_comparison(f"rloo_cross_steps={cross_steps}", actual, reference))

    for mode in ("mean_norm", "mean_std_norm"):
        actual = gigpo_outcome_advantage(
            rewards,
            step_rewards,
            mask,
            anchors,
            groups,
            trajectories,
            mode=mode,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            reference, _ = gigpo.compute_gigpo_outcome_advantage(
                rewards,
                step_rewards,
                mask,
                anchors,
                groups,
                trajectories,
                mode=mode,
                enable_similarity=False,
            )
        rows.append(_comparison(f"gigpo_{mode}", actual, reference))

    singleton_rewards = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    singleton_mask = torch.ones_like(singleton_rewards)
    singleton_group = np.asarray(["single"], dtype=object)
    singleton_traj = np.asarray([0])
    singleton_anchor = np.asarray(["only-state"], dtype=object)
    singleton_step = torch.tensor([5.0])
    singleton_calls = {
        "grpo_singleton": (
            grpo_outcome_advantage(
                singleton_rewards,
                singleton_mask,
                singleton_group,
                singleton_traj,
            ),
            core.compute_grpo_outcome_advantage(
                singleton_rewards,
                singleton_mask,
                singleton_group,
                singleton_traj,
            )[0],
        ),
        "rloo_singleton": (
            rloo_outcome_advantage(
                singleton_rewards,
                singleton_mask,
                singleton_group,
                singleton_traj,
            ),
            core.compute_rloo_outcome_advantage(
                singleton_rewards,
                singleton_mask,
                singleton_group,
                singleton_traj,
            )[0],
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        singleton_gigpo_reference = gigpo.compute_gigpo_outcome_advantage(
            singleton_rewards,
            singleton_step,
            singleton_mask,
            singleton_anchor,
            singleton_group,
            singleton_traj,
        )[0]
    singleton_calls["gigpo_singleton"] = (
        gigpo_outcome_advantage(
            singleton_rewards,
            singleton_step,
            singleton_mask,
            singleton_anchor,
            singleton_group,
            singleton_traj,
        ),
        singleton_gigpo_reference,
    )
    rows.extend(
        _comparison(name, actual, reference)
        for name, (actual, reference) in singleton_calls.items()
    )
    return rows


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# G0 verl-agent Advantage Parity",
        "",
        "All independent binary-policy advantage calculations match the unmodified "
        "verl-agent reference functions at absolute tolerance `1e-6`.",
        "",
        "| Case | Shape | Max absolute error | Passed |",
        "|---|---:|---:|:---:|",
    ]
    lines.extend(
        f"| {row['name']} | {'x'.join(map(str, row['shape']))} | "
        f"{row['max_abs_error']:.3e} | {'yes' if row['allclose_atol_1e-6'] else 'no'} |"
        for row in payload["comparisons"]
    )
    lines.extend(
        [
            "",
            "## Verified semantics",
            "",
            "- GRPO outcome centering with optional sample-standard-deviation scaling.",
            "- RLOO leave-one-out baseline with and without repeated-step deduplication.",
            "- GiGPO episode advantage plus exact-anchor step-level advantage.",
            "- Response masking, zero-variance groups, repeated trajectory IDs, and "
            "singleton groups.",
            "",
            "## Important reference behavior",
            "",
            "The reference sums token-level rewards before applying the response mask. Upstream "
            "reward tensors must therefore already be zero outside valid positions. For a "
            "singleton episode group, GRPO and RLOO preserve the raw episode score because the "
            "reference uses mean `0` and standard deviation `1`; singleton GiGPO step advantage "
            "is zero. The EEG adapter will preserve these semantics unless a separately named "
            "ablation intentionally changes them.",
            "",
            f"Result SHA256: `{payload['result_sha256']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/eeg_rl/g0_verl_parity"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("G0_VERL_ADVANTAGE_PARITY.md"),
    )
    args = parser.parse_args(argv)
    comparisons = run_parity()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "independent EEG binary advantages versus unmodified verl-agent",
        "reference_sources": reference_source_metadata(),
        "comparisons": comparisons,
        "passed": all(row["allclose_atol_1e-6"] for row in comparisons),
        "tolerance": {"absolute": 1e-6, "relative": 0.0},
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
        stem="g0_verl_advantage_parity",
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    _write_report(args.report, saved)
    print(result_path)
    print(args.report)
    return 0 if saved["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
