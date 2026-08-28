from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import torch

from eeg_alarm_policy.group_advantages import (
    gigpo_outcome_advantage,
    grpo_outcome_advantage,
    rloo_outcome_advantage,
)
from eeg_alarm_policy.verify_verl_advantages import run_parity
from eeg_alarm_policy.verl_reference import load_reference_modules


def test_grpo_and_rloo_have_expected_two_sample_values() -> None:
    rewards = torch.tensor([[1.0], [3.0]])
    mask = torch.ones_like(rewards)
    groups = np.asarray(["record", "record"], dtype=object)
    trajectories = np.asarray([0, 1])

    centered = grpo_outcome_advantage(
        rewards,
        mask,
        groups,
        trajectories,
        normalize_by_std=False,
    )
    normalized = grpo_outcome_advantage(rewards, mask, groups, trajectories)
    rloo = rloo_outcome_advantage(rewards, mask, groups, trajectories)

    assert torch.equal(centered, torch.tensor([[-1.0], [1.0]]))
    assert torch.allclose(
        normalized,
        torch.tensor([[-2**-0.5], [2**-0.5]]),
        atol=1e-6,
    )
    assert torch.equal(rloo, torch.tensor([[-2.0], [2.0]]))


def test_gigpo_adds_episode_and_exact_anchor_step_advantages() -> None:
    rewards = torch.tensor([[1.0], [3.0]])
    step_rewards = torch.tensor([0.0, 2.0])
    mask = torch.ones_like(rewards)
    groups = np.asarray(["record", "record"], dtype=object)
    trajectories = np.asarray([0, 1])
    anchors = np.asarray(["same-state", "same-state"], dtype=object)

    advantages = gigpo_outcome_advantage(
        rewards,
        step_rewards,
        mask,
        anchors,
        groups,
        trajectories,
        mode="mean_norm",
    )

    assert torch.equal(advantages, torch.tensor([[-2.0], [2.0]]))


def test_mask_and_singleton_semantics_match_reference_contract() -> None:
    rewards = torch.tensor([[2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    groups = np.asarray(["single"], dtype=object)
    trajectories = np.asarray([0])
    anchors = np.asarray(["only"], dtype=object)

    grpo = grpo_outcome_advantage(rewards, mask, groups, trajectories)
    rloo = rloo_outcome_advantage(rewards, mask, groups, trajectories)
    gigpo = gigpo_outcome_advantage(
        rewards,
        torch.tensor([5.0]),
        mask,
        anchors,
        groups,
        trajectories,
    )

    expected = torch.tensor([[2.0, 0.0]])
    assert torch.allclose(grpo, expected, atol=2e-6)
    assert torch.equal(rloo, expected)
    assert torch.allclose(gigpo, expected, atol=2e-6)


def test_all_g0_cases_match_unmodified_verl_agent() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        rows = run_parity()

    assert len(rows) == 11
    assert all(row["allclose_atol_1e-6"] for row in rows)
    assert max(row["max_abs_error"] for row in rows) <= 1e-6


def test_reference_loader_restores_global_verl_modules() -> None:
    before = {
        name: sys.modules.get(name)
        for name in ("verl", "verl.utils", "verl.utils.torch_functional")
    }

    load_reference_modules()

    for name, previous in before.items():
        assert sys.modules.get(name) is previous
