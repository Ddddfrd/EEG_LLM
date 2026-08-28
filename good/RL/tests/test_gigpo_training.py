from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_alarm_policy.environment import AlarmRewardConfig
from eeg_alarm_policy.features import compute_enrollment_statistics
from eeg_alarm_policy.gigpo_training import (
    GiGPOConfig,
    gigpo_advantages,
    return_to_go,
    step_group_statistics,
    train_gigpo,
)
from eeg_alarm_policy.grpo_training import (
    BernoulliActor,
    rollout_record_group,
    trajectory_grpo_advantages,
)


def test_gigpo_config_validates_new_fields() -> None:
    with pytest.raises(ValueError, match="step_advantage_weight"):
        GiGPOConfig(step_advantage_weight=-1.0).validate()
    with pytest.raises(ValueError, match="gamma"):
        GiGPOConfig(gamma=0.0).validate()
    with pytest.raises(ValueError, match="mode"):
        GiGPOConfig(mode="bogus").validate()
    GiGPOConfig().validate()


def test_return_to_go_matches_manual_computation() -> None:
    rewards = np.asarray([1.0, -0.02, 0.5, 0.0], dtype=np.float32)
    unit = return_to_go(rewards, gamma=1.0)
    assert np.allclose(unit, [1.48, 0.48, 0.5, 0.0], atol=1e-6)
    discounted = return_to_go(rewards, gamma=0.5)
    assert np.allclose(discounted, [1.115, 0.23, 0.5, 0.0], atol=1e-6)


def _sample_trajectories(sample_timeline) -> list[dict]:
    model = BernoulliActor()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    return rollout_record_group(
        model,
        sample_timeline,
        enrollment,
        record_index=0,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        rollouts_per_group=8,
        generator=torch.Generator().manual_seed(7),
    )


def test_step_groups_are_anchored_on_record_rows(sample_timeline) -> None:
    trajectories = _sample_trajectories(sample_timeline)
    advantages = gigpo_advantages(
        trajectories,
        epsilon=1e-6,
        normalize_by_std=True,
        gamma=1.0,
        step_advantage_weight=1.0,
    )
    statistics = step_group_statistics(
        trajectories, advantages["step_rewards"], gamma=1.0
    )
    assert statistics["mean_step_group_size"] == pytest.approx(8.0)
    total_steps = sum(len(trajectory["actions"]) for trajectory in trajectories)
    assert advantages["advantages"].shape == (total_steps,)
    assert advantages["episode_per_step"].shape == (total_steps,)
    assert advantages["step"].shape == (total_steps,)


def test_zero_step_weight_reduces_to_the_g1_episode_advantage(sample_timeline) -> None:
    trajectories = _sample_trajectories(sample_timeline)
    advantages = gigpo_advantages(
        trajectories,
        epsilon=1e-6,
        normalize_by_std=True,
        gamma=1.0,
        step_advantage_weight=0.0,
    )
    episode = trajectory_grpo_advantages(trajectories)
    broadcast = advantages["episode_per_step"]
    offset = 0
    for trajectory, value in zip(trajectories, episode, strict=True):
        assert np.allclose(broadcast[offset : offset + len(trajectory["actions"])], value)
        offset += len(trajectory["actions"])
    assert np.allclose(advantages["advantages"], broadcast, atol=1e-6)


def test_step_advantage_credits_rows_with_divergent_futures(sample_timeline) -> None:
    trajectories = _sample_trajectories(sample_timeline)
    baseline = gigpo_advantages(
        trajectories,
        epsilon=1e-6,
        normalize_by_std=True,
        gamma=1.0,
        step_advantage_weight=1.0,
    )
    # Give rollout 0 a hit only it earned at row 1: its step advantage there
    # must rise, the group must stay mean-centered, and the episode part of
    # every other trajectory must be untouched.
    rewarded = [
        dict(trajectory, rewards=trajectory["rewards"].copy())
        for trajectory in trajectories
    ]
    rewarded[0]["rewards"][1] += 1.0
    perturbed = gigpo_advantages(
        rewarded,
        epsilon=1e-6,
        normalize_by_std=True,
        gamma=1.0,
        step_advantage_weight=1.0,
    )
    lengths = [len(trajectory["actions"]) for trajectory in trajectories]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    row1_indices = starts + 1
    assert perturbed["step"][row1_indices[0]] > baseline["step"][row1_indices[0]]
    group_at_row1 = perturbed["step"][row1_indices]
    assert float(np.mean(group_at_row1)) == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(perturbed["episode_per_step"], baseline["episode_per_step"])


def test_training_is_deterministic_and_history_has_step_diagnostics(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    config = GiGPOConfig(
        epochs=2,
        rollouts_per_group=4,
        update_epochs=1,
        minibatch_size=32,
    )
    first, first_history = train_gigpo(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        config=config,
        seed=9,
    )
    second, second_history = train_gigpo(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        config=config,
        seed=9,
    )
    assert first_history == second_history
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    for entry in first_history:
        assert entry["mean_step_group_size"] == pytest.approx(4.0)
        assert np.isfinite(entry["mean_absolute_step_advantage"])
        assert np.isfinite(entry["mean_absolute_episode_advantage"])
