from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_alarm_policy.environment import AlarmRewardConfig
from eeg_alarm_policy.features import compute_enrollment_statistics
from eeg_alarm_policy.grpo_training import (
    BernoulliActor,
    GRPOConfig,
    actor_actions,
    rollout_record_group,
    train_grpo,
    trajectory_grpo_advantages,
)


def test_grpo_config_requires_a_group() -> None:
    with pytest.raises(ValueError, match="at least two"):
        GRPOConfig(rollouts_per_group=1).validate()


def test_init_logit_bias_is_applied_and_validated() -> None:
    model = BernoulliActor(hidden_size=8, init_logit_bias=-3.0)
    assert float(model.network[-1].bias.item()) == -3.0
    with pytest.raises(ValueError, match="finite"):
        GRPOConfig(init_logit_bias=float("nan")).validate()
    GRPOConfig(init_logit_bias=-3.0).validate()


def test_grouped_rollouts_share_record_and_sample_distinct_actions(sample_timeline) -> None:
    model = BernoulliActor()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    trajectories = rollout_record_group(
        model,
        sample_timeline,
        enrollment,
        record_index=0,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        rollouts_per_group=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(trajectories) == 8
    assert {trajectory["group_id"] for trajectory in trajectories} == {
        "chb99:record:0"
    }
    assert all(trajectory["observations"].shape == (4, 14) for trajectory in trajectories)
    patterns = {tuple(trajectory["actions"].tolist()) for trajectory in trajectories}
    assert len(patterns) > 1
    assert all(np.isfinite(trajectory["log_probabilities"]).all() for trajectory in trajectories)


def test_trajectory_advantages_are_centered_within_each_record() -> None:
    trajectories = [
        {"return": value, "group_id": group, "trajectory_id": f"t{index}"}
        for index, (group, value) in enumerate(
            [("a", 1.0), ("a", 2.0), ("a", 3.0), ("b", -2.0), ("b", 2.0)]
        )
    ]
    advantages = trajectory_grpo_advantages(trajectories)
    assert advantages.shape == (5,)
    assert float(advantages[:3].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(advantages[3:].mean()) == pytest.approx(0.0, abs=1e-6)


def test_grpo_training_is_deterministic_and_inference_is_finite(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    config = GRPOConfig(
        epochs=2,
        rollouts_per_group=4,
        update_epochs=1,
        minibatch_size=32,
    )
    first, first_history = train_grpo(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        config=config,
        seed=9,
    )
    second, second_history = train_grpo(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
        config=config,
        seed=9,
    )
    assert first_history == second_history
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    actions, rewards = actor_actions(
        first,
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=4.0),
    )
    assert actions.shape == (sample_timeline.row_count,)
    assert set(np.unique(actions)).issubset({0, 1})
    assert all(np.isfinite(value) for value in rewards.values())
