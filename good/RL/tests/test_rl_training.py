from __future__ import annotations

import numpy as np
import torch

from eeg_alarm_policy.environment import AlarmRewardConfig
from eeg_alarm_policy.features import compute_enrollment_statistics
from eeg_alarm_policy.rl_training import (
    ActorCritic,
    TabularConfig,
    discretize_observation,
    train_tabular_q,
)


def test_actor_critic_contract() -> None:
    model = ActorCritic()
    logits, values = model(torch.zeros((3, 14)))
    assert logits.shape == (3,)
    assert values.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000


def test_tabular_training_is_deterministic(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    config = TabularConfig(epochs=2)
    first = train_tabular_q(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(),
        config=config,
        seed=7,
    )
    second = train_tabular_q(
        sample_timeline,
        enrollment,
        reward_config=AlarmRewardConfig(),
        config=config,
        seed=7,
    )
    assert first.shape == (500, 2)
    assert np.array_equal(first, second)
    assert 0 <= discretize_observation(np.zeros(14, dtype=np.float32)) < 500
