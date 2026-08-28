from __future__ import annotations

import numpy as np

from eeg_alarm_policy.environment import (
    AlarmRecordEnvironment,
    AlarmRewardConfig,
    rollout_policy,
)
from eeg_alarm_policy.features import compute_enrollment_statistics


def test_environment_observation_hides_labels_and_resets(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    environment = AlarmRecordEnvironment(
        sample_timeline,
        record_index=0,
        enrollment=enrollment,
        reward_config=AlarmRewardConfig(refractory_seconds=0),
    )
    observation = environment.reset()
    assert observation.shape == (14,)
    assert observation[-1] == 1.0
    assert not np.shares_memory(observation, sample_timeline.labels)


def test_hit_false_alarm_and_miss_rewards(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2, 0.1])
    config = AlarmRewardConfig(
        hit_reward=1,
        miss_penalty=1,
        false_alarm_penalty=0.1,
        latency_penalty_per_minute=0,
        refractory_seconds=0,
    )
    actions, rewards = rollout_policy(
        sample_timeline,
        enrollment,
        lambda observation: int(observation[7] >= 0.8),
        reward_config=config,
    )
    assert actions.shape == (sample_timeline.row_count,)
    assert rewards["hit"] == 2.0
    assert rewards["miss"] == 0.0
    assert rewards["false_alarm"] < 0


def test_never_alarm_receives_miss_penalties(sample_timeline) -> None:
    enrollment = compute_enrollment_statistics([0.1, 0.2])
    _, rewards = rollout_policy(
        sample_timeline,
        enrollment,
        lambda observation: 0,
        reward_config=AlarmRewardConfig(latency_penalty_per_minute=0),
    )
    assert rewards["miss"] == -2.0
    assert rewards["hit"] == 0.0
