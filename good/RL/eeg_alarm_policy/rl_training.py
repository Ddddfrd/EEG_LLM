"""Small tabular and PPO policies for cached EEG alarm observations."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .contracts import ProbabilityTimeline
from .environment import AlarmRecordEnvironment, AlarmRewardConfig, rollout_policy
from .features import EnrollmentStatistics


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discretize_observation(observation: np.ndarray) -> int:
    current = min(9, int(float(observation[7]) * 10))
    local_mean = min(4, int(float(np.mean(observation[:8])) * 5))
    trend_value = float(observation[7] - observation[0])
    trend = min(4, max(0, int((trend_value + 1.0) * 2.5)))
    refractory = int(float(observation[12]) > 0)
    return (((current * 5) + local_mean) * 5 + trend) * 2 + refractory


@dataclass(frozen=True)
class TabularConfig:
    epochs: int = 30
    learning_rate: float = 0.15
    discount: float = 0.99
    epsilon_start: float = 0.3
    epsilon_end: float = 0.02


def train_tabular_q(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
    config: TabularConfig | None = None,
    seed: int = 42,
) -> np.ndarray:
    settings = config or TabularConfig()
    _seed(seed)
    rng = np.random.default_rng(seed)
    q_values = np.zeros((500, 2), dtype=np.float64)
    records = np.unique(timeline.record_indices)
    for epoch in range(settings.epochs):
        epsilon = settings.epsilon_start + (
            settings.epsilon_end - settings.epsilon_start
        ) * epoch / max(1, settings.epochs - 1)
        for record_index in rng.permutation(records):
            environment = AlarmRecordEnvironment(
                timeline,
                record_index=int(record_index),
                enrollment=enrollment,
                reward_config=reward_config,
            )
            observation = environment.reset()
            while True:
                state = discretize_observation(observation)
                action = (
                    int(rng.integers(2))
                    if rng.random() < epsilon
                    else int(np.argmax(q_values[state]))
                )
                next_observation, reward, done, _ = environment.step(action)
                target = reward
                if not done and next_observation is not None:
                    target += settings.discount * np.max(
                        q_values[discretize_observation(next_observation)]
                    )
                q_values[state, action] += settings.learning_rate * (
                    target - q_values[state, action]
                )
                if done:
                    break
                observation = next_observation
    return q_values


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(14, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )
        self.actor = nn.Linear(32, 1)
        self.critic = nn.Linear(32, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        return self.actor(hidden).squeeze(-1), self.critic(hidden).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 20
    update_epochs: int = 4
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    gradient_clip_norm: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _trajectory(
    model: ActorCritic,
    environment: AlarmRecordEnvironment,
    *,
    rng: torch.Generator,
    config: PPOConfig,
) -> dict[str, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    observation = environment.reset()
    while True:
        tensor = torch.from_numpy(observation).float().unsqueeze(0)
        with torch.no_grad():
            logit, value = model(tensor)
            probability = torch.sigmoid(logit)
            action = int(torch.rand((), generator=rng) < probability.squeeze())
            distribution = torch.distributions.Bernoulli(logits=logit)
            log_probability = distribution.log_prob(
                torch.tensor([float(action)])
            )
        next_observation, reward, done, _ = environment.step(action)
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(float(log_probability.item()))
        values.append(float(value.item()))
        rewards.append(float(reward))
        if done:
            break
        observation = next_observation
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + config.discount * next_value - values[index]
        last_advantage = (
            delta
            + config.discount * config.gae_lambda * last_advantage
        )
        advantages[index] = last_advantage
    returns = advantages + np.asarray(values, dtype=np.float32)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "log_probabilities": np.asarray(log_probabilities, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "rewards": np.asarray(rewards, dtype=np.float32),
    }


def train_ppo(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
    config: PPOConfig | None = None,
    seed: int,
) -> tuple[ActorCritic, list[dict[str, float]]]:
    settings = config or PPOConfig()
    _seed(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    records = np.unique(timeline.record_indices)
    history: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        trajectories = [
            _trajectory(
                model,
                AlarmRecordEnvironment(
                    timeline,
                    record_index=int(record_index),
                    enrollment=enrollment,
                    reward_config=reward_config,
                ),
                rng=generator,
                config=settings,
            )
            for record_index in records
        ]
        arrays = {
            key: np.concatenate([trajectory[key] for trajectory in trajectories])
            for key in trajectories[0]
        }
        advantages = arrays["advantages"]
        arrays["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).astype(np.float32)
        size = len(arrays["actions"])
        epoch_losses: list[float] = []
        for _ in range(settings.update_epochs):
            order = torch.randperm(size, generator=generator).numpy()
            for start in range(0, size, settings.minibatch_size):
                selected = order[start : start + settings.minibatch_size]
                observations = torch.from_numpy(arrays["observations"][selected])
                actions = torch.from_numpy(arrays["actions"][selected])
                old_log_probabilities = torch.from_numpy(
                    arrays["log_probabilities"][selected]
                )
                batch_advantages = torch.from_numpy(arrays["advantages"][selected])
                returns = torch.from_numpy(arrays["returns"][selected])
                logits, values = model(observations)
                distribution = torch.distributions.Bernoulli(logits=logits)
                log_probabilities = distribution.log_prob(actions)
                ratio = torch.exp(log_probabilities - old_log_probabilities)
                clipped = torch.clamp(
                    ratio,
                    1.0 - settings.clip_ratio,
                    1.0 + settings.clip_ratio,
                )
                policy_loss = -torch.minimum(
                    ratio * batch_advantages,
                    clipped * batch_advantages,
                ).mean()
                value_loss = torch.square(values - returns).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + settings.value_weight * value_loss
                    - settings.entropy_weight * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings.gradient_clip_norm,
                )
                optimizer.step()
                epoch_losses.append(float(loss.item()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_loss": float(np.mean(epoch_losses)),
                "mean_episode_return": float(
                    np.mean([np.sum(item["rewards"]) for item in trajectories])
                ),
            }
        )
    return model, history


def actor_actions(
    model: ActorCritic,
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    def policy(observation: np.ndarray) -> int:
        with torch.no_grad():
            logit, _ = model(torch.from_numpy(observation).float().unsqueeze(0))
        return int(float(logit.item()) >= 0.0)

    return rollout_policy(
        timeline,
        enrollment,
        policy,
        reward_config=reward_config,
    )
