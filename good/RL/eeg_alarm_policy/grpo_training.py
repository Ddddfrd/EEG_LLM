"""Grouped record rollouts and critic-free GRPO for binary alarm policies."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .contracts import ProbabilityTimeline
from .environment import AlarmRecordEnvironment, AlarmRewardConfig, rollout_policy
from .features import EnrollmentStatistics, causal_probability_histories
from .group_advantages import grpo_outcome_advantage


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BernoulliActor(nn.Module):
    """Small actor over the frozen 14-dimensional causal observation."""

    def __init__(self, hidden_size: int = 32, init_logit_bias: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(14, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        with torch.no_grad():
            nn.init.constant_(self.network[-1].bias, float(init_logit_bias))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


@dataclass(frozen=True)
class GRPOConfig:
    epochs: int = 8
    rollouts_per_group: int = 8
    update_epochs: int = 2
    minibatch_size: int = 2048
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    gradient_clip_norm: float = 0.5
    advantage_epsilon: float = 1e-6
    normalize_advantage_by_std: bool = True
    hidden_size: int = 32
    # A dense random-start policy (p~0.5) alarms so often that the 300 s
    # refractory saturates: every rollout of a record accepts the same alarm
    # schedule, all within-group returns come out identical, and the GRPO
    # advantage is identically zero. Starting sparse keeps rollout outcomes
    # distinct so the group-relative signal exists (measured on chb20: dense
    # init 0/29 active groups; p<=0.12 -> 29/29).
    init_logit_bias: float = -3.0

    def validate(self) -> None:
        if not np.isfinite(self.init_logit_bias):
            raise ValueError("init_logit_bias must be finite")
        if self.epochs < 1 or self.update_epochs < 1:
            raise ValueError("epochs and update_epochs must be positive")
        if self.rollouts_per_group < 2:
            raise ValueError("GRPO requires at least two rollouts per record group")
        if self.minibatch_size < 1 or self.hidden_size < 1:
            raise ValueError("minibatch_size and hidden_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning rate and gradient clipping must be positive")
        if not 0 <= self.clip_ratio < 1 or self.entropy_weight < 0:
            raise ValueError("clip ratio or entropy weight is invalid")
        if self.advantage_epsilon <= 0:
            raise ValueError("advantage epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rollout_record_group(
    model: BernoulliActor,
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    record_index: int,
    reward_config: AlarmRewardConfig,
    rollouts_per_group: int,
    generator: torch.Generator,
    histories: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Sample multiple trajectories from the exact same record initial state."""
    if rollouts_per_group < 2:
        raise ValueError("rollouts_per_group must be at least two")
    cached_histories = (
        causal_probability_histories(timeline, enrollment)
        if histories is None
        else histories
    )
    environments = [
        AlarmRecordEnvironment(
            timeline,
            record_index=record_index,
            enrollment=enrollment,
            reward_config=reward_config,
            histories=cached_histories,
        )
        for _ in range(rollouts_per_group)
    ]
    observations = np.stack([environment.reset() for environment in environments])
    trajectories = [
        {
            "observations": [],
            "actions": [],
            "log_probabilities": [],
            "rewards": [],
            "group_id": f"{timeline.subject_id}:record:{record_index}",
            "trajectory_id": f"{timeline.subject_id}:record:{record_index}:rollout:{index}",
        }
        for index in range(rollouts_per_group)
    ]
    while True:
        observation_tensor = torch.from_numpy(observations).float()
        with torch.no_grad():
            logits = model(observation_tensor)
            probabilities = torch.sigmoid(logits)
            actions = (torch.rand(logits.shape, generator=generator) < probabilities).float()
            distribution = torch.distributions.Bernoulli(logits=logits)
            log_probabilities = distribution.log_prob(actions)
        next_observations: list[np.ndarray] = []
        all_done = True
        for index, environment in enumerate(environments):
            next_observation, reward, done, _ = environment.step(int(actions[index].item()))
            trajectories[index]["observations"].append(observations[index])
            trajectories[index]["actions"].append(float(actions[index].item()))
            trajectories[index]["log_probabilities"].append(
                float(log_probabilities[index].item())
            )
            trajectories[index]["rewards"].append(float(reward))
            if not done:
                if next_observation is None:
                    raise RuntimeError("Non-terminal rollout lacks an observation")
                next_observations.append(next_observation)
                all_done = False
        if all_done:
            break
        if len(next_observations) != rollouts_per_group:
            raise RuntimeError("Grouped environments ended at different time steps")
        observations = np.stack(next_observations)

    result: list[dict[str, Any]] = []
    for trajectory in trajectories:
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
        result.append(
            {
                **trajectory,
                "observations": np.asarray(trajectory["observations"], dtype=np.float32),
                "actions": np.asarray(trajectory["actions"], dtype=np.float32),
                "log_probabilities": np.asarray(
                    trajectory["log_probabilities"], dtype=np.float32
                ),
                "rewards": rewards,
                "return": float(rewards.sum(dtype=np.float64)),
            }
        )
    return result


def trajectory_grpo_advantages(
    trajectories: list[dict[str, Any]],
    *,
    epsilon: float = 1e-6,
    normalize_by_std: bool = True,
) -> np.ndarray:
    """Compute one group-relative outcome advantage per trajectory."""
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    rewards = torch.tensor(
        [[float(trajectory["return"])] for trajectory in trajectories],
        dtype=torch.float32,
    )
    mask = torch.ones_like(rewards)
    advantages = grpo_outcome_advantage(
        rewards,
        mask,
        [trajectory["group_id"] for trajectory in trajectories],
        [trajectory["trajectory_id"] for trajectory in trajectories],
        epsilon=epsilon,
        normalize_by_std=normalize_by_std,
        compute_mean_std_cross_steps=False,
    )
    return advantages.squeeze(-1).cpu().numpy().astype(np.float32)


def _flatten_trajectories(
    trajectories: list[dict[str, Any]],
    trajectory_advantages: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(trajectories) != len(trajectory_advantages):
        raise ValueError("trajectory advantages must align with trajectories")
    return {
        "observations": np.concatenate(
            [trajectory["observations"] for trajectory in trajectories]
        ),
        "actions": np.concatenate([trajectory["actions"] for trajectory in trajectories]),
        "log_probabilities": np.concatenate(
            [trajectory["log_probabilities"] for trajectory in trajectories]
        ),
        "advantages": np.concatenate(
            [
                np.full(len(trajectory["actions"]), advantage, dtype=np.float32)
                for trajectory, advantage in zip(
                    trajectories, trajectory_advantages, strict=True
                )
            ]
        ),
    }


def ppo_update_epochs(
    model: BernoulliActor,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    *,
    settings: GRPOConfig,
    generator: torch.Generator,
) -> dict[str, list[float]]:
    """Run clipped-surrogate minibatch updates over flattened step arrays."""
    size = len(arrays["actions"])
    losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
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
            logits = model(observations)
            distribution = torch.distributions.Bernoulli(logits=logits)
            log_probabilities = distribution.log_prob(actions)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - settings.clip_ratio,
                1.0 + settings.clip_ratio,
            )
            policy_loss = -torch.minimum(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages,
            ).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss - settings.entropy_weight * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            with torch.no_grad():
                losses.append(float(loss.item()))
                entropies.append(float(entropy.item()))
                approximate_kls.append(
                    float((old_log_probabilities - log_probabilities).mean().item())
                )
                clip_fractions.append(
                    float((torch.abs(ratio - 1.0) > settings.clip_ratio).float().mean())
                )
    return {
        "losses": losses,
        "entropies": entropies,
        "approximate_kls": approximate_kls,
        "clip_fractions": clip_fractions,
    }


def train_grpo(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
    config: GRPOConfig | None = None,
    seed: int,
) -> tuple[BernoulliActor, list[dict[str, float]]]:
    """Train a critic-free actor from grouped record-level outcome rewards."""
    settings = config or GRPOConfig()
    settings.validate()
    _seed(seed)
    model = BernoulliActor(settings.hidden_size, settings.init_logit_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    records = np.unique(timeline.record_indices)
    histories = causal_probability_histories(timeline, enrollment)
    history: list[dict[str, float]] = []

    for epoch in range(settings.epochs):
        trajectories: list[dict[str, Any]] = []
        for record_index in records:
            trajectories.extend(
                rollout_record_group(
                    model,
                    timeline,
                    enrollment,
                    record_index=int(record_index),
                    reward_config=reward_config,
                    rollouts_per_group=settings.rollouts_per_group,
                    generator=generator,
                    histories=histories,
                )
            )
        advantages = trajectory_grpo_advantages(
            trajectories,
            epsilon=settings.advantage_epsilon,
            normalize_by_std=settings.normalize_advantage_by_std,
        )
        arrays = _flatten_trajectories(trajectories, advantages)
        update_metrics = ppo_update_epochs(
            model,
            optimizer,
            arrays,
            settings=settings,
            generator=generator,
        )
        losses = update_metrics["losses"]
        entropies = update_metrics["entropies"]
        approximate_kls = update_metrics["approximate_kls"]
        clip_fractions = update_metrics["clip_fractions"]

        returns_by_group: dict[str, list[float]] = {}
        for trajectory in trajectories:
            returns_by_group.setdefault(trajectory["group_id"], []).append(
                float(trajectory["return"])
            )
        group_stds = np.asarray(
            [np.std(values, ddof=1) for values in returns_by_group.values()],
            dtype=np.float64,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_loss": float(np.mean(losses)),
                "mean_entropy": float(np.mean(entropies)),
                "approximate_kl": float(np.mean(approximate_kls)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "mean_trajectory_return": float(
                    np.mean([trajectory["return"] for trajectory in trajectories])
                ),
                "mean_group_return_std": float(group_stds.mean()),
                "active_group_fraction": float(np.mean(group_stds > 1e-12)),
                "mean_absolute_advantage": float(np.mean(np.abs(advantages))),
            }
        )
    return model, history


def actor_actions(
    model: BernoulliActor,
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run deterministic logit>=0 inference for development evaluation."""

    def policy(observation: np.ndarray) -> int:
        with torch.no_grad():
            logit = model(torch.from_numpy(observation).float().unsqueeze(0))
        return int(float(logit.item()) >= 0.0)

    return rollout_policy(
        timeline,
        enrollment,
        policy,
        reward_config=reward_config,
    )
