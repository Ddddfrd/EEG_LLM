"""Step-level grouped advantages (GiGPO, G3) for binary alarm policies.

GiGPO (Group-in-Group, arXiv:2505.10978) composes two group-relative
advantages:

- episode level: the record return standardized across rollouts from the
  same record initial state (identical to G1's GRPO advantage);
- step level: the return-to-go standardized across rollouts anchored on the
  same decision point, so a hit/miss outcome credits the alarm rows that
  produced it instead of being broadcast over the whole ~900-step record.

Reference math: ``verl-agent-master/gigpo/core_gigpo.py`` (read-only) and the
parity-verified ports in :mod:`eeg_alarm_policy.group_advantages`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .contracts import ProbabilityTimeline
from .environment import AlarmRewardConfig
from .features import EnrollmentStatistics, causal_probability_histories
from .group_advantages import _step_advantage, exact_step_group_ids
from .grpo_training import (
    BernoulliActor,
    GRPOConfig,
    _seed,
    ppo_update_epochs,
    rollout_record_group,
    trajectory_grpo_advantages,
)


@dataclass(frozen=True)
class GiGPOConfig(GRPOConfig):
    """GRPO settings plus the step-level grouping controls."""

    # Weight of the step-level (return-to-go) advantage relative to the
    # episode-level advantage; 1.0 follows GiGPO Eq. 8.
    step_advantage_weight: float = 1.0
    # Discount for the step-level return-to-go (verl-agent default 1.0).
    gamma: float = 1.0
    # "mean_std_norm" divides both levels by the group std (verl-agent
    # default); "mean_norm" only mean-centers.
    mode: str = "mean_std_norm"

    def validate(self) -> None:
        super().validate()
        if self.step_advantage_weight < 0:
            raise ValueError("step_advantage_weight must be non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        if self.mode not in {"mean_norm", "mean_std_norm"}:
            raise ValueError("mode must be mean_norm or mean_std_norm")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted return-to-go per step (GiGPO Eq. 5)."""
    values = np.asarray(rewards, dtype=np.float64)
    output = np.empty_like(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + gamma * running
        output[index] = running
    return output.astype(np.float32)


def gigpo_advantages(
    trajectories: list[dict[str, Any]],
    *,
    epsilon: float,
    normalize_by_std: bool,
    gamma: float,
    step_advantage_weight: float,
) -> dict[str, np.ndarray]:
    """Compose episode and step advantages for every flattened step.

    The step anchor is ``(record_group_id, row)`` rather than verl-agent's
    exact-observation hash: the 14-dim observation already encodes this
    rollout's own alarm history (seconds-since-alarm, refractory remaining),
    so exact matching would fragment precisely at the divergent alarm rows
    where the counterfactual comparison is needed. Anchoring on the record
    row keeps every step group at ``rollouts_per_group`` members.
    """
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    episode = trajectory_grpo_advantages(
        trajectories,
        epsilon=epsilon,
        normalize_by_std=normalize_by_std,
    )
    episode_per_step = np.concatenate(
        [
            np.full(len(trajectory["actions"]), advantage, dtype=np.float32)
            for trajectory, advantage in zip(trajectories, episode, strict=True)
        ]
    )
    step_rewards = np.concatenate(
        [return_to_go(trajectory["rewards"], gamma) for trajectory in trajectories]
    )
    episode_group_ids = [
        trajectory["group_id"]
        for trajectory in trajectories
        for _ in trajectory["actions"]
    ]
    anchors = [row for trajectory in trajectories for row in range(len(trajectory["actions"]))]
    step_groups = exact_step_group_ids(anchors, episode_group_ids)
    mask = torch.ones((len(step_rewards), 1), dtype=torch.float32)
    step = (
        _step_advantage(
            torch.from_numpy(step_rewards),
            mask,
            step_groups,
            epsilon=epsilon,
            normalize_by_std=normalize_by_std,
        )
        .squeeze(-1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return {
        "episode": episode,
        "episode_per_step": episode_per_step,
        "step_rewards": step_rewards,
        "step": step,
        "advantages": episode_per_step + step_advantage_weight * step,
    }


def step_group_statistics(
    trajectories: list[dict[str, Any]],
    step_rewards: np.ndarray,
    *,
    gamma: float,
) -> dict[str, float]:
    """Diagnostics for the (record, row) step groups."""
    rows_by_group: dict[tuple[str, int], list[float]] = {}
    offset = 0
    for trajectory in trajectories:
        length = len(trajectory["actions"])
        for row in range(length):
            rows_by_group.setdefault((trajectory["group_id"], row), []).append(
                float(step_rewards[offset + row])
            )
        offset += length
    stds = np.asarray(
        [
            np.std(values, ddof=1) if len(values) > 1 else 0.0
            for values in rows_by_group.values()
        ],
        dtype=np.float64,
    )
    sizes = np.asarray([len(values) for values in rows_by_group.values()], dtype=np.float64)
    return {
        "mean_step_group_size": float(sizes.mean()),
        "active_step_group_fraction": float(np.mean(stds > 1e-12)),
        "mean_step_group_return_std": float(stds.mean()),
    }


def _flatten_steps(
    trajectories: list[dict[str, Any]],
    advantages: np.ndarray,
) -> dict[str, np.ndarray]:
    flattened = {
        "observations": np.concatenate(
            [trajectory["observations"] for trajectory in trajectories]
        ),
        "actions": np.concatenate([trajectory["actions"] for trajectory in trajectories]),
        "log_probabilities": np.concatenate(
            [trajectory["log_probabilities"] for trajectory in trajectories]
        ),
    }
    if len(advantages) != len(flattened["actions"]):
        raise ValueError("step advantages must align with flattened steps")
    flattened["advantages"] = advantages
    return flattened


def train_gigpo(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    *,
    reward_config: AlarmRewardConfig,
    config: GiGPOConfig | None = None,
    seed: int,
) -> tuple[BernoulliActor, list[dict[str, float]]]:
    """Train the actor with GiGPO's episode + step group-relative advantages."""
    settings = config or GiGPOConfig()
    settings.validate()
    _seed(seed)
    model = BernoulliActor(settings.hidden_size, settings.init_logit_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    records = np.unique(timeline.record_indices)
    histories = causal_probability_histories(timeline, enrollment)
    normalize_by_std = settings.mode == "mean_std_norm"
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
        advantages = gigpo_advantages(
            trajectories,
            epsilon=settings.advantage_epsilon,
            normalize_by_std=normalize_by_std,
            gamma=settings.gamma,
            step_advantage_weight=settings.step_advantage_weight,
        )
        arrays = _flatten_steps(trajectories, advantages["advantages"])
        update_metrics = ppo_update_epochs(
            model,
            optimizer,
            arrays,
            settings=settings,
            generator=generator,
        )
        returns_by_group: dict[str, list[float]] = {}
        for trajectory in trajectories:
            returns_by_group.setdefault(trajectory["group_id"], []).append(
                float(trajectory["return"])
            )
        group_stds = np.asarray(
            [np.std(values, ddof=1) for values in returns_by_group.values()],
            dtype=np.float64,
        )
        statistics = step_group_statistics(
            trajectories, advantages["step_rewards"], gamma=settings.gamma
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_loss": float(np.mean(update_metrics["losses"])),
                "mean_entropy": float(np.mean(update_metrics["entropies"])),
                "approximate_kl": float(np.mean(update_metrics["approximate_kls"])),
                "clip_fraction": float(np.mean(update_metrics["clip_fractions"])),
                "mean_trajectory_return": float(
                    np.mean([trajectory["return"] for trajectory in trajectories])
                ),
                "mean_group_return_std": float(group_stds.mean()),
                "active_group_fraction": float(np.mean(group_stds > 1e-12)),
                "mean_absolute_advantage": float(
                    np.mean(np.abs(advantages["advantages"]))
                ),
                "mean_absolute_episode_advantage": float(
                    np.mean(np.abs(advantages["episode_per_step"]))
                ),
                "mean_absolute_step_advantage": float(np.mean(np.abs(advantages["step"]))),
                **statistics,
            }
        )
    return model, history
