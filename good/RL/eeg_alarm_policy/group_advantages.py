"""Critic-free group-relative advantages for binary alarm trajectories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _identifier(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_inputs(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray,
    trajectory_ids: Sequence[Any] | np.ndarray,
) -> tuple[list[Any], list[Any]]:
    if token_level_rewards.ndim != 2:
        raise ValueError("token_level_rewards must have shape (rows, response_length)")
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError("response_mask must match token_level_rewards")
    rows = token_level_rewards.shape[0]
    groups = [_identifier(value) for value in group_ids]
    trajectories = [_identifier(value) for value in trajectory_ids]
    if len(groups) != rows or len(trajectories) != rows:
        raise ValueError("group and trajectory identifiers must match reward rows")
    if not torch.isfinite(token_level_rewards).all():
        raise ValueError("token_level_rewards must be finite")
    if not torch.isfinite(response_mask).all():
        raise ValueError("response_mask must be finite")
    return groups, trajectories


def _group_samples(
    scores: torch.Tensor,
    groups: list[Any],
    trajectories: list[Any],
    *,
    compute_mean_std_cross_steps: bool,
) -> dict[Any, list[torch.Tensor]]:
    samples: dict[Any, list[torch.Tensor]] = defaultdict(list)
    seen: set[tuple[Any, Any]] = set()
    for row, score in enumerate(scores):
        key = (groups[row], trajectories[row])
        if key in seen:
            continue
        samples[groups[row]].append(score)
        if not compute_mean_std_cross_steps:
            seen.add(key)
    return samples


def _sample_mean_std(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(values)
    return stacked.mean(), stacked.std(unbiased=True)


def grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray,
    trajectory_ids: Sequence[Any] | np.ndarray,
    *,
    epsilon: float = 1e-6,
    normalize_by_std: bool = True,
    compute_mean_std_cross_steps: bool = True,
) -> torch.Tensor:
    """Match verl-agent GRPO outcome advantage semantics."""
    groups, trajectories = _validate_inputs(
        token_level_rewards,
        response_mask,
        group_ids,
        trajectory_ids,
    )
    scores = token_level_rewards.sum(dim=-1).clone()
    samples = _group_samples(
        scores,
        groups,
        trajectories,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
    )
    statistics: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
    for group, values in samples.items():
        if len(values) == 1:
            statistics[group] = (scores.new_tensor(0.0), scores.new_tensor(1.0))
        else:
            statistics[group] = _sample_mean_std(values)
    for row, group in enumerate(groups):
        mean, std = statistics[group]
        scores[row] = (
            (scores[row] - mean) / (std + epsilon)
            if normalize_by_std
            else scores[row] - mean
        )
    return scores.unsqueeze(-1) * response_mask


def rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any] | np.ndarray,
    trajectory_ids: Sequence[Any] | np.ndarray,
    *,
    compute_mean_std_cross_steps: bool = True,
) -> torch.Tensor:
    """Match verl-agent leave-one-out outcome advantage semantics."""
    groups, trajectories = _validate_inputs(
        token_level_rewards,
        response_mask,
        group_ids,
        trajectory_ids,
    )
    scores = token_level_rewards.sum(dim=-1).clone()
    samples = _group_samples(
        scores,
        groups,
        trajectories,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
    )
    means = {
        group: (
            scores.new_tensor(0.0)
            if len(values) == 1
            else torch.stack(values).mean()
        )
        for group, values in samples.items()
    }
    for row, group in enumerate(groups):
        count = len(samples[group])
        if count > 1:
            scores[row] = count * (scores[row] - means[group]) / (count - 1)
    return scores.unsqueeze(-1) * response_mask


def _hashable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
        return _identifier(value)
    if isinstance(value, np.ndarray):
        return tuple(_hashable(item) for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported anchor observation type: {type(value)}")


def exact_step_group_ids(
    anchor_observations: Sequence[Any] | np.ndarray,
    episode_group_ids: Sequence[Any] | np.ndarray,
) -> list[tuple[Any, Any]]:
    """Create deterministic labels equivalent to verl-agent exact grouping."""
    anchors = list(anchor_observations)
    groups = [_identifier(value) for value in episode_group_ids]
    if len(anchors) != len(groups):
        raise ValueError("anchor observations and episode groups must align")
    return [
        (group, _hashable(anchor))
        for group, anchor in zip(groups, anchors, strict=True)
    ]


def _step_advantage(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: Sequence[Any],
    *,
    epsilon: float,
    normalize_by_std: bool,
) -> torch.Tensor:
    if step_rewards.ndim != 1 or step_rewards.shape[0] != response_mask.shape[0]:
        raise ValueError("step_rewards must have one value per response row")
    scores = step_rewards.clone()
    grouped: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for row, group in enumerate(step_group_ids):
        grouped[group].append(scores[row])
    statistics: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
    for group, values in grouped.items():
        if len(values) == 1:
            statistics[group] = (values[0], scores.new_tensor(1.0))
        else:
            statistics[group] = _sample_mean_std(values)
    for row, group in enumerate(step_group_ids):
        mean, std = statistics[group]
        scores[row] = (
            (scores[row] - mean) / (std + epsilon)
            if normalize_by_std
            else scores[row] - mean
        )
    return scores.unsqueeze(-1).tile((1, response_mask.shape[-1])) * response_mask


def gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_observations: Sequence[Any] | np.ndarray,
    episode_group_ids: Sequence[Any] | np.ndarray,
    trajectory_ids: Sequence[Any] | np.ndarray,
    *,
    epsilon: float = 1e-6,
    step_advantage_weight: float = 1.0,
    mode: str = "mean_norm",
) -> torch.Tensor:
    """Match exact-anchor GiGPO episode and step advantage composition."""
    if mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError(f"Unknown GiGPO mode: {mode}")
    normalize_by_std = mode == "mean_std_norm"
    episode = grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        episode_group_ids,
        trajectory_ids,
        epsilon=epsilon,
        normalize_by_std=normalize_by_std,
        compute_mean_std_cross_steps=True,
    )
    step_groups = exact_step_group_ids(anchor_observations, episode_group_ids)
    step = _step_advantage(
        step_rewards,
        response_mask,
        step_groups,
        epsilon=epsilon,
        normalize_by_std=normalize_by_std,
    )
    return episode + step_advantage_weight * step
