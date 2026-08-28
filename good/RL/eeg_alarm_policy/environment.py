"""Record-level binary alarm environment with private offline rewards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import ProbabilityTimeline
from .features import (
    EnrollmentStatistics,
    build_policy_observation,
    causal_probability_histories,
)


@dataclass(frozen=True)
class AlarmRewardConfig:
    hit_reward: float = 1.0
    miss_penalty: float = 1.0
    false_alarm_penalty: float = 0.02
    latency_penalty_per_minute: float = 0.001
    duplicate_penalty: float = 0.01
    refractory_seconds: float = 300.0

    def validate(self) -> None:
        values = (
            self.hit_reward,
            self.miss_penalty,
            self.false_alarm_penalty,
            self.latency_penalty_per_minute,
            self.duplicate_penalty,
            self.refractory_seconds,
        )
        if any(value < 0 for value in values):
            raise ValueError("Reward and refractory values must be non-negative")


class AlarmRecordEnvironment:
    """One EDF record episode; labels are never included in observations."""

    def __init__(
        self,
        timeline: ProbabilityTimeline,
        *,
        record_index: int,
        enrollment: EnrollmentStatistics,
        reward_config: AlarmRewardConfig,
        histories: np.ndarray | None = None,
    ) -> None:
        timeline.validate()
        enrollment.validate()
        reward_config.validate()
        rows = np.flatnonzero(timeline.record_indices == record_index)
        if not rows.size:
            raise ValueError(f"Timeline has no rows for record {record_index}")
        self.timeline = timeline
        self.record_index = int(record_index)
        self.rows = rows
        self.enrollment = enrollment
        self.reward_config = reward_config
        if histories is None:
            histories = causal_probability_histories(timeline, enrollment)
        histories = np.asarray(histories, dtype=np.float32)
        if histories.shape != (timeline.row_count, 8):
            raise ValueError("histories must match the timeline and observation contract")
        self.histories = histories
        self.position = 0
        self.last_accepted_seconds: float | None = None
        self.detected_events: set[int] = set()
        self.done = False

    def _seconds(self, row: int) -> float:
        return (
            float(self.timeline.start_samples[row])
            / self.timeline.sampling_frequency_hz
        )

    def _observation(self) -> np.ndarray:
        row = int(self.rows[self.position])
        now = self._seconds(row)
        if self.last_accepted_seconds is None:
            since_alarm = 300.0
        else:
            since_alarm = max(0.0, now - self.last_accepted_seconds)
        refractory_remaining = max(
            0.0,
            self.reward_config.refractory_seconds - since_alarm,
        )
        return build_policy_observation(
            self.histories[row],
            self.enrollment,
            seconds_since_alarm=since_alarm,
            refractory_remaining_seconds=refractory_remaining,
            record_start=self.position == 0,
        )

    def reset(self) -> np.ndarray:
        self.position = 0
        self.last_accepted_seconds = None
        self.detected_events.clear()
        self.done = False
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Cannot step a completed alarm episode")
        if action not in (0, 1):
            raise ValueError("Alarm action must be 0 or 1")
        row = int(self.rows[self.position])
        now = self._seconds(row)
        event_index = int(self.timeline.event_indices[row])
        accepted = False
        reward = 0.0
        reward_parts = {
            "hit": 0.0,
            "miss": 0.0,
            "false_alarm": 0.0,
            "latency": 0.0,
            "duplicate": 0.0,
        }
        if action == 1:
            accepted = (
                self.last_accepted_seconds is None
                or now - self.last_accepted_seconds
                >= self.reward_config.refractory_seconds
            )
            if accepted:
                self.last_accepted_seconds = now
                if event_index == 0:
                    reward_parts["false_alarm"] = -self.reward_config.false_alarm_penalty
                elif event_index not in self.detected_events:
                    self.detected_events.add(event_index)
                    reward_parts["hit"] = self.reward_config.hit_reward
                else:
                    reward_parts["duplicate"] = -self.reward_config.duplicate_penalty

        if event_index > 0 and event_index not in self.detected_events:
            reward_parts["latency"] = -(
                self.reward_config.latency_penalty_per_minute
                * self.timeline.stride_seconds
                / 60.0
            )
        next_position = self.position + 1
        next_event = (
            int(self.timeline.event_indices[int(self.rows[next_position])])
            if next_position < len(self.rows)
            else 0
        )
        if (
            event_index > 0
            and next_event != event_index
            and event_index not in self.detected_events
        ):
            reward_parts["miss"] = -self.reward_config.miss_penalty
        reward = float(sum(reward_parts.values()))
        self.position = next_position
        self.done = self.position >= len(self.rows)
        observation = None if self.done else self._observation()
        return observation, reward, self.done, {
            "row": row,
            "emitted_alarm": bool(action),
            "accepted_alarm": accepted,
            "event_index": event_index,
            "reward_parts": reward_parts,
        }


def rollout_policy(
    timeline: ProbabilityTimeline,
    enrollment: EnrollmentStatistics,
    policy: Callable[[np.ndarray], int],
    *,
    reward_config: AlarmRewardConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run a deterministic or stochastic policy over every record."""
    actions = np.zeros(timeline.row_count, dtype=np.uint8)
    reward_totals = {
        "return": 0.0,
        "hit": 0.0,
        "miss": 0.0,
        "false_alarm": 0.0,
        "latency": 0.0,
        "duplicate": 0.0,
    }
    for record_index in np.unique(timeline.record_indices):
        environment = AlarmRecordEnvironment(
            timeline,
            record_index=int(record_index),
            enrollment=enrollment,
            reward_config=reward_config,
        )
        observation = environment.reset()
        while True:
            action = int(policy(observation))
            next_observation, reward, done, info = environment.step(action)
            actions[int(info["row"])] = action
            reward_totals["return"] += reward
            for name, value in info["reward_parts"].items():
                reward_totals[name] += float(value)
            if done:
                break
            if next_observation is None:
                raise RuntimeError("Non-terminal environment step lacks observation")
            observation = next_observation
    return actions, reward_totals
