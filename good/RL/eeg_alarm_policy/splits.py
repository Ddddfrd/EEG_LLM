"""Protocol partitions and final-test access control for alarm policies.

Artifact ``partition_role`` values describe base-model provenance. Policy-level
roles (which subjects train the policy, which select it, which stay held out)
are defined here. ``chb01`` and ``chb21`` are two cases from the same subject,
so any strict subject-level partition must keep them together
(EEG_RL_ALARM_POLICY_PLAN.md section 3.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _subject_range(first: int, last: int) -> tuple[str, ...]:
    return tuple(f"chb{number:02d}" for number in range(first, last + 1))


EXPECTED_SUBJECTS = _subject_range(1, 24)
MODEL_TRAIN_SUBJECTS = _subject_range(1, 19)
MODEL_VALIDATION_SUBJECTS = ("chb20", "chb21")
MODEL_TEST_SUBJECTS = ("chb22", "chb23")
UNUSED_SUBJECTS = ("chb24",)

#: These two case IDs belong to one physical subject.
IDENTITY_OVERLAP_GROUP = frozenset({"chb01", "chb21"})

TRAIN_ROLE = "train"
VALIDATION_ROLE = "validation"
TEST_ROLE = "test"
AUDIT_ROLE = "audit"
ARTIFACT_ROLES = frozenset({TRAIN_ROLE, VALIDATION_ROLE, TEST_ROLE, AUDIT_ROLE})


class PartitionError(ValueError):
    """A subject partition violates the protocol contract."""


class HeldOutAccessError(PermissionError):
    """Final-test data was touched while the development gate was locked."""


def scheme_c_model_split() -> dict[str, tuple[str, ...]]:
    """Return the fixed Scheme C base-model split."""
    return {
        "model_train": MODEL_TRAIN_SUBJECTS,
        "model_validation": MODEL_VALIDATION_SUBJECTS,
        "model_test": MODEL_TEST_SUBJECTS,
        "unused": UNUSED_SUBJECTS,
    }


def base_model_role(subject: str) -> str:
    """Return the artifact partition role for a Scheme C subject."""
    subject = str(subject)
    if subject in MODEL_TRAIN_SUBJECTS:
        return TRAIN_ROLE
    if subject in MODEL_VALIDATION_SUBJECTS:
        return VALIDATION_ROLE
    if subject in MODEL_TEST_SUBJECTS:
        return TEST_ROLE
    if subject in UNUSED_SUBJECTS:
        raise ValueError(f"{subject} is unused by the Scheme C leaderboard")
    raise ValueError(f"Unknown subject: {subject}")


def validate_strict_subject_partition(
    partitions: Mapping[str, Sequence[str]],
) -> None:
    """Reject overlaps, incomplete coverage, and identity-group splits."""
    assigned: dict[str, str] = {}
    for name, subjects in partitions.items():
        for subject in subjects:
            subject = str(subject)
            if subject not in EXPECTED_SUBJECTS:
                raise PartitionError(f"Unknown subject {subject!r} in {name!r}")
            if subject in assigned:
                raise PartitionError(
                    f"{subject} appears in both {assigned[subject]!r} and {name!r}"
                )
            assigned[subject] = name
    group_names = sorted({assigned[subject] for subject in IDENTITY_OVERLAP_GROUP})
    if len(group_names) > 1:
        raise PartitionError(
            "chb01 and chb21 are the same subject and must share one partition; "
            f"found partitions {group_names}"
        )
    missing = sorted(set(EXPECTED_SUBJECTS) - set(assigned))
    if missing:
        raise PartitionError(f"Subjects are missing from the partition: {missing}")


@dataclass(frozen=True)
class PolicySplit:
    """Tier A policy protocol: train, select, and held-out subject sets.

    ``chb21`` carries base-model provenance ``validation`` but is excluded from
    policy selection because it shares a subject with ``chb01`` from
    policy-training data.
    """

    tier: str
    policy_train: tuple[str, ...]
    policy_selection: tuple[str, ...]
    selection_excluded_identity_overlap: tuple[str, ...]
    held_out: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        overlapping = (
            set(self.policy_train)
            | set(self.policy_selection)
            | set(self.held_out)
            | set(self.selection_excluded_identity_overlap)
        )
        if set(self.policy_train) & set(self.held_out):
            raise PartitionError("policy train and held-out subjects overlap")
        if set(self.policy_selection) & set(self.held_out):
            raise PartitionError("policy selection and held-out subjects overlap")
        if overlapping - set(EXPECTED_SUBJECTS):
            raise PartitionError("policy split references unknown subjects")
        if set(self.policy_train) & set(self.policy_selection):
            raise PartitionError("policy train and selection subjects overlap")
        if set(self.policy_selection) & IDENTITY_OVERLAP_GROUP:
            raise PartitionError(
                "identity-overlap subjects must not select the policy while the "
                "other group member trains it"
            )

    @classmethod
    def tier_a(cls) -> PolicySplit:
        """Development benchmark: in-sample base-model probabilities for train."""
        split = cls(
            tier="tier_a_development",
            policy_train=MODEL_TRAIN_SUBJECTS,
            policy_selection=("chb20",),
            selection_excluded_identity_overlap=("chb21",),
            held_out=MODEL_TEST_SUBJECTS,
        )
        split.validate()
        return split

    def role_for(self, subject: str) -> str:
        """Return the artifact partition role describing base-model provenance."""
        subject = str(subject)
        if subject in self.policy_train:
            return TRAIN_ROLE
        if subject in self.held_out:
            return TEST_ROLE
        if subject in self.policy_selection or (
            subject in self.selection_excluded_identity_overlap
        ):
            return VALIDATION_ROLE
        raise ValueError(f"Subject {subject} is not assigned in this policy split")

    def selection_subjects(self) -> tuple[str, ...]:
        """Subjects allowed to select policies, reward coefficients, and seeds."""
        return self.policy_selection


@dataclass(frozen=True)
class DevelopmentGate:
    """Refuse held-out access until the run is explicitly finalized.

    The gate is procedural (labels exist on this machine); it exists so that a
    misdirected flag or script cannot silently evaluate the final test during
    development (EEG_RL_ALARM_POLICY_PLAN.md section 11).
    """

    unlocked: bool = False

    def require_export_allowed(self, subjects: Sequence[str]) -> None:
        if self.unlocked:
            return
        requested = {str(subject) for subject in subjects}
        leaked = sorted(requested & set(MODEL_TEST_SUBJECTS))
        if leaked:
            raise HeldOutAccessError(
                "Development mode refuses final-test export for "
                f"{leaked}; pass the explicit finalization flag to unlock"
            )

    def require_role_allowed(self, partition_role: str) -> None:
        if self.unlocked:
            return
        if partition_role == TEST_ROLE:
            raise HeldOutAccessError(
                "Development mode refuses to load final-test artifacts; pass the "
                "explicit finalization flag to unlock"
            )
