from __future__ import annotations

import pytest

from eeg_alarm_policy.splits import (
    DevelopmentGate,
    HeldOutAccessError,
    PartitionError,
    PolicySplit,
    base_model_role,
    scheme_c_model_split,
    validate_strict_subject_partition,
)


def test_scheme_c_historical_split_violates_identity_rule() -> None:
    """Scheme C puts chb01 and chb21 in different partitions; it is retained
    for leaderboard comparability but must fail strict validation."""
    split = scheme_c_model_split()
    with pytest.raises(PartitionError, match="same subject"):
        validate_strict_subject_partition(split)


def test_identity_overlap_subjects_must_share_a_partition() -> None:
    with pytest.raises(PartitionError, match="same subject"):
        validate_strict_subject_partition(
            {"a": ["chb01"], "b": ["chb21"], "rest": ["chb02"]}
        )


def test_compliant_strict_partition_passes() -> None:
    strict = {
        "dev": ["chb01", "chb21", "chb02"],
        "holdout": ["chb20", "chb22", "chb23", "chb24"],
    }
    with pytest.raises(PartitionError, match="missing"):
        validate_strict_subject_partition(strict)
    full = {
        "dev": [f"chb{n:02d}" for n in range(1, 21)] + ["chb21"],
        "holdout": ["chb22", "chb23", "chb24"],
    }
    validate_strict_subject_partition(full)


def test_incomplete_partition_is_rejected() -> None:
    with pytest.raises(PartitionError, match="missing"):
        validate_strict_subject_partition({"all": ["chb01", "chb21"]})


def test_base_model_role_mapping() -> None:
    assert base_model_role("chb01") == "train"
    assert base_model_role("chb20") == "validation"
    assert base_model_role("chb22") == "test"
    with pytest.raises(ValueError, match="unused"):
        base_model_role("chb24")


def test_tier_a_split_excludes_chb21_from_selection() -> None:
    split = PolicySplit.tier_a()
    assert split.policy_selection == ("chb20",)
    assert split.selection_excluded_identity_overlap == ("chb21",)
    assert "chb21" not in split.selection_subjects()
    assert split.role_for("chb01") == "train"
    assert split.role_for("chb21") == "validation"
    assert split.role_for("chb23") == "test"


def test_selection_cannot_contain_identity_overlap_subjects() -> None:
    with pytest.raises(PartitionError, match="identity-overlap"):
        PolicySplit(
            tier="tier_a_development",
            policy_train=("chb02",),
            policy_selection=("chb01",),
            selection_excluded_identity_overlap=("chb21",),
            held_out=("chb22", "chb23"),
        )


def test_development_gate_blocks_held_out() -> None:
    gate = DevelopmentGate()
    with pytest.raises(HeldOutAccessError, match="chb22"):
        gate.require_export_allowed(["chb20", "chb22"])
    with pytest.raises(HeldOutAccessError, match="final-test"):
        gate.require_role_allowed("test")
    gate.require_export_allowed(["chb20", "chb21"])
    gate.require_role_allowed("validation")

    unlocked = DevelopmentGate(unlocked=True)
    unlocked.require_export_allowed(["chb22", "chb23"])
    unlocked.require_role_allowed("test")
