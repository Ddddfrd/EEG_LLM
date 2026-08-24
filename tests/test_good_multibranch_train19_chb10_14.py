from __future__ import annotations

from typing import Any

from good.e1_e2_e3_e4_fullband.train_19_vs_chb10_14 import (
    EXPECTED_SUBJECTS,
    TRAINING_SUBJECTS,
    VALIDATION_TEST_SUBJECTS,
    build_partition,
)


def _manifest() -> dict[str, Any]:
    return {
        "windows": [
            {"subject_id": subject, "label": label}
            for subject in EXPECTED_SUBJECTS
            for label in (0, 1)
        ]
    }


def test_requested_split_has_19_train_and_five_shared_validation_test_patients() -> None:
    partition = build_partition(_manifest())

    training = set(partition["training"]["subjects"])
    validation_test = set(partition["validation_test_sampled_manifest"]["subjects"])

    assert len(TRAINING_SUBJECTS) == 19
    assert len(VALIDATION_TEST_SUBJECTS) == 5
    assert validation_test == {"chb10", "chb11", "chb12", "chb13", "chb14"}
    assert not training & validation_test
    assert training | validation_test == set(EXPECTED_SUBJECTS)
    assert partition["training"]["window_count"] == 38
    assert partition["validation_test_sampled_manifest"]["window_count"] == 10
    assert partition["validation_equals_test"] is True
