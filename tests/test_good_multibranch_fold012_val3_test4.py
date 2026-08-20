from __future__ import annotations

from typing import Any

from ai.chbmit.eeg_continual_pretrain import PAPER_FOLDS
from good.e1_e2_e3_e4_fullband.train_fold012_val3_test4 import (
    TEST_FOLD,
    TRAIN_FOLDS,
    VALIDATION_FOLD,
    build_partition,
)


def _manifest() -> dict[str, Any]:
    windows = []
    for fold, subjects in PAPER_FOLDS.items():
        for subject in subjects:
            windows.extend(
                [
                    {"subject_id": subject, "label": 0, "fold": fold},
                    {"subject_id": subject, "label": 1, "fold": fold},
                ]
            )
    return {"windows": windows}


def test_good_multibranch_split_is_complete_and_patient_disjoint() -> None:
    partition = build_partition(_manifest())

    training = set(partition["training"]["subjects"])
    validation = set(partition["validation_sampled_manifest"]["subjects"])
    testing = set(partition["test_sampled_manifest"]["subjects"])
    expected = {subject for subjects in PAPER_FOLDS.values() for subject in subjects}

    assert partition["training_folds"] == list(TRAIN_FOLDS)
    assert partition["validation_fold"] == VALIDATION_FOLD
    assert partition["test_fold"] == TEST_FOLD
    assert not training & validation
    assert not training & testing
    assert not validation & testing
    assert training | validation | testing == expected
    assert partition["training"]["window_count"] == 30
    assert partition["validation_sampled_manifest"]["window_count"] == 10
    assert partition["test_sampled_manifest"]["window_count"] == 8
