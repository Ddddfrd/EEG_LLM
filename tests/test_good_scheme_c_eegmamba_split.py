from __future__ import annotations

import numpy as np

from ai.chbmit.index import canonical_hash
from good.e1_e2_e3_e4_fullband.train_scheme_c_eegmamba_split import (
    TEST_SUBJECTS,
    TRAINING_SUBJECTS,
    UNUSED_SUBJECTS,
    VALIDATION_SUBJECTS,
    build_eegmamba_partition,
)


def _manifest() -> dict[str, object]:
    windows = [
        {
            "window_id": f"chb{number:02d}/record@0",
            "subject_id": f"chb{number:02d}",
            "label": number % 2,
        }
        for number in range(1, 25)
    ]
    body = {
        "schema_version": "chbmit_windows_v1",
        "windows": windows,
    }
    return {**body, "window_manifest_sha256": canonical_hash(body)}


def test_eegmamba_patient_groups_are_exact_and_disjoint() -> None:
    assert TRAINING_SUBJECTS == tuple(f"chb{number:02d}" for number in range(1, 20))
    assert VALIDATION_SUBJECTS == ("chb20", "chb21")
    assert TEST_SUBJECTS == ("chb22", "chb23")
    assert UNUSED_SUBJECTS == ("chb24",)

    groups = (TRAINING_SUBJECTS, VALIDATION_SUBJECTS, TEST_SUBJECTS, UNUSED_SUBJECTS)
    flattened = [subject for group in groups for subject in group]
    assert len(flattened) == len(set(flattened)) == 24


def test_eegmamba_partition_assigns_every_manifest_row_once() -> None:
    partition = build_eegmamba_partition(_manifest())
    all_indices = np.concatenate([np.asarray(partition[name]["indices"]) for name in partition])

    assert sorted(all_indices.tolist()) == list(range(24))
    assert partition["training"]["window_count"] == 19
    assert partition["validation"]["window_count"] == 2
    assert partition["test"]["window_count"] == 2
    assert partition["unused"]["window_count"] == 1
    assert partition["training"]["ictal_windows"] == 10
    assert partition["training"]["normal_windows"] == 9
