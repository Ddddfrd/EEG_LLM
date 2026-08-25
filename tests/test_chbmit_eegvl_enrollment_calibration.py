from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ai.chbmit.deep_timeline import (
    DEEP_TIMELINE_SCHEMA_VERSION,
    EVENT_INDICES_FILENAME,
    LABELS_FILENAME,
    METADATA_FILENAME,
    RECORD_INDICES_FILENAME,
    START_SAMPLES_FILENAME,
    DeepTargetTimeline,
)
from ai.chbmit.eegvl_enrollment_calibration import (
    _earliest_contiguous_normal_run,
    apply_patient_score_calibration,
    artifact_path,
    build_strict_enrollment_partition,
    select_patient_calibration_rule,
)
from ai.chbmit.index import canonical_hash


def _timeline(tmp_path: Path, subject: str, labels: list[int]) -> DeepTargetTimeline:
    path = tmp_path / subject
    path.mkdir()
    count = len(labels)
    arrays = {
        LABELS_FILENAME: np.asarray(labels, dtype=np.uint8),
        RECORD_INDICES_FILENAME: np.zeros(count, dtype=np.int16),
        START_SAMPLES_FILENAME: np.arange(count, dtype=np.int32) * 1024,
        EVENT_INDICES_FILENAME: np.asarray(labels, dtype=np.int16),
    }
    for name, values in arrays.items():
        np.save(path / name, values, allow_pickle=False)
    body = {
        "schema_version": DEEP_TIMELINE_SCHEMA_VERSION,
        "target_subject": subject,
        "window_count": count,
        "window_config": {
            "stride_seconds": 4.0,
            "sampling_frequency_hz": 256,
        },
        "records": [{"record_id": f"{subject}_01.edf"}],
    }
    metadata = {**body, "metadata_sha256": canonical_hash(body)}
    (path / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    return DeepTargetTimeline(path)


class FakeNaturalCache:
    def __init__(self, labels: list[int], subject: str = "p1") -> None:
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.metadata = {
            "subject_order": [subject],
            "subjects": {
                subject: {"row_start": 0, "row_end": len(labels)},
            },
        }

    def subject_slice(self, subject: str) -> slice:
        values = self.metadata["subjects"][subject]
        return slice(values["row_start"], values["row_end"])


def test_artifact_path_translates_wsl_path_on_windows() -> None:
    translated = artifact_path("/mnt/c/ML/astar/example.json")
    if os.name == "nt":
        assert str(translated).replace("\\", "/") == "C:/ML/astar/example.json"
    else:
        assert translated == Path("/mnt/c/ML/astar/example.json")


def test_enrollment_selects_earliest_contiguous_normal_run(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path, "p1", [0, 0, 1, 0, 0, 0, 0, 1])

    selected = _earliest_contiguous_normal_run(timeline, window_count=3)

    assert selected.tolist() == [3, 4, 5]


def test_strict_partition_discards_prior_and_enrollment_rows(tmp_path: Path) -> None:
    labels = [0, 1, 0, 0, 0, 0, 1, 0]
    timeline = _timeline(tmp_path, "p1", labels)
    cache = FakeNaturalCache(labels)

    enrollment, scoring, summary = build_strict_enrollment_partition(
        cache,  # type: ignore[arg-type]
        {"p1": timeline},
        enrollment_windows=3,
    )

    assert enrollment["p1"].tolist() == [2, 3, 4]
    assert scoring["p1"].tolist() == [5, 6, 7]
    assert summary["subjects"]["p1"]["discarded_pre_enrollment_windows"] == 2
    assert summary["enrollment_rows_excluded_from_scoring"] is True
    assert not np.intersect1d(enrollment["p1"], scoring["p1"]).size


def test_patient_score_calibration_raises_shifted_patient_threshold() -> None:
    cache = FakeNaturalCache([0, 0, 0, 1, 0, 1])
    probabilities = np.asarray([0.4, 0.5, 0.6, 0.9, 0.2, 0.8], dtype=np.float32)

    adjusted, details = apply_patient_score_calibration(
        cache,  # type: ignore[arg-type]
        probabilities,
        {"p1": np.asarray([0, 1, 2])},
        global_threshold=0.3,
        quantile=1.0,
        margin=0.0,
    )

    assert np.isclose(details["patients"]["p1"]["patient_threshold"], 0.6)
    assert bool(adjusted[2] >= 0.3) == bool(probabilities[2] >= 0.6)
    assert bool(adjusted[4] >= 0.3) == bool(probabilities[4] >= 0.6)


def test_rule_selection_keeps_identity_candidate() -> None:
    cache = FakeNaturalCache([0, 0, 0, 0, 1, 0, 1])
    probabilities = np.asarray(
        [0.1, 0.2, 0.3, 0.25, 0.9, 0.2, 0.8], dtype=np.float32
    )
    enrollment = {"p1": np.asarray([0, 1, 2])}
    scoring = {"p1": np.asarray([3, 4, 5, 6])}

    selected, candidates = select_patient_calibration_rule(
        cache,  # type: ignore[arg-type]
        probabilities,
        enrollment,
        scoring,
        global_threshold=0.5,
        quantiles=(0.99,),
        margins=(0.0,),
        minimum_recall=0.5,
    )

    assert candidates[0]["quantile"] is None
    assert selected["selection_partition"] == "fold4_post_enrollment_only"
