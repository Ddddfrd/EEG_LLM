from __future__ import annotations

import json

import pytest

from eeg_alarm_policy.artifacts import (
    load_prediction_artifact,
    save_prediction_artifact,
)


def test_prediction_artifact_round_trip_is_content_addressed(
    tmp_path, sample_timeline
) -> None:
    first = save_prediction_artifact(
        sample_timeline,
        tmp_path,
        partition_role="audit",
        model_metadata={"checkpoint_sha256": "a" * 64, "pooling": "visual_mean"},
        source_metadata={"manifest_sha256": "b" * 64},
    )
    second = save_prediction_artifact(
        sample_timeline,
        tmp_path,
        partition_role="audit",
        model_metadata={"checkpoint_sha256": "a" * 64, "pooling": "visual_mean"},
        source_metadata={"manifest_sha256": "b" * 64},
    )

    assert first.metadata["artifact_id"] == second.metadata["artifact_id"]
    assert first.metadata_path == second.metadata_path
    assert first.timeline.subject_id == "chb99"
    assert first.timeline.probabilities.tolist() == sample_timeline.probabilities.tolist()
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_prediction_artifact_rejects_metadata_tampering(tmp_path, sample_timeline) -> None:
    artifact = save_prediction_artifact(
        sample_timeline,
        tmp_path,
        partition_role="train",
        model_metadata={"checkpoint_sha256": "a" * 64},
        source_metadata={"manifest_sha256": "b" * 64},
    )
    payload = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    payload["partition_role"] = "test"
    artifact.metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata hash"):
        load_prediction_artifact(artifact.metadata_path)


def test_artifact_rejects_unknown_partition(tmp_path, sample_timeline) -> None:
    with pytest.raises(ValueError, match="partition_role"):
        save_prediction_artifact(
            sample_timeline,
            tmp_path,
            partition_role="future",
            model_metadata={},
            source_metadata={},
        )


def test_subject_id_is_not_truncated(tmp_path, sample_timeline) -> None:
    timeline = sample_timeline.__class__.create(
        subject_id="subject-identifier-longer-than-sixteen",
        probabilities=sample_timeline.probabilities,
        labels=sample_timeline.labels,
        record_indices=sample_timeline.record_indices,
        start_samples=sample_timeline.start_samples,
        event_indices=sample_timeline.event_indices,
        records=sample_timeline.records,
        events=sample_timeline.events,
        sampling_frequency_hz=sample_timeline.sampling_frequency_hz,
        window_seconds=sample_timeline.window_seconds,
        stride_seconds=sample_timeline.stride_seconds,
    )
    artifact = save_prediction_artifact(
        timeline,
        tmp_path,
        partition_role="audit",
        model_metadata={},
        source_metadata={},
    )

    assert artifact.timeline.subject_id == timeline.subject_id
