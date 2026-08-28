"""Content-addressed storage for frozen EEG probability timelines."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import EventInterval, ProbabilityTimeline

PREDICTION_ARTIFACT_SCHEMA_VERSION = "eeg_alarm_probability_timeline_v1"
_ARRAY_NAMES = (
    "subject_id",
    "probability",
    "label",
    "record_index",
    "start_sample",
    "event_index",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        encoded = _canonical_json(dict(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain JSON-serializable values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a JSON object")
    return decoded


def _timeline_arrays(timeline: ProbabilityTimeline) -> dict[str, np.ndarray]:
    subject_dtype = f"<U{max(1, len(timeline.subject_id))}"
    return {
        "subject_id": np.full(timeline.row_count, timeline.subject_id, dtype=subject_dtype),
        "probability": np.asarray(timeline.probabilities, dtype=np.float32),
        "label": np.asarray(timeline.labels, dtype=np.uint8),
        "record_index": np.asarray(timeline.record_indices, dtype=np.int32),
        "start_sample": np.asarray(timeline.start_samples, dtype=np.int64),
        "event_index": np.asarray(timeline.event_indices, dtype=np.int32),
    }


def _identity_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "subject_id",
            "partition_role",
            "row_count",
            "records",
            "events",
            "sampling_frequency_hz",
            "window_seconds",
            "stride_seconds",
            "arrays",
            "model",
            "source",
        )
    }


@dataclass(frozen=True)
class PredictionArtifact:
    """A verified timeline together with immutable artifact metadata."""

    timeline: ProbabilityTimeline
    metadata: dict[str, Any]
    data_path: Path
    metadata_path: Path


def save_prediction_artifact(
    timeline: ProbabilityTimeline,
    output_dir: Path,
    *,
    partition_role: str,
    model_metadata: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> PredictionArtifact:
    """Persist one deterministic, content-addressed subject timeline."""
    timeline.validate()
    if partition_role not in {"train", "validation", "test", "audit"}:
        raise ValueError("partition_role is invalid")
    model = _json_object(model_metadata, name="model_metadata")
    source = _json_object(source_metadata, name="source_metadata")
    arrays = _timeline_arrays(timeline)
    array_metadata = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_hash(value),
        }
        for name, value in arrays.items()
    }
    metadata: dict[str, Any] = {
        "schema_version": PREDICTION_ARTIFACT_SCHEMA_VERSION,
        "subject_id": timeline.subject_id,
        "partition_role": partition_role,
        "row_count": timeline.row_count,
        "records": list(timeline.records),
        "events": [event.to_dict() for event in timeline.events],
        "sampling_frequency_hz": timeline.sampling_frequency_hz,
        "window_seconds": timeline.window_seconds,
        "stride_seconds": timeline.stride_seconds,
        "arrays": array_metadata,
        "model": model,
        "source": source,
    }
    artifact_id = _canonical_hash(_identity_payload(metadata))
    stem = f"predictions_{timeline.subject_id}_{artifact_id[:12]}"
    data_name = f"{stem}.npz"
    metadata_name = f"{stem}.json"
    metadata["artifact_id"] = artifact_id
    metadata["data_file"] = data_name
    metadata["metadata_sha256"] = _canonical_hash(metadata)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / data_name
    metadata_path = destination / metadata_name
    if data_path.exists() or metadata_path.exists():
        if not data_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f"Incomplete prediction artifact: {stem}")
        loaded = load_prediction_artifact(metadata_path)
        if loaded.metadata["artifact_id"] != artifact_id:
            raise ValueError("Existing prediction artifact identity mismatch")
        return loaded

    token = uuid.uuid4().hex
    temporary_data = destination / f".{data_name}.{token}.tmp"
    temporary_metadata = destination / f".{metadata_name}.{token}.tmp"
    try:
        with temporary_data.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_data, data_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_data.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return load_prediction_artifact(metadata_path)


def save_content_addressed_json(
    payload: Mapping[str, Any],
    destination_dir: Path,
    *,
    hash_field: str,
    stem: str,
) -> Path:
    """Write an immutable, self-hashing JSON result artifact atomically.

    The file name embeds the first twelve hex characters of the content hash,
    so identical payloads collapse onto one file.
    """
    if hash_field in payload:
        raise ValueError(f"payload must not predefine {hash_field!r}")
    digest = _canonical_hash(payload)
    body = {**payload, hash_field: digest}
    directory = Path(destination_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{digest[:12]}.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_prediction_artifact(metadata_path: Path) -> PredictionArtifact:
    """Load an artifact and reject any metadata or array identity mismatch."""
    metadata_file = Path(metadata_path).resolve()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != PREDICTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported prediction artifact schema")
    metadata_sha256 = metadata.get("metadata_sha256")
    body = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    if metadata_sha256 != _canonical_hash(body):
        raise ValueError("Prediction artifact metadata hash is invalid")
    if metadata.get("artifact_id") != _canonical_hash(_identity_payload(metadata)):
        raise ValueError("Prediction artifact identity is invalid")

    data_path = (metadata_file.parent / str(metadata["data_file"])).resolve()
    if data_path.parent != metadata_file.parent:
        raise ValueError("Prediction artifact data_file escapes its directory")
    with np.load(data_path, allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAY_NAMES):
            raise ValueError("Prediction artifact arrays are incomplete or unexpected")
        arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}

    for name, array in arrays.items():
        expected = metadata["arrays"][name]
        if array.dtype.str != expected["dtype"] or list(array.shape) != expected["shape"]:
            raise ValueError(f"Prediction artifact {name} contract changed")
        if _array_hash(array) != expected["sha256"]:
            raise ValueError(f"Prediction artifact {name} hash is invalid")

    subject_values = np.unique(arrays["subject_id"])
    if subject_values.tolist() != [metadata["subject_id"]]:
        raise ValueError("Prediction artifact subject_id array is invalid")
    timeline = ProbabilityTimeline.create(
        subject_id=str(metadata["subject_id"]),
        probabilities=arrays["probability"],
        labels=arrays["label"],
        record_indices=arrays["record_index"],
        start_samples=arrays["start_sample"],
        event_indices=arrays["event_index"],
        records=metadata["records"],
        events=[EventInterval.from_dict(value) for value in metadata["events"]],
        sampling_frequency_hz=float(metadata["sampling_frequency_hz"]),
        window_seconds=float(metadata["window_seconds"]),
        stride_seconds=float(metadata["stride_seconds"]),
    )
    if timeline.row_count != int(metadata["row_count"]):
        raise ValueError("Prediction artifact row_count is invalid")
    return PredictionArtifact(
        timeline=timeline,
        metadata=metadata,
        data_path=data_path,
        metadata_path=metadata_file,
    )
