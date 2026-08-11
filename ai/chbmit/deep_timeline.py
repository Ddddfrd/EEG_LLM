"""Labels-only CHB-MIT timelines for raw deep-model evaluation."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ai.v2.lightweight_dataset import write_content_addressed_json

from .index import canonical_hash
from .windows import WindowConfig, _record_windows


DEEP_TIMELINE_SCHEMA_VERSION = "chbmit_deep_target_timeline_v1"
LABELS_FILENAME = "labels.npy"
RECORD_INDICES_FILENAME = "record_indices.npy"
START_SAMPLES_FILENAME = "start_samples.npy"
EVENT_INDICES_FILENAME = "event_indices.npy"
METADATA_FILENAME = "metadata.json"


def _record_sequence(record_id: str) -> int:
    suffix = Path(record_id).stem.rsplit("_", maxsplit=1)[-1].rstrip("+")
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Cannot determine EDF sequence from {record_id}"
        ) from exc


class DeepTargetTimeline:
    """Timeline interface used by raw-model prediction and evaluation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        metadata_path = self.path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete deep target timeline: {metadata_path}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in self.metadata.items()
            if key != "metadata_sha256"
        }
        if self.metadata.get("metadata_sha256") != canonical_hash(body):
            raise ValueError("Deep target timeline metadata hash is invalid")
        if (
            self.metadata.get("schema_version")
            != DEEP_TIMELINE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported deep target timeline schema")
        self.labels = np.load(self.path / LABELS_FILENAME, mmap_mode="r")
        self.record_indices = np.load(
            self.path / RECORD_INDICES_FILENAME, mmap_mode="r"
        )
        self.start_samples = np.load(
            self.path / START_SAMPLES_FILENAME, mmap_mode="r"
        )
        self.event_indices = np.load(
            self.path / EVENT_INDICES_FILENAME, mmap_mode="r"
        )
        expected = (int(self.metadata["window_count"]),)
        for values in (
            self.labels,
            self.record_indices,
            self.start_samples,
            self.event_indices,
        ):
            if values.shape != expected:
                raise ValueError("Deep target timeline vector shape is invalid")

    def record_id(self, row: int) -> str:
        record_index = int(self.record_indices[row])
        return str(self.metadata["records"][record_index]["record_id"])


def build_deep_target_timeline(
    index: Mapping[str, Any],
    *,
    target_subject: str,
    output_dir: Path,
    window_config: WindowConfig,
) -> Path:
    window_config.validate()
    records = sorted(
        (
            record
            for record in index["records"]
            if str(record["subject_id"]) == target_subject
        ),
        key=lambda record: _record_sequence(str(record["record_id"])),
    )
    if not records:
        raise ValueError(f"Unknown target subject: {target_subject}")
    events: list[dict[str, Any]] = [
        {
            "event_index": event_index,
            "event_id": str(event["event_id"]),
            "record_id": str(record["record_id"]),
            "record_sequence": _record_sequence(str(record["record_id"])),
            "start_seconds": float(event["start_seconds"]),
            "end_seconds": float(event["end_seconds"]),
        }
        for event_index, (record, event) in enumerate(
            (
                (record, event)
                for record in records
                for event in record["seizures"]
            ),
            start=1,
        )
    ]
    event_lookup = {
        str(event["event_id"]): int(event["event_index"])
        for event in events
    }
    labels: list[int] = []
    record_indices: list[int] = []
    start_samples: list[int] = []
    event_indices: list[int] = []
    record_metadata: list[dict[str, Any]] = []
    guard_excluded = 0
    for record_index, record in enumerate(records):
        row_start = len(labels)
        record_guard = 0
        for label_name, event, start_sample in _record_windows(
            record, window_config
        ):
            if label_name == "guard_excluded":
                guard_excluded += 1
                record_guard += 1
                continue
            labels.append(1 if label_name == "ictal" else 0)
            record_indices.append(record_index)
            start_samples.append(int(start_sample))
            event_indices.append(
                0 if event is None else event_lookup[str(event["event_id"])]
            )
        row_end = len(labels)
        record_metadata.append({
            "record_index": record_index,
            "record_id": str(record["record_id"]),
            "record_sequence": _record_sequence(str(record["record_id"])),
            "row_start": row_start,
            "row_end": row_end,
            "window_count": row_end - row_start,
            "guard_excluded": record_guard,
            "duration_seconds": float(record["duration_seconds"]),
        })
    contract = {
        "schema_version": DEEP_TIMELINE_SCHEMA_VERSION,
        "dataset_index_sha256": str(index["index_sha256"]),
        "target_subject": target_subject,
        "window_config": {
            "window_seconds": window_config.window_seconds,
            "stride_seconds": window_config.stride_seconds,
            "ictal_overlap_fraction": window_config.ictal_overlap_fraction,
            "seizure_guard_seconds": window_config.seizure_guard_seconds,
            "sampling_frequency_hz": (
                window_config.sampling_frequency_hz
            ),
        },
        "row_order": "numeric EDF sequence then start sample",
        "feature_contract": "labels_only_no_unused_handcrafted_features",
    }
    cache_key = canonical_hash(contract)
    destination = Path(output_dir).resolve() / (
        f"deep_timeline_{target_subject}_{cache_key[:12]}"
    )
    if destination.exists():
        loaded = DeepTargetTimeline(destination)
        if loaded.metadata["cache_key"] != cache_key:
            raise ValueError("Existing deep timeline key does not match")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    try:
        arrays = {
            LABELS_FILENAME: np.asarray(labels, dtype=np.uint8),
            RECORD_INDICES_FILENAME: np.asarray(
                record_indices, dtype=np.int16
            ),
            START_SAMPLES_FILENAME: np.asarray(
                start_samples, dtype=np.int32
            ),
            EVENT_INDICES_FILENAME: np.asarray(
                event_indices, dtype=np.int16
            ),
        }
        for filename, values in arrays.items():
            np.save(temporary / filename, values, allow_pickle=False)
        label_counts = Counter(labels)
        body = {
            **contract,
            "cache_key": cache_key,
            "window_count": len(labels),
            "records": record_metadata,
            "events": events,
            "label_counts": {
                "normal": label_counts[0],
                "ictal": label_counts[1],
            },
            "guard_excluded": guard_excluded,
            "files": {
                filename: int((temporary / filename).stat().st_size)
                for filename in arrays
            },
            "complete": True,
        }
        metadata = {**body, "metadata_sha256": canonical_hash(body)}
        write_content_addressed_json(
            metadata,
            temporary / METADATA_FILENAME,
            hash_field="metadata_sha256",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    DeepTargetTimeline(destination)
    return destination
