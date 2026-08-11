"""Build full clean target-patient feature timelines for event-level evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyedflib

from ai.v2.lightweight_dataset import write_content_addressed_json

from .features import FEATURE_NAMES, extract_channel_features
from .index import canonical_hash
from .montage import MontageRecipe, MontageTerm, read_montage_window
from .windows import WindowConfig, _record_windows, load_chbmit_index


TIMELINE_CACHE_SCHEMA_VERSION = "chbmit_target_timeline_features_v1"
FEATURES_FILENAME = "features.npy"
LABELS_FILENAME = "labels.npy"
RECORD_INDICES_FILENAME = "record_indices.npy"
START_SAMPLES_FILENAME = "start_samples.npy"
EVENT_INDICES_FILENAME = "event_indices.npy"
METADATA_FILENAME = "metadata.json"


def _recipe_from_payload(payload: Mapping[str, Any]) -> MontageRecipe:
    return MontageRecipe(
        target_label=str(payload["target_label"]),
        mode=str(payload["mode"]),
        terms=tuple(
            MontageTerm(
                source_index=int(term["source_index"]),
                source_label=str(term["source_label"]),
                coefficient=float(term["coefficient"]),
            )
            for term in payload["terms"]
        ),
    )


def _record_sequence(record_id: str) -> int:
    suffix = Path(record_id).stem.rsplit("_", maxsplit=1)[-1].rstrip("+")
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Cannot determine EDF sequence from {record_id}") from exc


class TargetTimelineCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        metadata_path = self.path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete target timeline: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in self.metadata.items()
            if key != "metadata_sha256"
        }
        if self.metadata.get("metadata_sha256") != canonical_hash(body):
            raise ValueError("Target timeline metadata hash is invalid")
        if self.metadata.get("schema_version") != TIMELINE_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported target timeline schema")
        self.features = np.load(self.path / FEATURES_FILENAME, mmap_mode="r")
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
        rows = int(self.metadata["window_count"])
        expected_feature_shape = (
            rows,
            int(self.metadata["channel_count"]),
            len(self.metadata["feature_names"]),
        )
        if self.features.shape != expected_feature_shape:
            raise ValueError("Target timeline feature shape is invalid")
        for array in (
            self.labels,
            self.record_indices,
            self.start_samples,
            self.event_indices,
        ):
            if array.shape != (rows,):
                raise ValueError("Target timeline vector shape is invalid")

    def record_id(self, row: int) -> str:
        record_index = int(self.record_indices[row])
        return str(self.metadata["records"][record_index]["record_id"])


def _record_plan(
    record: Mapping[str, Any],
    config: WindowConfig,
    event_lookup: Mapping[str, int],
) -> dict[str, Any]:
    starts: list[int] = []
    labels: list[int] = []
    event_indices: list[int] = []
    guard_excluded = 0
    for label_name, event, start_sample in _record_windows(record, config):
        if label_name == "guard_excluded":
            guard_excluded += 1
            continue
        starts.append(start_sample)
        labels.append(1 if label_name == "ictal" else 0)
        event_indices.append(
            0 if event is None else event_lookup[str(event["event_id"])]
        )
    return {
        "record_id": str(record["record_id"]),
        "record_sequence": _record_sequence(str(record["record_id"])),
        "starts": starts,
        "labels": labels,
        "event_indices": event_indices,
        "guard_excluded": guard_excluded,
    }


def _feature_batches(
    waveform: np.ndarray,
    starts: list[int],
    *,
    sample_count: int,
    sampling_frequency_hz: int,
    batch_size: int,
) -> Any:
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = np.stack(
            [waveform[:, start : start + sample_count] for start in batch_starts],
            axis=0,
        )
        flattened = windows.reshape(-1, sample_count)
        features = extract_channel_features(
            flattened,
            sampling_frequency_hz=sampling_frequency_hz,
        )
        yield offset, features.reshape(
            len(batch_starts), waveform.shape[0], len(FEATURE_NAMES)
        )


def build_target_timeline_cache(
    index: Mapping[str, Any],
    *,
    target_subject: str,
    data_root: Path,
    output_dir: Path,
    window_config: WindowConfig | None = None,
    batch_size: int = 256,
    progress: bool = True,
) -> Path:
    config = window_config or WindowConfig()
    config.validate()
    if batch_size < 1:
        raise ValueError("Timeline feature batch size must be positive")
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
        str(event["event_id"]): int(event["event_index"]) for event in events
    }
    plans = [
        _record_plan(record, config, event_lookup) for record in records
    ]
    window_count = sum(len(plan["starts"]) for plan in plans)
    contract = {
        "schema_version": TIMELINE_CACHE_SCHEMA_VERSION,
        "dataset_index_sha256": str(index["index_sha256"]),
        "target_subject": target_subject,
        "window_config": {
            "window_seconds": config.window_seconds,
            "stride_seconds": config.stride_seconds,
            "ictal_overlap_fraction": config.ictal_overlap_fraction,
            "seizure_guard_seconds": config.seizure_guard_seconds,
            "sampling_frequency_hz": config.sampling_frequency_hz,
        },
        "window_count": window_count,
        "channel_count": len(index["target_montage"]),
        "feature_names": list(FEATURE_NAMES),
        "feature_dtype": "float32",
        "label_dtype": "uint8",
        "row_order": "numeric EDF sequence then start sample",
    }
    cache_key = canonical_hash(contract)
    destination = (
        Path(output_dir).resolve()
        / f"timeline_{target_subject}_{cache_key[:12]}"
    )
    if destination.exists():
        loaded = TargetTimelineCache(destination)
        if loaded.metadata["cache_key"] != cache_key:
            raise ValueError("Existing target timeline key does not match")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    try:
        feature_shape = (
            window_count,
            len(index["target_montage"]),
            len(FEATURE_NAMES),
        )
        feature_memmap = np.lib.format.open_memmap(
            temporary / FEATURES_FILENAME,
            mode="w+",
            dtype=np.float32,
            shape=feature_shape,
        )
        label_memmap = np.lib.format.open_memmap(
            temporary / LABELS_FILENAME,
            mode="w+",
            dtype=np.uint8,
            shape=(window_count,),
        )
        record_memmap = np.lib.format.open_memmap(
            temporary / RECORD_INDICES_FILENAME,
            mode="w+",
            dtype=np.int16,
            shape=(window_count,),
        )
        start_memmap = np.lib.format.open_memmap(
            temporary / START_SAMPLES_FILENAME,
            mode="w+",
            dtype=np.int32,
            shape=(window_count,),
        )
        event_memmap = np.lib.format.open_memmap(
            temporary / EVENT_INDICES_FILENAME,
            mode="w+",
            dtype=np.int16,
            shape=(window_count,),
        )
        root = Path(data_root).resolve()
        row_offset = 0
        record_metadata: list[dict[str, Any]] = []
        for record_index, (record, plan) in enumerate(zip(records, plans, strict=True)):
            path = root / Path(str(record["record_id"]))
            if not path.is_file():
                raise FileNotFoundError(f"EDF is missing: {path}")
            recipes = tuple(
                _recipe_from_payload(payload) for payload in record["montage"]
            )
            with pyedflib.EdfReader(str(path)) as reader:
                waveform = read_montage_window(
                    reader,
                    recipes,
                    start_sample=0,
                    sample_count=int(record["sample_count"]),
                )
            starts = list(map(int, plan["starts"]))
            row_end = row_offset + len(starts)
            for batch_offset, batch_features in _feature_batches(
                waveform,
                starts,
                sample_count=config.window_samples,
                sampling_frequency_hz=config.sampling_frequency_hz,
                batch_size=batch_size,
            ):
                begin = row_offset + batch_offset
                end = begin + len(batch_features)
                feature_memmap[begin:end] = batch_features
            del waveform
            label_memmap[row_offset:row_end] = np.asarray(
                plan["labels"], dtype=np.uint8
            )
            record_memmap[row_offset:row_end] = record_index
            start_memmap[row_offset:row_end] = np.asarray(
                starts, dtype=np.int32
            )
            event_memmap[row_offset:row_end] = np.asarray(
                plan["event_indices"], dtype=np.int16
            )
            record_metadata.append({
                "record_index": record_index,
                "record_id": str(record["record_id"]),
                "record_sequence": int(plan["record_sequence"]),
                "row_start": row_offset,
                "row_end": row_end,
                "window_count": len(starts),
                "guard_excluded": int(plan["guard_excluded"]),
                "duration_seconds": float(record["duration_seconds"]),
            })
            row_offset = row_end
            if progress:
                print(
                    f"timeline {target_subject}: "
                    f"{record_index + 1}/{len(records)} EDF files",
                    flush=True,
                )
        if row_offset != window_count:
            raise AssertionError("Target timeline row count changed during build")
        memmaps = (
            feature_memmap,
            label_memmap,
            record_memmap,
            start_memmap,
            event_memmap,
        )
        for memmap in memmaps:
            memmap.flush()
        label_counts = Counter(map(int, label_memmap.tolist()))
        for memmap in memmaps:
            mapped_file = getattr(memmap, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
        del memmaps
        del (
            feature_memmap,
            label_memmap,
            record_memmap,
            start_memmap,
            event_memmap,
        )
        files = {
            name: int((temporary / name).stat().st_size)
            for name in (
                FEATURES_FILENAME,
                LABELS_FILENAME,
                RECORD_INDICES_FILENAME,
                START_SAMPLES_FILENAME,
                EVENT_INDICES_FILENAME,
            )
        }
        body = {
            **contract,
            "cache_key": cache_key,
            "records": record_metadata,
            "events": events,
            "label_counts": {
                "normal": label_counts[0],
                "ictal": label_counts[1],
            },
            "guard_excluded": sum(
                int(plan["guard_excluded"]) for plan in plans
            ),
            "files": files,
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
    TargetTimelineCache(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/timelines"),
    )
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    index = load_chbmit_index(args.index)
    outputs = []
    for target in args.targets:
        output = build_target_timeline_cache(
            index,
            target_subject=target,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            progress=not args.quiet,
        )
        cache = TargetTimelineCache(output)
        outputs.append({
            "target": target,
            "cache": str(output),
            "window_count": cache.metadata["window_count"],
            "label_counts": cache.metadata["label_counts"],
        })
    print(json.dumps({"timelines": outputs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
