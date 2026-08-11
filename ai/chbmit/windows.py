"""Create deterministic 4-second CHB-MIT window manifests without signal reads."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ai.v2.lightweight_dataset import write_content_addressed_json

from .contracts import INDEX_SCHEMA_VERSION, WINDOW_SCHEMA_VERSION
from .index import canonical_hash, validate_chbmit_v1_index


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    ictal_overlap_fraction: float = 0.5
    seizure_guard_seconds: float = 60.0
    normal_to_ictal_ratio: float = 3.0
    sampling_frequency_hz: int = 256
    sampling_seed: int = 42

    def validate(self) -> None:
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("Window and stride durations must be positive")
        if not 0 < self.ictal_overlap_fraction < 1:
            raise ValueError("Ictal overlap fraction must be between zero and one")
        if self.seizure_guard_seconds < 0:
            raise ValueError("Seizure guard must be non-negative")
        if self.normal_to_ictal_ratio < 0:
            raise ValueError("Normal-to-ictal ratio must be non-negative")
        if self.sampling_frequency_hz < 1:
            raise ValueError("Sampling frequency must be positive")
        for value, name in (
            (self.window_seconds, "window_seconds"),
            (self.stride_seconds, "stride_seconds"),
        ):
            samples = value * self.sampling_frequency_hz
            if not float(samples).is_integer():
                raise ValueError(f"{name} does not map to an integer sample count")

    @property
    def window_samples(self) -> int:
        return int(self.window_seconds * self.sampling_frequency_hz)

    @property
    def stride_samples(self) -> int:
        return int(self.stride_seconds * self.sampling_frequency_hz)


def load_chbmit_index(path: Path, *, strict_v1: bool = True) -> dict[str, Any]:
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("Unsupported CHB-MIT index schema")
    expected = canonical_hash({
        key: value for key, value in index.items() if key != "index_sha256"
    })
    if index.get("index_sha256") != expected:
        raise ValueError("CHB-MIT index hash is invalid")
    if strict_v1:
        validate_chbmit_v1_index(index)
    return index


def _overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _classify_window(
    record: Mapping[str, Any],
    *,
    start_sample: int,
    config: WindowConfig,
) -> tuple[str, Mapping[str, Any] | None]:
    start_seconds = start_sample / config.sampling_frequency_hz
    end_seconds = start_seconds + config.window_seconds
    best_event: Mapping[str, Any] | None = None
    best_overlap = 0.0
    for event in record["seizures"]:
        overlap = _overlap_seconds(
            start_seconds,
            end_seconds,
            float(event["start_seconds"]),
            float(event["end_seconds"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_event = event
    if best_overlap / config.window_seconds > config.ictal_overlap_fraction:
        return "ictal", best_event
    for event in record["seizures"]:
        guard_start = max(
            0.0, float(event["start_seconds"]) - config.seizure_guard_seconds
        )
        guard_end = min(
            float(record["duration_seconds"]),
            float(event["end_seconds"]) + config.seizure_guard_seconds,
        )
        if _overlap_seconds(start_seconds, end_seconds, guard_start, guard_end) > 0:
            return "guard_excluded", None
    return "normal", None


def _window_row(
    record: Mapping[str, Any],
    *,
    start_sample: int,
    config: WindowConfig,
    label_name: str,
    event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record_id = str(record["record_id"])
    window_id = f"{record_id}@{start_sample:010d}+{config.window_samples:04d}"
    event_id = None if event is None else str(event["event_id"])
    return {
        "window_id": window_id,
        "record_id": record_id,
        "subject_id": str(record["subject_id"]),
        "start_sample": start_sample,
        "sample_count": config.window_samples,
        "start_seconds": start_sample / config.sampling_frequency_hz,
        "end_seconds": (
            start_sample + config.window_samples
        ) / config.sampling_frequency_hz,
        "label": 1 if label_name == "ictal" else 0,
        "label_name": label_name,
        "event_id": event_id,
        "split_group_id": event_id or f"{record_id}:normal",
    }


def _record_windows(
    record: Mapping[str, Any], config: WindowConfig
) -> Iterator[tuple[str, Mapping[str, Any] | None, int]]:
    if int(record["sampling_frequency_hz"]) != config.sampling_frequency_hz:
        raise ValueError(
            f"{record['record_id']} does not match window sampling frequency"
        )
    total_samples = int(record["sample_count"])
    for start_sample in range(
        0,
        total_samples - config.window_samples + 1,
        config.stride_samples,
    ):
        label_name, event = _classify_window(
            record, start_sample=start_sample, config=config
        )
        yield label_name, event, start_sample


def _sampling_rank(window_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{window_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_window_manifest(
    index: Mapping[str, Any],
    *,
    config: WindowConfig | None = None,
) -> dict[str, Any]:
    settings = config or WindowConfig()
    settings.validate()
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("Unsupported CHB-MIT index schema")

    ictal_rows: list[dict[str, Any]] = []
    candidate_normal_counts: Counter[str] = Counter()
    guard_counts: Counter[str] = Counter()
    ictal_counts: Counter[str] = Counter()
    for record in index["records"]:
        subject_id = str(record["subject_id"])
        for label_name, event, start_sample in _record_windows(record, settings):
            if label_name == "ictal":
                row = _window_row(
                    record,
                    start_sample=start_sample,
                    config=settings,
                    label_name=label_name,
                    event=event,
                )
                ictal_rows.append(row)
                ictal_counts[subject_id] += 1
            elif label_name == "normal":
                candidate_normal_counts[subject_id] += 1
            else:
                guard_counts[subject_id] += 1

    targets = {
        subject: min(
            candidate_normal_counts[subject],
            int(math.ceil(ictal_counts[subject] * settings.normal_to_ictal_ratio)),
        )
        for subject in sorted(candidate_normal_counts)
    }
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for record in index["records"]:
        subject_id = str(record["subject_id"])
        target = targets.get(subject_id, 0)
        if target == 0:
            continue
        heap = heaps[subject_id]
        for label_name, event, start_sample in _record_windows(record, settings):
            if label_name != "normal":
                continue
            row = _window_row(
                record,
                start_sample=start_sample,
                config=settings,
                label_name=label_name,
                event=event,
            )
            rank = _sampling_rank(str(row["window_id"]), settings.sampling_seed)
            item = (-rank, str(row["window_id"]), row)
            if len(heap) < target:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    normal_rows = [
        item[2]
        for subject in sorted(heaps)
        for item in heaps[subject]
    ]
    windows = sorted(
        [*ictal_rows, *normal_rows],
        key=lambda row: (
            str(row["record_id"]),
            int(row["start_sample"]),
            int(row["label"]),
        ),
    )
    if len({str(row["window_id"]) for row in windows}) != len(windows):
        raise ValueError("Window manifest contains duplicate window IDs")
    selected_normal_counts = Counter(
        str(row["subject_id"]) for row in normal_rows
    )
    body = {
        "schema_version": WINDOW_SCHEMA_VERSION,
        "dataset_index_schema": str(index["schema_version"]),
        "dataset_index_sha256": str(index["index_sha256"]),
        "config": asdict(settings),
        "window_shape": [
            len(index["target_montage"]),
            settings.window_samples,
        ],
        "selection_contract": {
            "ictal": "all windows with seizure overlap strictly greater than threshold",
            "normal": (
                "per-subject lowest seeded SHA256 ranks after seizure guard exclusion"
            ),
            "grouping": (
                "ictal windows use seizure-event groups; normal windows use EDF-file groups"
            ),
        },
        "windows": windows,
        "statistics": {
            "selected_windows": len(windows),
            "selected_ictal": len(ictal_rows),
            "selected_normal": len(normal_rows),
            "candidate_normal": sum(candidate_normal_counts.values()),
            "guard_excluded": sum(guard_counts.values()),
            "by_subject": {
                subject: {
                    "ictal": ictal_counts[subject],
                    "selected_normal": selected_normal_counts[subject],
                    "candidate_normal": candidate_normal_counts[subject],
                    "guard_excluded": guard_counts[subject],
                }
                for subject in sorted({
                    *ictal_counts,
                    *candidate_normal_counts,
                    *guard_counts,
                })
            },
        },
    }
    return {**body, "window_manifest_sha256": canonical_hash(body)}


def validate_group_integrity(windows: Iterable[Mapping[str, Any]]) -> None:
    event_groups: dict[str, str] = {}
    normal_groups: dict[str, str] = {}
    for row in windows:
        label = int(row["label"])
        group = str(row["split_group_id"])
        if label == 1:
            event_id = str(row["event_id"])
            if not event_id or group != event_id:
                raise ValueError("Ictal window does not inherit its event group")
            event_groups.setdefault(event_id, group)
            if event_groups[event_id] != group:
                raise ValueError(f"Event split across groups: {event_id}")
        else:
            record_id = str(row["record_id"])
            expected = f"{record_id}:normal"
            if group != expected:
                raise ValueError("Normal window does not inherit its EDF-file group")
            normal_groups.setdefault(record_id, group)
            if normal_groups[record_id] != group:
                raise ValueError(f"Normal EDF split across groups: {record_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/chbmit")
    )
    parser.add_argument("--normal-to-ictal-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    index = load_chbmit_index(args.index)
    manifest = build_window_manifest(
        index,
        config=WindowConfig(
            normal_to_ictal_ratio=args.normal_to_ictal_ratio,
            sampling_seed=args.seed,
        ),
    )
    validate_group_integrity(manifest["windows"])
    output = args.output_dir.resolve() / (
        f"{WINDOW_SCHEMA_VERSION}_{manifest['window_manifest_sha256'][:12]}.json"
    )
    write_content_addressed_json(
        manifest, output, hash_field="window_manifest_sha256"
    )
    print(json.dumps({
        "manifest": str(output),
        "window_manifest_sha256": manifest["window_manifest_sha256"],
        "statistics": manifest["statistics"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
