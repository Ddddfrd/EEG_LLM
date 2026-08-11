"""Build a lightweight, deterministic EEG index and base-model v3 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.io

from project_config import DATA_ROOT

from .config import PATIENT_IDS, TARGET_POINTS, TARGET_SR
from .data_utils import CLIP_FILENAME_PATTERN, assess_signal_quality, load_mat_raw
from .evaluation_protocol import (
    BASE_MODEL_PROTOCOL,
    EVALUATION_PROTOCOL_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
)


INDEX_SCHEMA_VERSION = "eeg_lightweight_index_v1"
UNKNOWN = "unknown"
INTERICTAL_PROXY_SECONDS = 300


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_float(mat: Mapping[str, Any], key: str) -> float | str:
    value = mat.get(key)
    if value is None or np.asarray(value).size != 1:
        return UNKNOWN
    candidate = float(np.asarray(value).reshape(-1)[0])
    return candidate if np.isfinite(candidate) else UNKNOWN


def _data_shape(path: Path) -> tuple[int, int] | None:
    variables = {name: shape for name, shape, _ in scipy.io.whosmat(path)}
    shape = variables.get("data")
    if shape is None or len(shape) != 2:
        return None
    channels, timepoints = map(int, shape)
    if channels > timepoints:
        channels, timepoints = timepoints, channels
    return channels, timepoints


def inspect_lightweight_record(
    path: Path, *, data_root: Path, expected_patient_id: int
) -> dict[str, Any]:
    relative_path = path.relative_to(data_root).as_posix()
    row: dict[str, Any] = {
        "record_id": relative_path,
        "relative_path": relative_path,
        "bytes": int(path.stat().st_size),
        "patient_id": expected_patient_id,
        "label_name": UNKNOWN,
        "label": UNKNOWN,
        "segment_number": UNKNOWN,
        "sampling_frequency_hz": UNKNOWN,
        "shape": UNKNOWN,
        "channel_count": UNKNOWN,
        "timepoints": UNKNOWN,
        "latency_seconds": UNKNOWN,
        "episode_id": UNKNOWN,
        "group_id": UNKNOWN,
        "online_group_id": UNKNOWN,
        "eligible_v2": False,
        "exclusion_reasons": [],
    }
    match = CLIP_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        row["exclusion_reasons"].append("invalid_filename")
    else:
        file_patient = int(match.group("patient_id"))
        label_name = match.group("label").lower()
        row.update({
            "patient_id": file_patient,
            "label_name": label_name,
            "label": {"interictal": 0, "ictal": 1}.get(label_name, UNKNOWN),
            "segment_number": int(match.group("segment")),
        })
        if file_patient != expected_patient_id:
            row["exclusion_reasons"].append("filename_patient_mismatch")
        if label_name == "test":
            row["exclusion_reasons"].append("unlabeled_test_clip")

    try:
        shape = _data_shape(path)
        metadata = scipy.io.loadmat(path, variable_names=["freq", "latency"])
        frequency = _scalar_float(metadata, "freq")
        latency = _scalar_float(metadata, "latency")
        if shape is None:
            row["exclusion_reasons"].append("missing_or_invalid_data_shape")
        else:
            row.update({
                "shape": [shape[0], shape[1]],
                "channel_count": shape[0],
                "timepoints": shape[1],
            })
            if shape[1] != TARGET_POINTS:
                row["exclusion_reasons"].append(
                    f"v2_contract:timepoints_not_{TARGET_POINTS}"
                )
        row["sampling_frequency_hz"] = frequency
        row["latency_seconds"] = latency
        if not isinstance(frequency, float) or not np.isclose(
            frequency, TARGET_SR, atol=1.0
        ):
            row["exclusion_reasons"].append(
                f"v2_contract:sampling_frequency_not_{TARGET_SR}hz"
            )
        if row["label_name"] == "ictal" and not isinstance(latency, float):
            row["exclusion_reasons"].append("missing_or_invalid_ictal_latency")
    except Exception as exc:
        row["exclusion_reasons"].append(
            f"mat_metadata_error:{type(exc).__name__}"
        )

    row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"]))
    row["eligible_v2"] = not row["exclusion_reasons"]
    return row


def assign_groups(records: list[dict[str, Any]]) -> None:
    by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["eligible_v2"]:
            by_patient[int(record["patient_id"])].append(record)

    for patient_id, patient_records in by_patient.items():
        ictal = sorted(
            (row for row in patient_records if row["label"] == 1),
            key=lambda row: (int(row["segment_number"]), row["relative_path"]),
        )
        episode = 0
        previous_latency: float | None = None
        for row in ictal:
            latency = float(row["latency_seconds"])
            if previous_latency is None or latency <= previous_latency:
                episode += 1
            previous_latency = latency
            group_id = f"P{patient_id}:ictal_episode:{episode:03d}"
            row["episode_id"] = group_id
            row["group_id"] = group_id
            row["online_group_id"] = group_id

        for row in patient_records:
            if row["label"] != 0:
                continue
            row["group_id"] = f"P{patient_id}:interictal_patient_supergroup"
            block = (int(row["segment_number"]) - 1) // INTERICTAL_PROXY_SECONDS + 1
            row["online_group_id"] = (
                f"P{patient_id}:interictal_proxy_block:{block:04d}"
            )


def _statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    eligible = [row for row in rows if row["eligible_v2"]]
    return {
        "total_records": len(rows),
        "eligible_records": len(eligible),
        "excluded_records": len(rows) - len(eligible),
        "by_patient": dict(sorted(Counter(
            str(row["patient_id"]) for row in rows
        ).items())),
        "eligible_by_patient_and_label": {
            str(patient_id): {
                label_name: sum(
                    row["eligible_v2"]
                    and row["patient_id"] == patient_id
                    and row["label_name"] == label_name
                    for row in rows
                )
                for label_name in ("interictal", "ictal")
            }
            for patient_id in sorted({int(row["patient_id"]) for row in rows})
        },
        "episodes_by_patient": {
            str(patient_id): len({
                row["episode_id"]
                for row in eligible
                if row["patient_id"] == patient_id and row["label"] == 1
            })
            for patient_id in sorted({int(row["patient_id"]) for row in rows})
        },
        "by_exclusion_reason": dict(sorted(Counter(
            reason for row in rows for reason in row["exclusion_reasons"]
        ).items())),
    }


def _sample_quality_audit(
    records: list[dict[str, Any]],
    *,
    data_root: Path,
    per_class: int,
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    by_patient_label: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["eligible_v2"]:
            by_patient_label[(int(row["patient_id"]), int(row["label"]))].append(row)
    for (patient_id, label), rows in sorted(by_patient_label.items()):
        ordered = sorted(rows, key=lambda row: row["relative_path"])
        if per_class >= len(ordered):
            selected = ordered
        else:
            indices = np.linspace(0, len(ordered) - 1, per_class, dtype=int)
            selected = [ordered[int(index)] for index in indices]
        for row in selected:
            path = data_root / row["relative_path"]
            try:
                data, _, metadata = load_mat_raw(path, return_metadata=True)
                quality = assess_signal_quality(data, metadata["model_unit"])
                audit.append({
                    "record_id": row["record_id"],
                    "patient_id": patient_id,
                    "label": label,
                    "metadata_status": metadata["metadata_status"],
                    "source_unit": metadata["source_unit"],
                    "quality": quality,
                })
            except Exception as exc:
                audit.append({
                    "record_id": row["record_id"],
                    "patient_id": patient_id,
                    "label": label,
                    "error": type(exc).__name__,
                })
    return audit


def build_lightweight_index(
    data_root: Path,
    *,
    patient_ids: Iterable[int] = PATIENT_IDS,
    audit_per_class: int = 3,
    progress_every: int = 2000,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EEG data root does not exist: {root}")
    records: list[dict[str, Any]] = []
    paths: list[tuple[Path, int]] = []
    for patient_id in sorted(set(map(int, patient_ids))):
        folder = root / f"Patient_{patient_id}"
        if not folder.is_dir():
            raise FileNotFoundError(f"Patient directory does not exist: {folder}")
        paths.extend((path, patient_id) for path in sorted(folder.glob("*.mat")))
    for index, (path, patient_id) in enumerate(paths, start=1):
        records.append(
            inspect_lightweight_record(
                path, data_root=root, expected_patient_id=patient_id
            )
        )
        if progress_every and index % progress_every == 0:
            print(f"indexed {index}/{len(paths)} MAT files", file=sys.stderr)
    assign_groups(records)
    records.sort(key=lambda row: row["relative_path"])
    body = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "path_semantics": "relative_to_data_root",
        "patient_ids": sorted(set(map(int, patient_ids))),
        "target_contract": {
            "sampling_frequency_hz": TARGET_SR,
            "timepoints": TARGET_POINTS,
        },
        "group_contract": {
            "ictal": "latency_reset_episode",
            "interictal_base": "patient_supergroup",
            "interictal_online": (
                f"simulation_proxy_{INTERICTAL_PROXY_SECONDS}_second_blocks"
            ),
        },
        "records": records,
        "statistics": _statistics(records),
        "sample_quality_audit": _sample_quality_audit(
            records, data_root=root, per_class=audit_per_class
        ),
    }
    return {**body, "index_sha256": _canonical_hash(body)}


def _partition_summary(
    records: list[dict[str, Any]], patient_ids: Iterable[int]
) -> dict[str, Any]:
    selected_patients = set(map(int, patient_ids))
    rows = [
        row for row in records
        if row["eligible_v2"] and int(row["patient_id"]) in selected_patients
    ]
    return {
        "patient_ids": sorted(selected_patients),
        "record_count": len(rows),
        "label_counts": {
            "interictal": sum(row["label"] == 0 for row in rows),
            "ictal": sum(row["label"] == 1 for row in rows),
        },
        "group_ids": sorted({str(row["group_id"]) for row in rows}),
        "record_rule": "all eligible_v2 records for listed patient_ids",
    }


def build_base_split(
    index: Mapping[str, Any], *, target_patient_id: int, seed: int = 42
) -> dict[str, Any]:
    patient_ids = list(map(int, index["patient_ids"]))
    if target_patient_id not in patient_ids:
        raise ValueError(f"Target P{target_patient_id} is absent from dataset index")
    source_patients = [pid for pid in patient_ids if pid != target_patient_id]
    if len(source_patients) < 2:
        raise ValueError("Base split requires at least two source patients")
    calibration_index = (seed + target_patient_id) % len(source_patients)
    calibration_patient = source_patients[calibration_index]
    train_patients = [pid for pid in source_patients if pid != calibration_patient]
    records = list(index["records"])
    partitions = {
        "train": _partition_summary(records, train_patients),
        "calibration": _partition_summary(records, [calibration_patient]),
        "test": _partition_summary(records, [target_patient_id]),
    }
    group_sets = {
        name: set(summary["group_ids"]) for name, summary in partitions.items()
    }
    for left, right in (
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    ):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} group overlap: {sorted(overlap)[:3]}")
    body = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "model_protocol": BASE_MODEL_PROTOCOL,
        "dataset_index_schema": index["schema_version"],
        "dataset_index_sha256": index["index_sha256"],
        "seed": seed,
        "target_patient_id": target_patient_id,
        "selection_contract": {
            "train": "source patients used for parameter fitting",
            "calibration": "one source patient used for model/threshold selection",
            "test": "target patient used once after all choices are frozen",
        },
        "partitions": partitions,
        "group_overlap": {
            "train_calibration": 0,
            "train_test": 0,
            "calibration_test": 0,
        },
    }
    return {**body, "split_sha256": _canonical_hash(body)}


def write_content_addressed_json(
    payload: Mapping[str, Any], output: Path, *, hash_field: str
) -> Path:
    if not str(payload.get(hash_field, "")):
        raise ValueError(f"Artifact is missing hash field {hash_field}")
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")
        return destination
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"Artifact appeared during publication: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_v3"))
    parser.add_argument("--patients", type=int, nargs="+", default=PATIENT_IDS)
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 5, 7, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-per-class", type=int, default=3)
    args = parser.parse_args(argv)
    index = build_lightweight_index(
        args.data_root,
        patient_ids=args.patients,
        audit_per_class=args.audit_per_class,
    )
    output_dir = args.output_dir.resolve()
    index_path = output_dir / f"index_{index['index_sha256'][:12]}.json"
    write_content_addressed_json(index, index_path, hash_field="index_sha256")
    split_paths = []
    for target in args.targets:
        split = build_base_split(index, target_patient_id=target, seed=args.seed)
        split_path = output_dir / (
            f"base_p{target}_{split['split_sha256'][:12]}.json"
        )
        write_content_addressed_json(split, split_path, hash_field="split_sha256")
        split_paths.append(str(split_path))
    print(json.dumps({
        "index": str(index_path),
        "index_sha256": index["index_sha256"],
        "statistics": index["statistics"],
        "splits": split_paths,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
