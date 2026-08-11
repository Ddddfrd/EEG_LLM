"""Build an immutable, deterministic manifest for every EEG MAT file."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.io

from project_config import DATA_ROOT
from server.utils.hashing import sha256_file

from .config import TARGET_POINTS, TARGET_SR
from .data_utils import (
    CLIP_FILENAME_PATTERN,
    assess_signal_quality,
    load_mat_raw,
)
from .evaluation_protocol import DATA_MANIFEST_SCHEMA_VERSION


UNKNOWN = "unknown"


def _mat_scalar_text(mat: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in mat:
            continue
        value: Any = mat[key]
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.flat[0]
        text = str(value).strip()
        if text:
            return text
    return UNKNOWN


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _filename_fields(
    path: Path, folder_patient_id: int | str
) -> dict[str, Any]:
    match = CLIP_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        return {
            "patient_id": folder_patient_id,
            "label_name": UNKNOWN,
            "label": UNKNOWN,
            "segment_number": UNKNOWN,
            "filename_valid": False,
            "filename_owner_matches_folder": False,
        }
    patient_id = int(match.group("patient_id"))
    label_name = match.group("label").lower()
    return {
        "patient_id": patient_id,
        "label_name": label_name,
        "label": {"interictal": 0, "ictal": 1}.get(label_name, UNKNOWN),
        "segment_number": int(match.group("segment")),
        "filename_valid": True,
        "filename_owner_matches_folder": patient_id == folder_patient_id,
    }


def _strict_channel_names(
    mat: Mapping[str, Any], channel_count: int
) -> list[str] | str:
    raw = mat.get("channels")
    if not isinstance(raw, np.ndarray):
        return UNKNOWN
    if raw.dtype.names:
        record = raw.flat[0]
        values = [record[field_name] for field_name in raw.dtype.names]
    else:
        values = list(raw.flat)
    names = []
    for value in values:
        current = value
        while isinstance(current, np.ndarray) and current.size == 1:
            current = current.flat[0]
        text = str(current).strip()
        if not text:
            return UNKNOWN
        names.append(text)
    if len(names) != channel_count or len(set(names)) != channel_count:
        return UNKNOWN
    return names


def _assign_groups(rows: list[dict[str, Any]]) -> None:
    by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["filename_valid"] and row["filename_owner_matches_folder"]:
            by_patient[int(row["patient_id"])].append(row)

    for patient_id, patient_rows in by_patient.items():
        ictal_episode = 0
        previous_latency: float | None = None
        ordered = sorted(
            patient_rows,
            key=lambda row: (
                str(row["label_name"]),
                int(row["segment_number"]),
                str(row["relative_path"]),
            ),
        )
        for row in ordered:
            if row["label_name"] == "ictal":
                latency = row.get("latency_seconds")
                if not isinstance(latency, (int, float)) or not np.isfinite(latency):
                    row["episode_id"] = UNKNOWN
                    row["group_id"] = UNKNOWN
                    row["group_source"] = "missing_or_invalid_latency"
                    row["exclusion_reasons"].append("missing_or_invalid_ictal_latency")
                    continue
                if previous_latency is None or latency <= previous_latency:
                    ictal_episode += 1
                previous_latency = float(latency)
                row["episode_id"] = (
                    f"P{patient_id}:ictal_episode:{ictal_episode:03d}"
                )
                row["group_id"] = row["episode_id"]
                row["group_source"] = "mat_latency_reset"
            elif row["label_name"] == "interictal":
                session_id = row["session_id"]
                if session_id == UNKNOWN:
                    row["group_id"] = (
                        f"P{patient_id}:interictal_patient_supergroup"
                    )
                    row["group_source"] = (
                        "patient_supergroup_recording_id_unavailable"
                    )
                else:
                    session_hash = hashlib.sha256(
                        str(session_id).encode("utf-8")
                    ).hexdigest()[:12]
                    row["group_id"] = (
                        f"P{patient_id}:interictal_session:{session_hash}"
                    )
                    row["group_source"] = "mat_session_id"
                row["episode_id"] = UNKNOWN
            else:
                row["group_id"] = UNKNOWN
                row["episode_id"] = UNKNOWN
                row["group_source"] = "unlabeled_or_unknown"


def inspect_mat_file(
    path: Path,
    *,
    data_root: Path,
    folder_patient_id: int | str,
) -> dict[str, Any]:
    fields = _filename_fields(path, folder_patient_id)
    reasons: list[str] = []
    if not fields["filename_valid"]:
        reasons.append("invalid_filename")
    elif not fields["filename_owner_matches_folder"]:
        reasons.append("filename_patient_mismatch")
    if folder_patient_id == UNKNOWN:
        reasons.append("invalid_patient_folder")
    if fields["label_name"] == "test":
        reasons.append("unlabeled_test_clip")

    row: dict[str, Any] = {
        "relative_path": path.relative_to(data_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **fields,
        "episode_id": UNKNOWN,
        "session_id": UNKNOWN,
        "group_id": UNKNOWN,
        "group_source": UNKNOWN,
        "sampling_frequency_hz": UNKNOWN,
        "shape": UNKNOWN,
        "channel_count": UNKNOWN,
        "channel_names": UNKNOWN,
        "source_unit": UNKNOWN,
        "model_unit": UNKNOWN,
        "gain": UNKNOWN,
        "reference": UNKNOWN,
        "latency_seconds": UNKNOWN,
        "quality_status": "unavailable",
        "quality_flags": [],
        "exclusion_reasons": reasons,
    }
    try:
        mat = scipy.io.loadmat(path)
        data, frequency, metadata = load_mat_raw(path, return_metadata=True)
        quality = assess_signal_quality(data, metadata["model_unit"])
        latency = mat.get("latency")
        latency_value: float | str = UNKNOWN
        if latency is not None and np.asarray(latency).size:
            candidate = float(np.asarray(latency).flat[0])
            latency_value = candidate if np.isfinite(candidate) else UNKNOWN
        row.update({
            "sampling_frequency_hz": float(frequency),
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "channel_count": int(data.shape[0]),
            "channel_names": _strict_channel_names(mat, int(data.shape[0])),
            "source_unit": (
                UNKNOWN
                if "unit" in metadata.get("missing_metadata", [])
                else metadata.get("source_unit", UNKNOWN)
            ),
            "model_unit": metadata.get("model_unit", UNKNOWN),
            "gain": (
                UNKNOWN
                if "gain" in metadata.get("missing_metadata", [])
                else metadata.get("gain", UNKNOWN)
            ),
            "reference": (
                UNKNOWN
                if "reference" in metadata.get("missing_metadata", [])
                else metadata.get("reference", UNKNOWN)
            ),
            "session_id": _mat_scalar_text(
                mat, "session_id", "recording_id", "recording"
            ),
            "latency_seconds": latency_value,
            "quality_status": quality["status"],
            "quality_flags": sorted(map(str, quality["flags"])),
        })
        if not quality["passed"]:
            row["exclusion_reasons"].extend(
                f"quality:{flag}" for flag in sorted(quality["flags"])
            )
        if not np.isclose(frequency, TARGET_SR, atol=1.0):
            row["exclusion_reasons"].append(
                f"v2_contract:sampling_frequency_not_{TARGET_SR}hz"
            )
        if data.shape[1] != TARGET_POINTS:
            row["exclusion_reasons"].append(
                f"v2_contract:timepoints_not_{TARGET_POINTS}"
            )
    except Exception as exc:
        row["exclusion_reasons"].append(
            f"mat_contract_error:{type(exc).__name__}"
        )
    row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"]))
    return row


def manifest_statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_patient = Counter(str(row["patient_id"]) for row in rows)
    by_label = Counter(str(row["label_name"]) for row in rows)
    by_episode = Counter(
        str(row["episode_id"])
        for row in rows
        if row["episode_id"] != UNKNOWN
    )
    quality_flags = Counter(
        str(flag) for row in rows for flag in row["quality_flags"]
    )
    quality_status = Counter(str(row["quality_status"]) for row in rows)
    exclusions = Counter(
        str(reason) for row in rows for reason in row["exclusion_reasons"]
    )
    return {
        "total_files": len(rows),
        "included_files": sum(not row["exclusion_reasons"] for row in rows),
        "excluded_files": sum(bool(row["exclusion_reasons"]) for row in rows),
        "by_patient": dict(sorted(by_patient.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_episode": dict(sorted(by_episode.items())),
        "by_quality_status": dict(sorted(quality_status.items())),
        "by_quality_flag": dict(sorted(quality_flags.items())),
        "by_exclusion_reason": dict(sorted(exclusions.items())),
    }


def build_dataset_manifest(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    rows: list[dict[str, Any]] = []
    if not data_root.is_dir():
        raise FileNotFoundError(f"EEG data root does not exist: {data_root}")
    for folder in sorted(data_root.glob("Patient_*")):
        if not folder.is_dir():
            continue
        folder_patient_id: int | str
        try:
            folder_patient_id = int(folder.name.removeprefix("Patient_"))
        except ValueError:
            folder_patient_id = UNKNOWN
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".mat":
                rows.append(
                    inspect_mat_file(
                        path,
                        data_root=data_root,
                        folder_patient_id=folder_patient_id,
                    )
                )
    _assign_groups(rows)
    rows.sort(key=lambda row: str(row["relative_path"]))
    body = {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        "dataset_scope": "all_mat_files_under_Patient_*_directories",
        "path_semantics": "relative_to_data_root",
        "missing_value": UNKNOWN,
        "records": rows,
        "statistics": manifest_statistics(rows),
    }
    return {**body, "manifest_sha256": _canonical_hash(body)}


def validate_split_group_disjoint(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {
        "train": set(),
        "calibration": set(),
        "test": set(),
    }
    for row in records:
        split = str(row.get("split", ""))
        if split == "validation":
            split = "calibration"
        if split not in groups:
            raise ValueError(f"Record has unsupported split: {split!r}")
        group_id = str(row.get("group_id", UNKNOWN))
        if group_id == UNKNOWN:
            raise ValueError("Split records require a known group_id")
        groups[split].add(group_id)
    for left, right in (
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    ):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(
                f"{left}/{right} group overlap: {sorted(overlap)[:3]}"
            )
    return {name: sorted(values) for name, values in groups.items()}


def write_manifest(manifest: Mapping[str, Any], output: Path) -> None:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite data manifest: {output}")
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_dataset_manifest(args.data_root)
    write_manifest(manifest, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "statistics": manifest["statistics"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
