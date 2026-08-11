"""Build the immutable CHB-MIT EDF metadata and annotation index."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyedflib

from ai.v2.lightweight_dataset import write_content_addressed_json

from .annotations import SummaryRecord, parse_summary
from .contracts import (
    CANONICAL_BIPOLAR_CHANNELS,
    EXPECTED_EDF_FILES,
    EXPECTED_SAMPLE_RATE_HZ,
    EXPECTED_SEIZURES,
    EXPECTED_SUBJECTS,
    INDEX_SCHEMA_VERSION,
)
from .montage import MontageRecipe, build_montage_recipes


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Invalid SHA256 manifest line {line_number}")
        relative_path = parts[1].strip().replace("\\", "/")
        if relative_path in entries:
            raise ValueError(f"Duplicate SHA256 manifest path: {relative_path}")
        entries[relative_path] = parts[0].lower()
    return entries


def _recipe_payload(recipe: MontageRecipe) -> dict[str, Any]:
    return {
        "target_label": recipe.target_label,
        "mode": recipe.mode,
        "terms": [
            {
                "source_index": term.source_index,
                "source_label": term.source_label,
                "coefficient": term.coefficient,
            }
            for term in recipe.terms
        ],
    }


def _seizure_payload(
    summary: SummaryRecord, *, subject_id: str, duration_seconds: float
) -> list[dict[str, Any]]:
    seizures: list[dict[str, Any]] = []
    previous_end = -1
    for number, interval in enumerate(summary.seizures, start=1):
        if interval.end_seconds > duration_seconds:
            raise ValueError(
                f"{summary.file_name} seizure {number} ends after EDF duration"
            )
        if interval.start_seconds < previous_end:
            raise ValueError(f"{summary.file_name} has overlapping seizure intervals")
        previous_end = interval.end_seconds
        seizures.append({
            "event_id": f"{subject_id}:{summary.file_name}:seizure:{number:02d}",
            "event_number": number,
            "start_seconds": interval.start_seconds,
            "end_seconds": interval.end_seconds,
            "duration_seconds": interval.end_seconds - interval.start_seconds,
        })
    return seizures


def inspect_edf_record(
    path: Path,
    *,
    data_root: Path,
    subject_id: str,
    summary: SummaryRecord,
    source_sha256: str,
) -> dict[str, Any]:
    relative_path = path.relative_to(data_root).as_posix()
    with pyedflib.EdfReader(str(path)) as reader:
        signal_count = int(reader.signals_in_file)
        labels = list(reader.getSignalLabels())
        dimensions = [
            reader.getPhysicalDimension(index) for index in range(signal_count)
        ]
        sample_rates = [
            float(reader.getSampleFrequency(index)) for index in range(signal_count)
        ]
        duration_seconds = float(reader.getFileDuration())
        start = reader.getStartdatetime()
        recipes = build_montage_recipes(labels)
        used_indices = {
            term.source_index for recipe in recipes for term in recipe.terms
        }
        used_sample_rates = {sample_rates[index] for index in used_indices}
        if used_sample_rates != {float(EXPECTED_SAMPLE_RATE_HZ)}:
            raise ValueError(
                f"{relative_path} montage sample rates are {sorted(used_sample_rates)}"
            )
        sample_count = int(round(duration_seconds * EXPECTED_SAMPLE_RATE_HZ))
        available_samples = {
            int(reader.getNSamples()[index]) for index in used_indices
        }
        if available_samples != {sample_count}:
            raise ValueError(
                f"{relative_path} montage channels have unexpected sample counts "
                f"{sorted(available_samples)} != {sample_count}"
            )
    if path.name != summary.file_name:
        raise ValueError(f"Summary/EDF name mismatch: {summary.file_name} != {path.name}")
    seizures = _seizure_payload(
        summary, subject_id=subject_id, duration_seconds=duration_seconds
    )
    return {
        "record_id": relative_path,
        "subject_id": subject_id,
        "file_name": path.name,
        "source_sha256": source_sha256,
        "bytes": int(path.stat().st_size),
        "start_datetime": start.isoformat(),
        "end_datetime": (start + timedelta(seconds=duration_seconds)).isoformat(),
        "summary_start_time": summary.start_time,
        "summary_end_time": summary.end_time,
        "duration_seconds": duration_seconds,
        "sampling_frequency_hz": EXPECTED_SAMPLE_RATE_HZ,
        "sample_count": sample_count,
        "signal_count": signal_count,
        "signal_labels": labels,
        "physical_dimensions": dimensions,
        "montage": [_recipe_payload(recipe) for recipe in recipes],
        "montage_modes": dict(sorted(Counter(
            recipe.mode for recipe in recipes
        ).items())),
        "annotation_file_present": path.with_suffix(
            path.suffix + ".seizures"
        ).is_file(),
        "seizure_count": len(seizures),
        "seizures": seizures,
    }


def _load_summaries(root: Path) -> dict[str, dict[str, SummaryRecord]]:
    summaries: dict[str, dict[str, SummaryRecord]] = {}
    for subject_dir in sorted(root.glob("chb[0-9][0-9]")):
        subject_id = subject_dir.name
        path = subject_dir / f"{subject_id}-summary.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing patient summary: {path}")
        summaries[subject_id] = parse_summary(path)
    return summaries


def _statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    subjects = sorted({str(row["subject_id"]) for row in rows})
    return {
        "subject_count": len(subjects),
        "record_count": len(rows),
        "recording_hours": sum(
            float(row["duration_seconds"]) for row in rows
        ) / 3600.0,
        "seizure_count": sum(int(row["seizure_count"]) for row in rows),
        "ictal_seconds": sum(
            int(event["duration_seconds"])
            for row in rows
            for event in row["seizures"]
        ),
        "sampling_frequencies_hz": sorted({
            int(row["sampling_frequency_hz"]) for row in rows
        }),
        "signal_counts": dict(sorted(Counter(
            str(row["signal_count"]) for row in rows
        ).items())),
        "montage_modes": dict(sorted(Counter(
            mode
            for row in rows
            for mode, count in row["montage_modes"].items()
            for _ in range(int(count))
        ).items())),
        "by_subject": {
            subject: {
                "record_count": sum(row["subject_id"] == subject for row in rows),
                "recording_hours": sum(
                    float(row["duration_seconds"])
                    for row in rows
                    if row["subject_id"] == subject
                ) / 3600.0,
                "seizure_count": sum(
                    int(row["seizure_count"])
                    for row in rows
                    if row["subject_id"] == subject
                ),
            }
            for subject in subjects
        },
    }


def validate_chbmit_v1_index(index: Mapping[str, Any]) -> None:
    statistics = index["statistics"]
    expected = {
        "subject_count": EXPECTED_SUBJECTS,
        "record_count": EXPECTED_EDF_FILES,
        "seizure_count": EXPECTED_SEIZURES,
        "sampling_frequencies_hz": [EXPECTED_SAMPLE_RATE_HZ],
    }
    mismatches = {
        key: {"expected": value, "actual": statistics.get(key)}
        for key, value in expected.items()
        if statistics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CHB-MIT v1 acceptance failed: {mismatches}")
    if any(
        len(record["montage"]) != len(CANONICAL_BIPOLAR_CHANNELS)
        for record in index["records"]
    ):
        raise ValueError("One or more EDF records lack the canonical montage")


def build_chbmit_index(
    data_root: Path,
    *,
    strict_v1: bool = True,
    progress_every: int = 50,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CHB-MIT data root does not exist: {root}")
    sha_manifest_path = root / "SHA256SUMS.txt"
    records_manifest_path = root / "RECORDS"
    if not sha_manifest_path.is_file() or not records_manifest_path.is_file():
        raise FileNotFoundError("CHB-MIT RECORDS or SHA256SUMS.txt is missing")
    checksums = parse_sha256_manifest(sha_manifest_path)
    record_ids = [
        line.strip().replace("\\", "/")
        for line in records_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("RECORDS contains duplicate EDF paths")
    summaries = _load_summaries(root)
    indexed: list[dict[str, Any]] = []
    for number, record_id in enumerate(sorted(record_ids), start=1):
        path = root / Path(record_id)
        if not path.is_file():
            raise FileNotFoundError(f"EDF listed by RECORDS is missing: {record_id}")
        subject_id = Path(record_id).parts[0]
        summary = summaries.get(subject_id, {}).get(path.name)
        if summary is None:
            annotation_path = path.with_suffix(path.suffix + ".seizures")
            if annotation_path.is_file():
                raise ValueError(
                    f"Annotated EDF is missing from patient summary: {record_id}"
                )
            summary = SummaryRecord(
                file_name=path.name,
                start_time=None,
                end_time=None,
                seizure_count=0,
                seizures=(),
            )
        source_sha256 = checksums.get(record_id)
        if source_sha256 is None:
            raise ValueError(f"EDF is missing from SHA256SUMS.txt: {record_id}")
        indexed.append(
            inspect_edf_record(
                path,
                data_root=root,
                subject_id=subject_id,
                summary=summary,
                source_sha256=source_sha256,
            )
        )
        if progress_every and number % progress_every == 0:
            print(f"indexed EDF headers {number}/{len(record_ids)}", flush=True)
    body = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "path_semantics": "relative_to_data_root",
        "dataset_release": "chbmit/1.0.0",
        "source_manifests": {
            "records_sha256": _file_sha256(records_manifest_path),
            "sha256sums_sha256": _file_sha256(sha_manifest_path),
            "sha256_entry_count": len(checksums),
        },
        "target_montage": list(CANONICAL_BIPOLAR_CHANNELS),
        "records": indexed,
        "statistics": _statistics(indexed),
    }
    index = {**body, "index_sha256": canonical_hash(body)}
    if strict_v1:
        validate_chbmit_v1_index(index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/chbmit/1.0.0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit"),
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-strict-v1", action="store_true")
    args = parser.parse_args(argv)
    index = build_chbmit_index(
        args.data_root,
        strict_v1=not args.no_strict_v1,
        progress_every=args.progress_every,
    )
    output = args.output_dir.resolve() / (
        f"{INDEX_SCHEMA_VERSION}_{index['index_sha256'][:12]}.json"
    )
    write_content_addressed_json(index, output, hash_field="index_sha256")
    print(json.dumps({
        "index": str(output),
        "index_sha256": index["index_sha256"],
        "statistics": index["statistics"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

