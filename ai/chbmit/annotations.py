"""Parse the human-readable CHB-MIT patient summary files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FILE_BLOCK_PATTERN = re.compile(
    r"^File Name:[ \t]*(?P<name>\S+)\s*$"
    r"(?P<body>.*?)(?=^File Name:|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
TIME_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$")
SEIZURE_START_PATTERN = re.compile(
    r"^Seizure(?:\s+\d+)? Start Time:[ \t]*(\d+)\s*seconds\s*$",
    flags=re.MULTILINE,
)
SEIZURE_END_PATTERN = re.compile(
    r"^Seizure(?:\s+\d+)? End Time:[ \t]*(\d+)\s*seconds\s*$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class SeizureInterval:
    start_seconds: int
    end_seconds: int


@dataclass(frozen=True)
class SummaryRecord:
    file_name: str
    start_time: str | None
    end_time: str | None
    seizure_count: int
    seizures: tuple[SeizureInterval, ...]


def _extract_line(body: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:[ \t]*(.*?)\s*$", body, re.MULTILINE)
    if match is None:
        raise ValueError(f"Summary block is missing {field}")
    return match.group(1)


def _normalize_clock(value: str) -> str:
    match = TIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid summary clock time: {value!r}")
    hour, minute, second = map(int, match.groups())
    if minute > 59 or second > 59:
        raise ValueError(f"Invalid summary clock time: {value!r}")
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def parse_summary(path: Path) -> dict[str, SummaryRecord]:
    """Return summary metadata keyed by EDF basename."""
    text = Path(path).read_text(encoding="utf-8", errors="strict")
    records: dict[str, SummaryRecord] = {}
    for match in FILE_BLOCK_PATTERN.finditer(text):
        file_name = match.group("name")
        body = match.group("body")
        if file_name in records:
            raise ValueError(f"Duplicate summary entry for {file_name}")
        start_match = re.search(r"^File Start Time:[ \t]*(.*?)\s*$", body, re.MULTILINE)
        end_match = re.search(r"^File End Time:[ \t]*(.*?)\s*$", body, re.MULTILINE)
        start_time = (
            None if start_match is None else _normalize_clock(start_match.group(1))
        )
        end_time = None if end_match is None else _normalize_clock(end_match.group(1))
        seizure_count = int(_extract_line(body, "Number of Seizures in File"))
        starts = [int(value) for value in SEIZURE_START_PATTERN.findall(body)]
        ends = [int(value) for value in SEIZURE_END_PATTERN.findall(body)]
        if len(starts) != seizure_count or len(ends) != seizure_count:
            raise ValueError(
                f"{file_name} declares {seizure_count} seizures but has "
                f"{len(starts)} starts and {len(ends)} ends"
            )
        seizures = tuple(
            SeizureInterval(start_seconds=start, end_seconds=end)
            for start, end in zip(starts, ends, strict=True)
        )
        for interval in seizures:
            if interval.start_seconds < 0 or interval.end_seconds <= interval.start_seconds:
                raise ValueError(f"Invalid seizure interval in {file_name}: {interval}")
        records[file_name] = SummaryRecord(
            file_name=file_name,
            start_time=start_time,
            end_time=end_time,
            seizure_count=seizure_count,
            seizures=seizures,
        )
    if not records:
        raise ValueError(f"No EDF records found in summary: {path}")
    return records


