"""Preprocessed natural-timeline cache for repeated EEG-VL evaluation."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyedflib
import torch
from torch.utils.data import Dataset

from ai.v2.lightweight_dataset import write_content_addressed_json

from .cache import _recipe_from_payload
from .deep_timeline import DeepTargetTimeline
from .eegvl_s1_data import S1PreprocessConfig, preprocess_s1_batch
from .index import canonical_hash
from .montage import read_montage_window


NATURAL_FOLD_CACHE_SCHEMA_VERSION = "eeg_continual_natural_fold_cache_v1"
IMAGE_FILENAME = "images.npy"
LABEL_FILENAME = "labels.npy"
METADATA_FILENAME = "metadata.json"


class NaturalFoldImageCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        metadata_path = self.path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete natural fold cache: {metadata_path}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in self.metadata.items()
            if key != "metadata_sha256"
        }
        if self.metadata.get("metadata_sha256") != canonical_hash(body):
            raise ValueError("Natural fold cache metadata hash is invalid")
        if (
            self.metadata.get("schema_version")
            != NATURAL_FOLD_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported natural fold cache schema")
        self.images = np.load(self.path / IMAGE_FILENAME, mmap_mode="r")
        self.labels = np.load(self.path / LABEL_FILENAME, mmap_mode="r")
        expected_images = tuple(self.metadata["image_shape"])
        expected_labels = (int(self.metadata["window_count"]),)
        if self.images.shape != expected_images:
            raise ValueError("Natural fold image shape is invalid")
        if self.labels.shape != expected_labels:
            raise ValueError("Natural fold label shape is invalid")

    def subject_slice(self, subject: str) -> slice:
        try:
            payload = self.metadata["subjects"][subject]
        except KeyError as exc:
            raise ValueError(f"Unknown cached subject: {subject}") from exc
        return slice(int(payload["row_start"]), int(payload["row_end"]))


class NaturalFoldImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache: NaturalFoldImageCache,
        *,
        subject_baselines: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.cache = cache
        self.subject_baselines = {
            str(subject): np.asarray(values, dtype=np.float32)
            for subject, values in (subject_baselines or {}).items()
        }

    def __len__(self) -> int:
        return int(self.cache.metadata["window_count"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        image = np.asarray(self.cache.images[index], dtype=np.float32)
        result = {
            "image": torch.from_numpy(image.copy()),
            "label": torch.tensor(
                int(self.cache.labels[index]), dtype=torch.long
            ),
        }
        if self.subject_baselines:
            subject = next(
                (
                    str(candidate)
                    for candidate in self.cache.metadata["subject_order"]
                    if (
                        int(
                            self.cache.metadata["subjects"][candidate][
                                "row_start"
                            ]
                        )
                        <= index
                        < int(
                            self.cache.metadata["subjects"][candidate]["row_end"]
                        )
                    )
                ),
                None,
            )
            if subject is None or subject not in self.subject_baselines:
                raise ValueError(
                    f"Missing spectral baseline for natural-cache row {index}"
                )
            result["baseline_log_magnitude"] = torch.from_numpy(
                self.subject_baselines[subject]
            )
        return result


def _cache_contract(
    index: Mapping[str, Any],
    timelines: Mapping[str, DeepTargetTimeline],
    preprocess: S1PreprocessConfig,
) -> dict[str, Any]:
    subjects: dict[str, Any] = {}
    offset = 0
    for subject, timeline in timelines.items():
        count = int(timeline.metadata["window_count"])
        subjects[subject] = {
            "row_start": offset,
            "row_end": offset + count,
            "window_count": count,
            "normal_windows": int(timeline.metadata["label_counts"]["normal"]),
            "ictal_windows": int(timeline.metadata["label_counts"]["ictal"]),
            "timeline_metadata_sha256": timeline.metadata[
                "metadata_sha256"
            ],
        }
        offset += count
    return {
        "schema_version": NATURAL_FOLD_CACHE_SCHEMA_VERSION,
        "dataset_index_sha256": str(index["index_sha256"]),
        "preprocess": preprocess.to_dict(),
        "subjects": subjects,
        "subject_order": list(timelines),
        "window_count": offset,
        "image_shape": [offset, 1, len(index["target_montage"]), 1024],
        "image_dtype": "float16",
        "label_dtype": "uint8",
        "row_order": "subject_order_then_timeline_row_order",
        "construction": (
            "read each EDF montage once, then preprocess indexed windows in batches"
        ),
    }


def _close_memmap(values: np.memmap[Any, Any]) -> None:
    values.flush()
    mapped = getattr(values, "_mmap", None)
    if mapped is not None:
        mapped.close()


def build_natural_fold_image_cache(
    index: Mapping[str, Any],
    timelines: Mapping[str, DeepTargetTimeline],
    *,
    data_root: Path,
    output_dir: Path,
    preprocess: S1PreprocessConfig,
    batch_size: int = 128,
    progress_every: int = 10_000,
) -> Path:
    if batch_size < 1 or progress_every < 0:
        raise ValueError("Natural fold cache batch/progress settings are invalid")
    if not timelines:
        raise ValueError("Natural fold cache requires at least one timeline")
    contract = _cache_contract(index, timelines, preprocess)
    cache_key = canonical_hash(contract)
    destination = Path(output_dir).resolve() / (
        f"natural_fold_{cache_key[:12]}"
    )
    if destination.exists():
        loaded = NaturalFoldImageCache(destination)
        if loaded.metadata["cache_key"] != cache_key:
            raise ValueError("Existing natural fold cache key does not match")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    images_cleanup: np.memmap[Any, Any] | None = None
    labels_cleanup: np.memmap[Any, Any] | None = None
    try:
        images = np.lib.format.open_memmap(
            temporary / IMAGE_FILENAME,
            mode="w+",
            dtype=np.float16,
            shape=tuple(contract["image_shape"]),
        )
        labels = np.lib.format.open_memmap(
            temporary / LABEL_FILENAME,
            mode="w+",
            dtype=np.uint8,
            shape=(int(contract["window_count"]),),
        )
        images_cleanup = images
        labels_cleanup = labels
        records = {
            str(record["record_id"]): record for record in index["records"]
        }
        root = Path(data_root).resolve()
        written = 0
        nonfinite_replacements = 0
        next_progress = progress_every
        for subject, timeline in timelines.items():
            subject_offset = int(contract["subjects"][subject]["row_start"])
            labels[
                subject_offset : subject_offset + len(timeline.labels)
            ] = timeline.labels
            for timeline_record in timeline.metadata["records"]:
                local_start = int(timeline_record["row_start"])
                local_end = int(timeline_record["row_end"])
                if local_end <= local_start:
                    continue
                record_id = str(timeline_record["record_id"])
                record = records.get(record_id)
                if record is None:
                    raise ValueError(f"Timeline references unknown EDF: {record_id}")
                path = root / Path(record_id)
                if not path.is_file():
                    raise FileNotFoundError(f"EDF is missing: {path}")
                recipes = tuple(
                    _recipe_from_payload(payload)
                    for payload in record["montage"]
                )
                reader = pyedflib.EdfReader(str(path))
                try:
                    full_waveform = read_montage_window(
                        reader,
                        recipes,
                        start_sample=0,
                        sample_count=int(record["sample_count"]),
                    )
                finally:
                    reader.close()
                starts = np.asarray(
                    timeline.start_samples[local_start:local_end],
                    dtype=np.int64,
                )
                for batch_start in range(0, len(starts), batch_size):
                    batch_end = min(batch_start + batch_size, len(starts))
                    waveforms = np.stack([
                        full_waveform[
                            :,
                            int(start) : int(start) + 1024,
                        ]
                        for start in starts[batch_start:batch_end]
                    ])
                    processed, replacements = preprocess_s1_batch(
                        waveforms,
                        config=preprocess,
                    )
                    target_start = subject_offset + local_start + batch_start
                    target_end = target_start + len(processed)
                    images[target_start:target_end] = processed.astype(
                        np.float16
                    )
                    written += len(processed)
                    nonfinite_replacements += replacements
                    if progress_every and written >= next_progress:
                        images.flush()
                        labels.flush()
                        print(
                            "cached natural Fold windows "
                            f"{written}/{contract['window_count']}",
                            flush=True,
                        )
                        next_progress += progress_every
                del full_waveform
        if written != int(contract["window_count"]):
            raise ValueError(
                f"Natural fold cache wrote {written} rows, expected "
                f"{contract['window_count']}"
            )
        _close_memmap(images)
        _close_memmap(labels)
        images_cleanup = None
        labels_cleanup = None
        body = {
            **contract,
            "cache_key": cache_key,
            "nonfinite_replacements": nonfinite_replacements,
            "files": {
                IMAGE_FILENAME: int((temporary / IMAGE_FILENAME).stat().st_size),
                LABEL_FILENAME: int((temporary / LABEL_FILENAME).stat().st_size),
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
        if images_cleanup is not None:
            _close_memmap(images_cleanup)
        if labels_cleanup is not None:
            _close_memmap(labels_cleanup)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    NaturalFoldImageCache(destination)
    return destination
