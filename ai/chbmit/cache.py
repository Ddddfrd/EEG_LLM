"""Materialize selected CHB-MIT raw windows and lightweight features."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyedflib
import scipy

from ai.v2.lightweight_dataset import write_content_addressed_json

from .contracts import (
    FEATURE_CACHE_SCHEMA_VERSION,
    RAW_CACHE_SCHEMA_VERSION,
    WINDOW_SCHEMA_VERSION,
)
from .features import FEATURE_NAMES, extract_channel_features
from .index import canonical_hash
from .montage import MontageRecipe, MontageTerm, read_montage_window
from .windows import load_chbmit_index


RAW_FILENAME = "raw_windows_uv.npy"
FEATURE_FILENAME = "channel_features.npy"
LABEL_FILENAME = "labels.npy"
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


def load_window_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != WINDOW_SCHEMA_VERSION:
        raise ValueError("Unsupported CHB-MIT window manifest schema")
    expected = canonical_hash({
        key: value
        for key, value in manifest.items()
        if key != "window_manifest_sha256"
    })
    if manifest.get("window_manifest_sha256") != expected:
        raise ValueError("CHB-MIT window manifest hash is invalid")
    return manifest


class ChbmitWindowCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        metadata_path = self.path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete CHB-MIT cache: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in self.metadata.items()
            if key != "metadata_sha256"
        }
        if self.metadata.get("metadata_sha256") != canonical_hash(body):
            raise ValueError("CHB-MIT cache metadata hash is invalid")
        self.raw_windows = np.load(self.path / RAW_FILENAME, mmap_mode="r")
        self.features = np.load(self.path / FEATURE_FILENAME, mmap_mode="r")
        self.labels = np.load(self.path / LABEL_FILENAME, mmap_mode="r")
        expected_raw_shape = tuple(self.metadata["raw_shape"])
        expected_feature_shape = tuple(self.metadata["feature_shape"])
        if self.raw_windows.shape != expected_raw_shape:
            raise ValueError("CHB-MIT raw cache shape is invalid")
        if self.features.shape != expected_feature_shape:
            raise ValueError("CHB-MIT feature cache shape is invalid")
        if self.labels.shape != (expected_raw_shape[0],):
            raise ValueError("CHB-MIT label cache shape is invalid")


def _cache_contract(
    index: Mapping[str, Any],
    windows: Mapping[str, Any],
) -> dict[str, Any]:
    window_count = len(windows["windows"])
    channels, samples = map(int, windows["window_shape"])
    return {
        "raw_schema_version": RAW_CACHE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "dataset_index_sha256": str(index["index_sha256"]),
        "window_manifest_sha256": str(windows["window_manifest_sha256"]),
        "window_count": window_count,
        "raw_shape": [window_count, channels, samples],
        "feature_shape": [window_count, channels, len(FEATURE_NAMES)],
        "raw_unit": "microvolts",
        "raw_dtype": "float32",
        "feature_dtype": "float32",
        "label_dtype": "uint8",
        "feature_names": list(FEATURE_NAMES),
        "library_versions": {
            "numpy": np.__version__,
            "pyedflib": pyedflib.__version__,
            "scipy": scipy.__version__,
        },
        "row_order": "window_manifest_order",
    }


def build_window_caches(
    index: Mapping[str, Any],
    windows: Mapping[str, Any],
    *,
    data_root: Path,
    output_dir: Path,
    progress_every: int = 500,
) -> Path:
    if windows.get("dataset_index_sha256") != index.get("index_sha256"):
        raise ValueError("Window manifest and dataset index do not match")
    contract = _cache_contract(index, windows)
    cache_key = canonical_hash(contract)
    destination = Path(output_dir).resolve() / f"windows_{cache_key[:12]}"
    if destination.exists():
        loaded = ChbmitWindowCache(destination)
        if loaded.metadata["cache_key"] != cache_key:
            raise ValueError("Existing CHB-MIT cache key does not match")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    try:
        raw_shape = tuple(contract["raw_shape"])
        feature_shape = tuple(contract["feature_shape"])
        raw = np.lib.format.open_memmap(
            temporary / RAW_FILENAME,
            mode="w+",
            dtype=np.float32,
            shape=raw_shape,
        )
        features = np.lib.format.open_memmap(
            temporary / FEATURE_FILENAME,
            mode="w+",
            dtype=np.float32,
            shape=feature_shape,
        )
        labels = np.lib.format.open_memmap(
            temporary / LABEL_FILENAME,
            mode="w+",
            dtype=np.uint8,
            shape=(raw_shape[0],),
        )
        records = {
            str(record["record_id"]): record for record in index["records"]
        }
        rows = list(windows["windows"])
        root = Path(data_root).resolve()
        current_record_id: str | None = None
        reader: pyedflib.EdfReader | None = None
        recipes: tuple[MontageRecipe, ...] = ()
        try:
            for row_index, row in enumerate(rows):
                record_id = str(row["record_id"])
                if record_id != current_record_id:
                    if reader is not None:
                        reader.close()
                    record = records.get(record_id)
                    if record is None:
                        raise ValueError(f"Window references unknown EDF: {record_id}")
                    path = root / Path(record_id)
                    if not path.is_file():
                        raise FileNotFoundError(f"EDF is missing: {path}")
                    reader = pyedflib.EdfReader(str(path))
                    current_record_id = record_id
                    recipes = tuple(
                        _recipe_from_payload(payload)
                        for payload in record["montage"]
                    )
                if reader is None:
                    raise AssertionError("EDF reader was not initialized")
                waveform = read_montage_window(
                    reader,
                    recipes,
                    start_sample=int(row["start_sample"]),
                    sample_count=int(row["sample_count"]),
                )
                raw[row_index] = waveform
                features[row_index] = extract_channel_features(
                    waveform,
                    sampling_frequency_hz=int(
                        windows["config"]["sampling_frequency_hz"]
                    ),
                )
                labels[row_index] = int(row["label"])
                if progress_every and (row_index + 1) % progress_every == 0:
                    raw.flush()
                    features.flush()
                    labels.flush()
                    print(
                        f"cached CHB-MIT windows {row_index + 1}/{len(rows)}",
                        flush=True,
                    )
        finally:
            if reader is not None:
                reader.close()
        raw.flush()
        features.flush()
        labels.flush()
        del raw, features, labels
        files = {
            name: int((temporary / name).stat().st_size)
            for name in (RAW_FILENAME, FEATURE_FILENAME, LABEL_FILENAME)
        }
        body = {
            **contract,
            "cache_key": cache_key,
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
    ChbmitWindowCache(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/chbmit/1.0.0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/chbmit/cache"),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args(argv)
    index = load_chbmit_index(args.index)
    windows = load_window_manifest(args.windows)
    output = build_window_caches(
        index,
        windows,
        data_root=args.data_root,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    cache = ChbmitWindowCache(output)
    print(json.dumps({
        "cache": str(output),
        "cache_key": cache.metadata["cache_key"],
        "raw_shape": cache.metadata["raw_shape"],
        "feature_shape": cache.metadata["feature_shape"],
        "files": cache.metadata["files"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
