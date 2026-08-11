"""Memory-mapped CHB-MIT datasets with train-only robust feature scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import ChbmitWindowCache


@dataclass(frozen=True)
class RobustFeatureScaler:
    median: tuple[float, ...]
    scale: tuple[float, ...]
    quantile_range: tuple[float, float] = (10.0, 90.0)
    fit_scope: str = "source_train_channels_only"

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        indices: Sequence[int] | np.ndarray,
        *,
        quantile_range: tuple[float, float] = (10.0, 90.0),
    ) -> "RobustFeatureScaler":
        selected_indices = np.asarray(indices, dtype=np.int64)
        if selected_indices.ndim != 1 or selected_indices.size < 1:
            raise ValueError("Scaler fitting requires at least one window")
        if selected_indices.min() < 0 or selected_indices.max() >= len(features):
            raise IndexError("Scaler index is outside the feature cache")
        lower, upper = quantile_range
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("Invalid robust-scaler quantile range")
        selected = np.asarray(features[selected_indices], dtype=np.float32)
        if selected.ndim != 3:
            raise ValueError("Expected cached features shaped (N, C, F)")
        flattened = selected.reshape(-1, selected.shape[-1])
        median = np.median(flattened, axis=0)
        low = np.percentile(flattened, lower, axis=0)
        high = np.percentile(flattened, upper, axis=0)
        scale = high - low
        scale[scale < 1e-8] = 1.0
        if not np.isfinite(median).all() or not np.isfinite(scale).all():
            raise ValueError("Scaler fitting produced non-finite parameters")
        return cls(
            median=tuple(map(float, median)),
            scale=tuple(map(float, scale)),
            quantile_range=quantile_range,
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != len(self.median):
            raise ValueError("Feature width does not match scaler")
        median = np.asarray(self.median, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        transformed = (values - median) / scale
        if not np.isfinite(transformed).all():
            raise ValueError("Feature transform produced non-finite values")
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "robust_per_feature",
            "median": list(self.median),
            "scale": list(self.scale),
            "quantile_range": list(self.quantile_range),
            "fit_scope": self.fit_scope,
        }


def subject_indices(
    window_manifest: Mapping[str, Any],
    subjects: Sequence[str] | set[str],
) -> np.ndarray:
    selected = set(map(str, subjects))
    indices = [
        index
        for index, row in enumerate(window_manifest["windows"])
        if str(row["subject_id"]) in selected
    ]
    if not indices:
        raise ValueError(f"No windows found for subjects: {sorted(selected)}")
    return np.asarray(indices, dtype=np.int64)


def partition_indices(
    window_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    partition: str,
) -> np.ndarray:
    try:
        subjects = split["partitions"][partition]["subjects"]
    except KeyError as exc:
        raise ValueError(f"Unknown LOPO partition: {partition}") from exc
    return subject_indices(window_manifest, subjects)


def aggregate_channel_features(features: np.ndarray) -> np.ndarray:
    """Aggregate ``(N,C,F)`` features to channel-order invariant ``(N,3F)``."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] < 1:
        raise ValueError("Expected features shaped (windows, channels, features)")
    return np.concatenate(
        (
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.max(values, axis=1),
        ),
        axis=1,
    ).astype(np.float32)


class _BaseChbmitDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache: ChbmitWindowCache,
        window_manifest: Mapping[str, Any],
        indices: Sequence[int] | np.ndarray,
    ) -> None:
        self.cache = cache
        self.window_manifest = window_manifest
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size < 1:
            raise ValueError("Dataset requires at least one cache index")
        if self.indices.min() < 0 or self.indices.max() >= len(cache.labels):
            raise IndexError("Dataset index is outside the CHB-MIT cache")
        if len(window_manifest["windows"]) != len(cache.labels):
            raise ValueError("Window manifest and cache row counts differ")
        if (
            cache.metadata["window_manifest_sha256"]
            != window_manifest["window_manifest_sha256"]
        ):
            raise ValueError("Window manifest and cache hashes differ")

    def __len__(self) -> int:
        return int(self.indices.size)

    def _metadata(self, cache_index: int) -> dict[str, Any]:
        row = self.window_manifest["windows"][cache_index]
        return {
            "cache_index": cache_index,
            "window_id": str(row["window_id"]),
            "record_id": str(row["record_id"]),
            "subject_id": str(row["subject_id"]),
            "event_id": row["event_id"],
            "start_seconds": float(row["start_seconds"]),
        }


class ChbmitFeatureDataset(_BaseChbmitDataset):
    def __init__(
        self,
        cache: ChbmitWindowCache,
        window_manifest: Mapping[str, Any],
        indices: Sequence[int] | np.ndarray,
        *,
        scaler: RobustFeatureScaler | None = None,
    ) -> None:
        super().__init__(cache, window_manifest, indices)
        self.scaler = scaler

    def __getitem__(self, item: int) -> dict[str, Any]:
        cache_index = int(self.indices[item])
        features = np.asarray(
            self.cache.features[cache_index], dtype=np.float32
        )
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return {
            "features": torch.from_numpy(np.array(features, copy=True)),
            "label": int(self.cache.labels[cache_index]),
            **self._metadata(cache_index),
        }


class ChbmitRawDataset(_BaseChbmitDataset):
    def __getitem__(self, item: int) -> dict[str, Any]:
        cache_index = int(self.indices[item])
        waveform = np.asarray(
            self.cache.raw_windows[cache_index], dtype=np.float32
        )
        return {
            "waveform_uv": torch.from_numpy(np.array(waveform, copy=True)),
            "label": int(self.cache.labels[cache_index]),
            **self._metadata(cache_index),
        }
