"""Preprocessed caches and train-only augmentation for EEG-VL S1."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

from ai.v2.lightweight_dataset import write_content_addressed_json

from .cache import ChbmitWindowCache
from .eegvl_preprocess import EegvlPreprocessConfig, design_bandpass_sos
from .index import canonical_hash


EEGVL_S1_CACHE_SCHEMA_VERSION = "eegvl_s1_preprocessed_cache_v1"
EEGVL_S1_AUGMENT_VERSION = "eegvl_s1_online_augmentation_v1"
IMAGE_FILENAME = "images.npy"
METADATA_FILENAME = "metadata.json"
LEFT_RIGHT_MIRROR = (
    12, 13, 14, 15, 8, 9, 10, 11, 4,
    5, 6, 7, 0, 1, 2, 3, 16, 17,
)


@dataclass(frozen=True)
class S1PreprocessConfig:
    recipe_id: str = "p1_bandpass_clip_scale"
    sampling_frequency_hz: int = 256
    low_cut_hz: float = 0.5
    high_cut_hz: float = 45.0
    filter_order: int = 4
    clip_uv: float = 1024.0
    zscore_clip: float = 5.0

    def validate(self) -> None:
        if self.recipe_id not in {
            "p0_clip_scale",
            "p1_bandpass_clip_scale",
            "p2_bandpass_channel_zscore",
        }:
            raise ValueError(f"Unknown S1 preprocessing recipe: {self.recipe_id}")
        EegvlPreprocessConfig(
            sampling_frequency_hz=self.sampling_frequency_hz,
            low_cut_hz=self.low_cut_hz,
            high_cut_hz=self.high_cut_hz,
            filter_order=self.filter_order,
            clip_uv=self.clip_uv,
        ).validate()
        if self.zscore_clip <= 0:
            raise ValueError("zscore_clip must be positive")

    @property
    def apply_bandpass(self) -> bool:
        return self.recipe_id != "p0_clip_scale"

    @property
    def normalization(self) -> str:
        return (
            "channel_zscore"
            if self.recipe_id == "p2_bandpass_channel_zscore"
            else "clip_scale"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "apply_bandpass": self.apply_bandpass,
            "normalization": self.normalization,
            "output_range": [-1.0, 1.0],
        }


@dataclass(frozen=True)
class S1AugmentConfig:
    recipe_id: str = "a0_none"
    temporal_shift_samples: int = 0
    amplitude_min: float = 1.0
    amplitude_max: float = 1.0
    maximum_channel_cutout: int = 0
    gaussian_noise_probability: float = 0.0
    gaussian_noise_scale: tuple[float, float] = (0.0, 0.05)
    random_filter_probability: float = 0.0
    random_low_cut_hz: tuple[float, float] = (0.5, 2.0)
    random_high_cut_hz: tuple[float, float] = (35.0, 45.0)
    mirror_probability: float = 0.0
    sampling_frequency_hz: int = 256
    filter_order: int = 4

    def validate(self) -> None:
        if self.temporal_shift_samples < 0:
            raise ValueError("temporal_shift_samples must be non-negative")
        if not 0 < self.amplitude_min <= self.amplitude_max:
            raise ValueError("Invalid amplitude range")
        if not 0 <= self.maximum_channel_cutout < 18:
            raise ValueError("Invalid channel-cutout limit")
        for probability in (
            self.gaussian_noise_probability,
            self.random_filter_probability,
            self.mirror_probability,
        ):
            if not 0 <= probability <= 1:
                raise ValueError("Augmentation probabilities must be in [0, 1]")
        noise_min, noise_max = self.gaussian_noise_scale
        if not 0 <= noise_min <= noise_max:
            raise ValueError("Invalid Gaussian-noise scale")
        low_min, low_max = self.random_low_cut_hz
        high_min, high_max = self.random_high_cut_hz
        if not 0 < low_min <= low_max < high_min <= high_max:
            raise ValueError("Invalid random-filter frequency ranges")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EEGVL_S1_AUGMENT_VERSION,
            **asdict(self),
            "mirror_indices": list(LEFT_RIGHT_MIRROR),
        }


def augmentation_recipe(recipe_id: str) -> S1AugmentConfig:
    recipes = {
        "a0_none": S1AugmentConfig(),
        "a1_shift_amplitude": S1AugmentConfig(
            recipe_id="a1_shift_amplitude",
            temporal_shift_samples=128,
            amplitude_min=0.8,
            amplitude_max=1.2,
        ),
        "a2_channel_cutout": S1AugmentConfig(
            recipe_id="a2_channel_cutout",
            maximum_channel_cutout=2,
            temporal_shift_samples=128,
            amplitude_min=0.8,
            amplitude_max=1.2,
        ),
        "a3_random_filter": S1AugmentConfig(
            recipe_id="a3_random_filter",
            maximum_channel_cutout=2,
            gaussian_noise_probability=0.5,
            random_filter_probability=0.5,
            temporal_shift_samples=128,
            amplitude_min=0.8,
            amplitude_max=1.2,
        ),
        "a4_channel_mirror": S1AugmentConfig(
            recipe_id="a4_channel_mirror",
            maximum_channel_cutout=2,
            gaussian_noise_probability=0.5,
            random_filter_probability=0.5,
            mirror_probability=0.5,
            temporal_shift_samples=128,
            amplitude_min=0.8,
            amplitude_max=1.2,
        ),
    }
    try:
        recipe = recipes[recipe_id]
    except KeyError as exc:
        raise ValueError(f"Unknown augmentation recipe: {recipe_id}") from exc
    recipe.validate()
    return recipe


def preprocess_s1_batch(
    waveforms_uv: np.ndarray,
    *,
    config: S1PreprocessConfig,
) -> tuple[np.ndarray, int]:
    config.validate()
    values = np.asarray(waveforms_uv, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1] not in {18, 20}
        or values.shape[2] != 1024
    ):
        raise ValueError(
            "Expected S1 waveform batch shaped (batch, 18|20, 1024)"
        )
    finite = np.isfinite(values)
    replacements = int(values.size - np.count_nonzero(finite))
    if replacements:
        values = values.copy()
        values[~finite] = 0.0
    if config.apply_bandpass:
        base_config = EegvlPreprocessConfig(
            sampling_frequency_hz=config.sampling_frequency_hz,
            low_cut_hz=config.low_cut_hz,
            high_cut_hz=config.high_cut_hz,
            filter_order=config.filter_order,
            clip_uv=config.clip_uv,
        )
        values = sosfiltfilt(
            design_bandpass_sos(base_config),
            values,
            axis=-1,
        )
    if config.normalization == "channel_zscore":
        center = values.mean(axis=-1, keepdims=True)
        scale = values.std(axis=-1, keepdims=True)
        scale = np.maximum(scale, np.finfo(np.float32).eps)
        values = np.clip(
            (values - center) / scale,
            -config.zscore_clip,
            config.zscore_clip,
        ) / config.zscore_clip
    else:
        values = np.clip(values, -config.clip_uv, config.clip_uv)
        values = values / config.clip_uv
    result = values[:, np.newaxis].astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("S1 preprocessing produced non-finite values")
    return result, replacements


class S1PreprocessedCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        metadata_path = self.path / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete S1 cache: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in self.metadata.items()
            if key != "metadata_sha256"
        }
        if self.metadata.get("metadata_sha256") != canonical_hash(body):
            raise ValueError("S1 cache metadata hash is invalid")
        if self.metadata.get("schema_version") != EEGVL_S1_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported S1 cache schema")
        self.images = np.load(self.path / IMAGE_FILENAME, mmap_mode="r")
        if tuple(self.images.shape) != tuple(self.metadata["image_shape"]):
            raise ValueError("S1 image cache shape is invalid")


def build_s1_preprocessed_cache(
    source_cache: ChbmitWindowCache,
    *,
    output_dir: Path,
    config: S1PreprocessConfig,
    batch_size: int = 64,
) -> Path:
    config.validate()
    if batch_size < 1:
        raise ValueError("S1 cache batch_size must be positive")
    contract = {
        "schema_version": EEGVL_S1_CACHE_SCHEMA_VERSION,
        "source_metadata_sha256": source_cache.metadata["metadata_sha256"],
        "preprocess": config.to_dict(),
        "window_count": int(len(source_cache.labels)),
        "image_shape": [
            int(len(source_cache.labels)),
            1,
            int(source_cache.raw_windows.shape[1]),
            int(source_cache.raw_windows.shape[2]),
        ],
        "image_dtype": "float16",
        "row_order": "source_cache_row_order",
    }
    cache_key = canonical_hash(contract)
    destination = Path(output_dir).resolve() / (
        f"preprocessed_{cache_key[:12]}"
    )
    if destination.exists():
        loaded = S1PreprocessedCache(destination)
        if loaded.metadata["cache_key"] != cache_key:
            raise ValueError("Existing S1 cache key does not match")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    try:
        image_memmap = np.lib.format.open_memmap(
            temporary / IMAGE_FILENAME,
            mode="w+",
            dtype=np.float16,
            shape=tuple(contract["image_shape"]),
        )
        replacements = 0
        for start in range(0, len(source_cache.labels), batch_size):
            end = min(start + batch_size, len(source_cache.labels))
            processed, replaced = preprocess_s1_batch(
                source_cache.raw_windows[start:end],
                config=config,
            )
            image_memmap[start:end] = processed.astype(np.float16)
            replacements += replaced
        image_memmap.flush()
        mapped_file = getattr(image_memmap, "_mmap", None)
        if mapped_file is not None:
            mapped_file.close()
        del image_memmap
        body = {
            **contract,
            "cache_key": cache_key,
            "nonfinite_replacements": replacements,
            "image_bytes": int((temporary / IMAGE_FILENAME).stat().st_size),
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
    S1PreprocessedCache(destination)
    return destination


def _reflect_shift(values: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return values
    padding = abs(shift)
    padded = np.pad(values, ((0, 0), (padding, padding)), mode="reflect")
    start = padding - shift
    return padded[:, start : start + values.shape[1]]


def augment_s1_image(
    image: np.ndarray,
    *,
    config: S1AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    config.validate()
    values = np.asarray(image, dtype=np.float32)
    if values.shape != (1, 18, 1024):
        raise ValueError("S1 augmentation expects an image shaped (1, 18, 1024)")
    values = values[0].copy()
    mirrored = bool(rng.random() < config.mirror_probability)
    if mirrored:
        values = values[np.asarray(LEFT_RIGHT_MIRROR)]
    shift = (
        int(rng.integers(
            -config.temporal_shift_samples,
            config.temporal_shift_samples + 1,
        ))
        if config.temporal_shift_samples
        else 0
    )
    values = _reflect_shift(values, shift)
    amplitude = float(rng.uniform(config.amplitude_min, config.amplitude_max))
    values *= amplitude

    noise_applied = bool(rng.random() < config.gaussian_noise_probability)
    noise_fraction: float | None = None
    if noise_applied:
        noise_fraction = float(rng.uniform(*config.gaussian_noise_scale))
        channel_median = np.median(values, axis=-1, keepdims=True)
        channel_mad = np.median(
            np.abs(values - channel_median),
            axis=-1,
            keepdims=True,
        )
        robust_scale = np.maximum(
            1.4826 * channel_mad,
            np.finfo(np.float32).eps,
        )
        values += rng.normal(
            loc=0.0,
            scale=noise_fraction * robust_scale,
            size=values.shape,
        )

    filter_applied = bool(rng.random() < config.random_filter_probability)
    filter_range: list[float] | None = None
    if filter_applied:
        low = float(rng.uniform(*config.random_low_cut_hz))
        high = float(rng.uniform(*config.random_high_cut_hz))
        sos = butter(
            config.filter_order,
            [low, high],
            btype="bandpass",
            fs=config.sampling_frequency_hz,
            output="sos",
        )
        values = sosfiltfilt(sos, values, axis=-1)
        filter_range = [low, high]

    cutout_count = (
        int(rng.integers(0, config.maximum_channel_cutout + 1))
        if config.maximum_channel_cutout
        else 0
    )
    cutout_channels: list[int] = []
    if cutout_count:
        cutout_channels = sorted(map(
            int,
            rng.choice(18, size=cutout_count, replace=False),
        ))
        values[cutout_channels] = 0.0
    values = np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)
    return values[np.newaxis], {
        "mirrored": mirrored,
        "temporal_shift_samples": shift,
        "amplitude_scale": amplitude,
        "gaussian_noise_fraction": noise_fraction,
        "random_filter_hz": filter_range,
        "cutout_channels": cutout_channels,
    }


class S1ImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        image_cache: S1PreprocessedCache,
        source_cache: ChbmitWindowCache,
        window_manifest: Mapping[str, Any],
        indices: Sequence[int] | np.ndarray,
        *,
        augmentation: S1AugmentConfig | None = None,
        seed: int = 42,
        subject_baselines: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.image_cache = image_cache
        self.source_cache = source_cache
        self.window_manifest = window_manifest
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augmentation = augmentation
        self.seed = int(seed)
        self.epoch = 0
        self.subject_baselines = {
            str(subject): np.asarray(values, dtype=np.float32)
            for subject, values in (subject_baselines or {}).items()
        }
        if self.indices.ndim != 1 or not self.indices.size:
            raise ValueError("S1ImageDataset requires at least one index")
        if self.indices.min() < 0 or self.indices.max() >= len(source_cache.labels):
            raise IndexError("S1 dataset index is outside the source cache")
        if len(image_cache.images) != len(source_cache.labels):
            raise ValueError("S1 image and source cache lengths differ")
        if (
            image_cache.metadata["source_metadata_sha256"]
            != source_cache.metadata["metadata_sha256"]
        ):
            raise ValueError("S1 image cache does not match source cache")
        if augmentation is not None:
            augmentation.validate()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        cache_index = int(self.indices[item])
        image = np.asarray(self.image_cache.images[cache_index], dtype=np.float32)
        if self.augmentation is not None:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, self.epoch, cache_index]
            ))
            image, _ = augment_s1_image(
                image,
                config=self.augmentation,
                rng=rng,
            )
        row = self.window_manifest["windows"][cache_index]
        result = {
            "image": torch.from_numpy(np.array(image, copy=True)),
            "label": int(self.source_cache.labels[cache_index]),
            "cache_index": cache_index,
            "window_id": str(row["window_id"]),
            "subject_id": str(row["subject_id"]),
        }
        if self.subject_baselines:
            subject = str(row["subject_id"])
            try:
                baseline = self.subject_baselines[subject]
            except KeyError as exc:
                raise ValueError(
                    f"Missing spectral baseline for {subject}"
                ) from exc
            result["baseline_log_magnitude"] = torch.from_numpy(baseline)
        return result
