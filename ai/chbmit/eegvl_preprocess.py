"""Versioned preprocessing for EEG-VL raw waveform experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


EEGVL_PREPROCESS_VERSION = "eegvl_preprocess_v1"


@dataclass(frozen=True)
class EegvlPreprocessConfig:
    sampling_frequency_hz: int = 256
    low_cut_hz: float = 0.5
    high_cut_hz: float = 45.0
    filter_order: int = 4
    clip_uv: float = 1024.0
    expected_channels: int = 18
    expected_samples: int = 1024
    apply_bandpass: bool = True

    def validate(self) -> None:
        if self.sampling_frequency_hz < 1:
            raise ValueError("sampling_frequency_hz must be positive")
        nyquist = self.sampling_frequency_hz / 2.0
        if not 0.0 < self.low_cut_hz < self.high_cut_hz < nyquist:
            raise ValueError("Band-pass frequencies must lie inside Nyquist")
        if self.filter_order < 1:
            raise ValueError("filter_order must be positive")
        if self.clip_uv <= 0:
            raise ValueError("clip_uv must be positive")
        if self.expected_channels < 1 or self.expected_samples < 1:
            raise ValueError("Expected waveform dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EEGVL_PREPROCESS_VERSION,
            **asdict(self),
            "normalization": "clip_then_divide_by_clip_uv",
            "output_layout": "image_channel,eeg_channel,time",
        }


@dataclass(frozen=True)
class EegvlPreprocessResult:
    values: np.ndarray
    nonfinite_replacements: int


def design_bandpass_sos(config: EegvlPreprocessConfig) -> np.ndarray:
    config.validate()
    return butter(
        config.filter_order,
        [config.low_cut_hz, config.high_cut_hz],
        btype="bandpass",
        fs=config.sampling_frequency_hz,
        output="sos",
    )


def preprocess_eeg_window(
    waveform_uv: np.ndarray,
    *,
    config: EegvlPreprocessConfig | None = None,
) -> EegvlPreprocessResult:
    settings = config or EegvlPreprocessConfig()
    settings.validate()
    values = np.asarray(waveform_uv, dtype=np.float64)
    expected = (settings.expected_channels, settings.expected_samples)
    if values.shape != expected:
        raise ValueError(f"Expected EEG window shaped {expected}, got {values.shape}")

    finite_mask = np.isfinite(values)
    replacement_count = int(values.size - np.count_nonzero(finite_mask))
    if replacement_count:
        values = values.copy()
        values[~finite_mask] = 0.0

    if settings.apply_bandpass:
        values = sosfiltfilt(design_bandpass_sos(settings), values, axis=-1)
    values = np.clip(values, -settings.clip_uv, settings.clip_uv)
    values = (values / settings.clip_uv).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("EEG-VL preprocessing produced non-finite values")
    return EegvlPreprocessResult(
        values=values[np.newaxis, ...],
        nonfinite_replacements=replacement_count,
    )
