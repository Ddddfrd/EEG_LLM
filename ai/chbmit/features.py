"""Channel-wise spectral and robust statistical features for CHB-MIT windows."""
from __future__ import annotations

import numpy as np
from scipy import signal


FREQUENCY_BANDS = (
    ("delta_log_power", 0.5, 4.0),
    ("theta_log_power", 4.0, 8.0),
    ("alpha_log_power", 8.0, 13.0),
    ("beta_log_power", 13.0, 30.0),
    ("gamma_log_power", 30.0, 45.0),
)
STATISTIC_FEATURES = (
    "mean_uv",
    "std_uv",
    "rms_uv",
    "peak_to_peak_uv",
    "mean_line_length_uv",
)
FEATURE_NAMES = tuple(name for name, _, _ in FREQUENCY_BANDS) + STATISTIC_FEATURES


def extract_channel_features(
    waveform_uv: np.ndarray,
    *,
    sampling_frequency_hz: int = 256,
) -> np.ndarray:
    """Return ``(channels, 10)`` log-power and amplitude features."""
    data = np.asarray(waveform_uv, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("Expected waveform with shape (channels, samples)")
    if not np.isfinite(data).all():
        raise ValueError("Waveform contains non-finite values")
    frequencies, psd = signal.welch(
        data,
        fs=sampling_frequency_hz,
        axis=1,
        nperseg=min(sampling_frequency_hz, data.shape[1]),
        noverlap=min(sampling_frequency_hz // 2, data.shape[1] // 2),
        detrend="constant",
        scaling="density",
    )
    spectral: list[np.ndarray] = []
    for _, lower, upper in FREQUENCY_BANDS:
        mask = (frequencies >= lower) & (frequencies < upper)
        if mask.sum() < 2:
            raise ValueError(f"Insufficient frequency bins for {lower}-{upper} Hz")
        power = np.trapz(psd[:, mask], frequencies[mask], axis=1)
        spectral.append(np.log10(np.maximum(power, np.finfo(np.float32).tiny)))
    centered = data - np.mean(data, axis=1, keepdims=True)
    statistics = (
        np.mean(data, axis=1),
        np.std(data, axis=1),
        np.sqrt(np.mean(np.square(centered), axis=1)),
        np.ptp(data, axis=1),
        np.mean(np.abs(np.diff(data, axis=1)), axis=1),
    )
    output = np.stack((*spectral, *statistics), axis=1).astype(np.float32)
    if output.shape != (data.shape[0], len(FEATURE_NAMES)):
        raise AssertionError("Feature extractor returned an invalid shape")
    if not np.isfinite(output).all():
        raise ValueError("Feature extraction produced non-finite values")
    return output
