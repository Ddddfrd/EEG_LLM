"""Versioned contract for model input features and frequency bands."""
from __future__ import annotations

import json
from pathlib import Path


FEATURE_SCHEMA_VERSION = "eeg_features_welch_v2"
BAND_SCHEMA_PATH = Path(__file__).with_name("band_schema.json")


def _load_band_schema(path=BAND_SCHEMA_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bands = payload.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("Band schema must define at least one band")
    names = set()
    for band in bands:
        name = str(band.get("name", "")).strip().lower()
        low = float(band["low_hz"])
        high = float(band["high_hz"])
        if not name or name in names or low < 0 or high <= low:
            raise ValueError(f"Invalid frequency band: {band}")
        names.add(name)
    if not payload.get("schema_version") or not payload.get("history"):
        raise ValueError("Band schema requires schema_version and history")
    return payload


BAND_SCHEMA = _load_band_schema()
BAND_SCHEMA_VERSION = BAND_SCHEMA["schema_version"]
BAND_DEFS = tuple(
    (band["name"], float(band["low_hz"]), float(band["high_hz"]))
    for band in BAND_SCHEMA["bands"]
)
BAND_NAMES = tuple(name for name, _, _ in BAND_DEFS)
LEGACY_IMPLICIT_BAND_SCHEMAS = {
    "eeg_features_welch_v2": "eeg_bands_v1",
}


def checkpoint_band_schema_version(checkpoint):
    """Resolve explicit metadata or a narrowly versioned legacy contract."""
    explicit = checkpoint.get("band_schema_version")
    if explicit is not None:
        return explicit
    return LEGACY_IMPLICIT_BAND_SCHEMAS.get(checkpoint.get("feature_schema_version"))


PSD_METHOD = {
    "estimator": "scipy.signal.welch",
    "window": "hann",
    "nperseg": 2500,
    "noverlap": 1250,
    "detrend": "constant",
    "scaling": "density",
    "average": "mean",
    "band_integration": "sum(psd_bins) * frequency_resolution_hz",
    "band_feature": "log1p(integrated_band_power)",
}
