"""Stable contracts shared by the CHB-MIT ingestion and cache stages."""
from __future__ import annotations

CANONICAL_BIPOLAR_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)

INDEX_SCHEMA_VERSION = "chbmit_index_v1"
WINDOW_SCHEMA_VERSION = "chbmit_windows_v1"
RAW_CACHE_SCHEMA_VERSION = "chbmit_raw_windows_v1"
FEATURE_CACHE_SCHEMA_VERSION = "chbmit_features_v1"

EXPECTED_EDF_FILES = 686
EXPECTED_SUBJECTS = 24
EXPECTED_SEIZURES = 198
EXPECTED_SAMPLE_RATE_HZ = 256
