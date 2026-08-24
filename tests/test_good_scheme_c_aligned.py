from __future__ import annotations

import numpy as np

from ai.chbmit.contracts import CANONICAL_BIPOLAR_CHANNELS
from ai.chbmit.direct20 import DIRECT20_CHANNELS, build_direct20_index
from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegvl_s1_data import S1PreprocessConfig, preprocess_s1_batch
from ai.chbmit.index import canonical_hash
from good.e1_e2_e3_e4_fullband.train_scheme_c_aligned import (
    _calibration_count,
    _cap_manifest_per_patient,
)


def _minimal_index() -> dict[str, object]:
    body = {
        "schema_version": "chbmit_index_v1",
        "target_montage": list(CANONICAL_BIPOLAR_CHANNELS),
        "records": [
            {
                "subject_id": "chb01",
                "signal_labels": list(CANONICAL_BIPOLAR_CHANNELS),
                "montage": [],
                "montage_modes": {},
            }
        ],
    }
    return {**body, "index_sha256": canonical_hash(body)}


def test_direct20_uses_real_extras_and_explicit_missing_zero() -> None:
    index = build_direct20_index(_minimal_index())

    assert tuple(index["target_montage"]) == DIRECT20_CHANNELS
    recipes = index["records"][0]["montage"]
    assert len(recipes) == 20
    assert recipes[-2]["target_label"] == "T7-FT9"
    assert recipes[-2]["mode"] == "missing_zero"
    assert recipes[-2]["terms"] == []
    assert recipes[-1]["target_label"] == "FT10-T8"
    assert recipes[-1]["mode"] == "missing_zero"


def test_preprocess_and_stft_accept_direct20() -> None:
    processed, replacements = preprocess_s1_batch(
        np.zeros((2, 20, 1024), dtype=np.float32),
        config=S1PreprocessConfig(recipe_id="p0_clip_scale"),
    )
    config = ServerSTFTConfig(
        source_channels=20,
        eeg_channels=20,
        n_fft=64,
        win_length=64,
        hop_length=32,
    )

    assert processed.shape == (2, 1, 20, 1024)
    assert replacements == 0
    assert config.image_shape == (660, 33)


def test_scheme_c_calibration_fraction_and_cap_are_deterministic() -> None:
    assert _calibration_count(1) == 1
    assert _calibration_count(100) == 20
    assert _calibration_count(30_000) == 4000

    rows = [
        {
            "window_id": f"chb01/x@{index}",
            "subject_id": "chb01",
            "label": index % 2,
        }
        for index in range(6)
    ]
    body = {
        "schema_version": "chbmit_windows_v1",
        "dataset_index_schema": "chbmit_index_v1",
        "dataset_index_sha256": "index",
        "config": {},
        "window_shape": [20, 1024],
        "selection_contract": {},
        "windows": rows,
        "statistics": {
            "selected_windows": 6,
            "selected_ictal": 3,
            "selected_normal": 3,
        },
    }
    manifest = {**body, "window_manifest_sha256": canonical_hash(body)}
    capped = _cap_manifest_per_patient(manifest, maximum=4)

    assert len(capped["windows"]) == 4
    assert capped["statistics"]["patient_cap"] == 4
    assert capped["statistics"]["selected_ictal"] == 2
    assert capped["statistics"]["selected_normal"] == 2
    expected = canonical_hash({
        key: value
        for key, value in capped.items()
        if key != "window_manifest_sha256"
    })
    assert capped["window_manifest_sha256"] == expected
