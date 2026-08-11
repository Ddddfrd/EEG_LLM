from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy import signal

from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegmamba_b import (
    EEGMambaBE2Classifier,
    EEGMambaInputConfig,
    fourier_resample_real,
    waveform_to_eegmamba_patches,
)


class FakeEEGMamba(nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        summary = patches.mean(dim=-1, keepdim=True)
        return summary.expand(*summary.shape[:-1], 200)


def test_fourier_resample_matches_scipy_real_signal() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2, 3, 1024)).astype(np.float32)
    expected = signal.resample(values, 800, axis=-1)
    actual = fourier_resample_real(torch.from_numpy(values), 800).numpy()

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_waveform_to_eegmamba_patches_restores_official_scale() -> None:
    waveform = torch.full((2, 1, 18, 1024), 100.0 / 1024.0)
    patches = waveform_to_eegmamba_patches(
        waveform,
        config=EEGMambaInputConfig(),
    )

    assert patches.shape == (2, 18, 4, 200)
    torch.testing.assert_close(patches, torch.ones_like(patches))


def test_waveform_to_eegmamba_patches_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="18, 1024"):
        waveform_to_eegmamba_patches(
            torch.zeros(2, 1, 20, 1024),
            config=EEGMambaInputConfig(),
        )


def test_mamba_b_fuses_backbone_and_e2_without_optional_mamba_package() -> None:
    model = EEGMambaBE2Classifier(
        backbone=FakeEEGMamba(),
        stft_config=ServerSTFTConfig(
            n_fft=64,
            win_length=64,
            hop_length=32,
        ),
        checkpoint_sha256="test-checkpoint",
    )
    waveform = torch.randn(3, 1, 18, 1024) * 0.01
    baseline = torch.zeros(3, 20, 33)

    logits = model(waveform, baseline_log_magnitude=baseline)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    contract = model.contract()
    assert contract["e2"]["features"] == 120
    assert contract["input"]["output_shape"] == [18, 4, 200]


def test_mamba_b_requires_patient_baseline() -> None:
    model = EEGMambaBE2Classifier(backbone=FakeEEGMamba())

    with pytest.raises(ValueError, match="baseline_log_magnitude"):
        model(torch.zeros(1, 1, 18, 1024))
