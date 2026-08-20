"""Exact factory for the retained E1+E2 STFT-64 architecture."""

from __future__ import annotations

from ai.chbmit.eeg_continual_pretrain_model import (
    PatientRelativeSpectrumEncoder,
    ServerEEGVLPretrainModel,
    ServerSTFTConfig,
)
from ai.chbmit.eegvl_m9_model import LoRAConfig
from ai.chbmit.eegvl_models import DEFAULT_QWEN_MODEL


MODEL_LABEL = "E1+E2 STFT-64"
MODEL_VERSION = "eeg_continual_pretrain_stft_qwen_v1"
HISTORICAL_RESULT = (
    "artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/"
    "fold0_pretrain_c27817a49668.json"
)
HISTORICAL_CHECKPOINT = (
    "artifacts/chbmit/eeg_continual_pretrain_strict_e2_smoke/checkpoints/"
    "fold0_lora_stft_best.pt"
)
HISTORICAL_CHECKPOINT_SHA256 = (
    "c7c0683738d66a8476b17c642ff380078e78306665be4d54c180ee9cc1a48bde"
)


def stft_config() -> ServerSTFTConfig:
    """Return the immutable signal contract used by the retained checkpoint."""
    return ServerSTFTConfig(
        n_fft=64,
        win_length=64,
        hop_length=32,
        zscore_input=False,
    )


def build_model(
    *,
    qwen_model_name: str = DEFAULT_QWEN_MODEL,
    local_files_only: bool = True,
    pretrained_visual_encoder: bool = True,
) -> ServerEEGVLPretrainModel:
    """Build E1 visual residual + E2 patient-relative spectral residual."""
    return ServerEEGVLPretrainModel.from_pretrained(
        qwen_model_name=qwen_model_name,
        local_files_only=local_files_only,
        pretrained_visual_encoder=pretrained_visual_encoder,
        stft_config=stft_config(),
        lora_config=LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
        pooling="mean",
        visual_bypass=True,
        relative_spectral_bypass=True,
    )


__all__ = [
    "HISTORICAL_CHECKPOINT",
    "HISTORICAL_CHECKPOINT_SHA256",
    "HISTORICAL_RESULT",
    "MODEL_LABEL",
    "MODEL_VERSION",
    "PatientRelativeSpectrumEncoder",
    "ServerEEGVLPretrainModel",
    "ServerSTFTConfig",
    "build_model",
    "stft_config",
]

