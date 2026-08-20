"""Exact factory for the retained E1+E2+E3+E4 full-band architecture."""

from __future__ import annotations

from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegvl_m9_model import LoRAConfig
from ai.chbmit.eegvl_models import DEFAULT_QWEN_MODEL
from ai.chbmit.eegvl_multibranch_model import (
    E3E4PhysiologyEncoder,
    EEGVLE1E2E3E4Classifier,
    load_portable_multibranch_state_dict,
    portable_multibranch_state_dict,
)


MODEL_LABEL = "E1+E2+E3+E4 full-band direct residual"
MODEL_VERSION = "eegvl_e1_e2_e3_e4_residual_v1"
HISTORICAL_RESULT = (
    "artifacts/chbmit/eegvl_multibranch_fullband/"
    "fold0_e1_e2_e3_e4_f1660457394b.json"
)
HISTORICAL_CHECKPOINT = (
    "artifacts/chbmit/eegvl_multibranch_fullband/checkpoints/"
    "fold0_e1_e2_e3_e4_best.pt"
)
HISTORICAL_CHECKPOINT_SHA256 = (
    "52d85560992237d661270e4ebd3a3db83391b8d288248f8971269e127c3a1873"
)


def stft_config() -> ServerSTFTConfig:
    """Return the full-band STFT contract used by the retained checkpoint."""
    return ServerSTFTConfig(
        n_fft=256,
        win_length=128,
        hop_length=32,
        zscore_input=False,
    )


def build_model(
    *,
    qwen_model_name: str = DEFAULT_QWEN_MODEL,
    local_files_only: bool = True,
    pretrained_visual_encoder: bool = True,
) -> EEGVLE1E2E3E4Classifier:
    """Build the ungated E1/E2/E3/E4 additive residual classifier."""
    return EEGVLE1E2E3E4Classifier.from_pretrained(
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
    )


__all__ = [
    "E3E4PhysiologyEncoder",
    "EEGVLE1E2E3E4Classifier",
    "HISTORICAL_CHECKPOINT",
    "HISTORICAL_CHECKPOINT_SHA256",
    "HISTORICAL_RESULT",
    "MODEL_LABEL",
    "MODEL_VERSION",
    "build_model",
    "load_portable_multibranch_state_dict",
    "portable_multibranch_state_dict",
    "stft_config",
]

