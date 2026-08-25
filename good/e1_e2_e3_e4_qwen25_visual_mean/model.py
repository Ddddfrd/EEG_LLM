"""Fixed factory for the promoted Qwen2.5 visual_mean classifier."""

from __future__ import annotations

from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegvl_models import DEFAULT_QWEN_MODEL
from ai.chbmit.eegvl_multibranch_model import EEGVLE1E2E3E4Classifier
from good.e1_e2_e3_e4_fullband.model import build_model as _build_fullband_model


MODEL_LABEL = "Qwen2.5-0.5B E1+E2+E3+E4 visual_mean"
POOLING = "visual_mean"
RESULT_REPORT = (
    "artifacts/chbmit/scheme_c_qwen25_05b_pooling_ablation/"
    "visual_mean/SCHEME_C_EEGMAMBA_SPLIT_RESULTS.md"
)


def stft_config() -> ServerSTFTConfig:
    """Return the STFT contract used by the promoted leaderboard artifact."""
    return ServerSTFTConfig(
        source_channels=20,
        eeg_channels=20,
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
    stft_config_override: ServerSTFTConfig | None = None,
) -> EEGVLE1E2E3E4Classifier:
    """Build Scheme C with pooling fixed to the 32 visual hidden states."""
    return _build_fullband_model(
        qwen_model_name=qwen_model_name,
        local_files_only=local_files_only,
        pretrained_visual_encoder=pretrained_visual_encoder,
        stft_config_override=stft_config_override or stft_config(),
        pooling=POOLING,
    )


__all__ = [
    "MODEL_LABEL",
    "POOLING",
    "RESULT_REPORT",
    "build_model",
    "stft_config",
]
