from good.e1_e2_e3_e4_fullband.model import (
    HISTORICAL_CHECKPOINT_SHA256 as FULLBAND_SHA256,
)
from good.e1_e2_e3_e4_fullband.model import stft_config as fullband_stft_config
from good.e1_e2_stft64.model import (
    HISTORICAL_CHECKPOINT_SHA256 as STFT64_SHA256,
)
from good.e1_e2_stft64.model import stft_config as stft64_config


def test_retained_stft64_contract() -> None:
    config = stft64_config()

    assert config.source_channels == 18
    assert config.eeg_channels == 20
    assert config.n_fft == 64
    assert config.win_length == 64
    assert config.hop_length == 32
    assert config.zscore_input is False
    assert len(STFT64_SHA256) == 64


def test_retained_fullband_contract() -> None:
    config = fullband_stft_config()

    assert config.source_channels == 18
    assert config.eeg_channels == 20
    assert config.n_fft == 256
    assert config.win_length == 128
    assert config.hop_length == 32
    assert config.zscore_input is False
    assert len(FULLBAND_SHA256) == 64
