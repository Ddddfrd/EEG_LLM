from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ai.chbmit.eeg_continual_pretrain_model import (
    Canonical18ToServer20,
    ServerSTFTConfig,
)
from ai.chbmit.eegvl_m9_model import LoRAConfig, LoRALinear
from ai.chbmit.eegvl_multibranch_experiment import (
    MultibranchTrainingConfig,
    _optimizer_groups,
)
from ai.chbmit.eegvl_multibranch_model import (
    E3E4PhysiologyEncoder,
    EEGVLE1E2E3E4Classifier,
)


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 896)
        self.q_proj = nn.Linear(896, 896)
        self.v_proj = nn.Linear(896, 896)
        self.config = SimpleNamespace(use_cache=True)
        self.last_input_shape: tuple[int, ...] | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> object:
        self.last_input_shape = tuple(inputs_embeds.shape)
        hidden = 0.5 * (
            self.q_proj(inputs_embeds) + self.v_proj(inputs_embeds)
        )
        return SimpleNamespace(last_hidden_state=hidden)


class FakeVisualEncoder(nn.Module):
    output_dim = 1280

    def __init__(self, *, zscore_input: bool = False) -> None:
        super().__init__()
        self.config = ServerSTFTConfig(
            n_fft=256,
            win_length=128,
            hop_length=32,
            zscore_input=zscore_input,
        )
        self.channel_adapter = Canonical18ToServer20()
        self.features = nn.Sequential(nn.Linear(1, 2))

    def log_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            waveform.shape[0],
            20,
            129,
            33,
            device=waveform.device,
        )

    def forward_log_magnitude(self, log_magnitude: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            log_magnitude.shape[0],
            32,
            1280,
            device=log_magnitude.device,
        )


def build_model() -> tuple[EEGVLE1E2E3E4Classifier, FakeLanguageModel]:
    language = FakeLanguageModel()
    model = EEGVLE1E2E3E4Classifier(
        language_model=language,
        prompt_input_ids=torch.zeros(1, 35, dtype=torch.long),
        visual_encoder=FakeVisualEncoder(),  # type: ignore[arg-type]
        lora_config=LoRAConfig(target_modules=("q_proj", "v_proj")),
        qwen_model_name="fake-qwen",
        qwen_revision="test",
    )
    return model, language


def test_e3_e4_feature_shapes_and_finite_values() -> None:
    encoder = E3E4PhysiologyEncoder(
        config=ServerSTFTConfig(
            n_fft=256,
            win_length=128,
            hop_length=32,
            zscore_input=False,
        )
    )
    waveform = torch.randn(3, 20, 1024) * 50.0

    e3, e4 = encoder(waveform)

    assert e3.shape == (3, 40)
    assert e4.shape == (3, 80)
    assert torch.isfinite(e3).all()
    assert torch.isfinite(e4).all()


def test_e3_e4_are_invariant_to_positive_amplitude_gain() -> None:
    encoder = E3E4PhysiologyEncoder(config=ServerSTFTConfig())
    waveform = torch.randn(2, 20, 1024) * 25.0

    original = encoder(waveform)
    amplified = encoder(waveform * 7.0)

    torch.testing.assert_close(original[0], amplified[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original[1], amplified[1], rtol=2e-5, atol=2e-6)


def test_multibranch_forward_uses_67_qwen_tokens_and_zero_head() -> None:
    model, language = build_model()
    waveform = torch.randn(2, 1, 18, 1024) * 0.01
    baseline = torch.zeros(2, 20, 129)

    logits = model(waveform, baseline_log_magnitude=baseline)

    assert logits.shape == (2, 2)
    assert language.last_input_shape == (2, 67, 896)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert isinstance(language.q_proj, LoRALinear)
    assert isinstance(language.v_proj, LoRALinear)


def test_multibranch_rejects_per_window_zscore() -> None:
    with pytest.raises(ValueError, match="zscore_input=False"):
        EEGVLE1E2E3E4Classifier(
            language_model=FakeLanguageModel(),
            prompt_input_ids=torch.zeros(1, 35, dtype=torch.long),
            visual_encoder=FakeVisualEncoder(zscore_input=True),  # type: ignore[arg-type]
            lora_config=LoRAConfig(target_modules=("q_proj", "v_proj")),
            qwen_model_name="fake-qwen",
            qwen_revision=None,
        )


def test_multibranch_optimizer_uses_independent_e2_learning_rate() -> None:
    model, _ = build_model()
    config = MultibranchTrainingConfig()

    groups = _optimizer_groups(model, config=config)
    learning_rates = {
        str(group["group_name"]): float(group["lr"])
        for group in groups
    }

    assert learning_rates == {
        "efficientnet": 1e-4,
        "head": 1e-4,
        "lora": 2e-5,
        "e2": 5e-3,
        "e3": 1e-4,
        "e4": 1e-4,
    }


def test_multibranch_contract_declares_no_zscore_or_gate() -> None:
    model, _ = build_model()

    contract = model.contract()

    assert contract["input"]["model_shape_after_channel_adapter"] == [20, 1024]
    assert contract["input"]["per_window_channel_zscore"] is False
    assert contract["e2"]["features"] == 120
    assert contract["e3"]["features"] == 40
    assert contract["e4"]["features"] == 80
    assert contract["gate_fusion"] is False
