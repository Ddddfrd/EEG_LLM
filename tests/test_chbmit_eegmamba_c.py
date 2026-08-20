from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ai.chbmit.eeg_continual_pretrain_model import ServerSTFTConfig
from ai.chbmit.eegmamba_c import (
    EEGMambaCQwenE2Classifier,
    portable_mamba_c_state_dict,
)
from ai.chbmit.eegvl_m9_model import LoRAConfig, LoRALinear


class FakeEEGMamba(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 200)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.projection(patches.mean(dim=-1, keepdim=True))


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


def build_model() -> tuple[EEGMambaCQwenE2Classifier, FakeLanguageModel]:
    language = FakeLanguageModel()
    model = EEGMambaCQwenE2Classifier(
        backbone=FakeEEGMamba(),
        language_model=language,
        prompt_input_ids=torch.zeros(1, 35, dtype=torch.long),
        lora_config=LoRAConfig(target_modules=("q_proj", "v_proj")),
        qwen_model_name="fake-qwen",
        qwen_revision="test",
        stft_config=ServerSTFTConfig(
            n_fft=64,
            win_length=64,
            hop_length=32,
        ),
        checkpoint_sha256="test-checkpoint",
    )
    return model, language


def test_mamba_c_passes_all_72_visual_tokens_to_qwen() -> None:
    model, language = build_model()
    waveform = torch.randn(2, 1, 18, 1024) * 0.01
    baseline = torch.zeros(2, 20, 33)

    logits = model(waveform, baseline_log_magnitude=baseline)

    assert logits.shape == (2, 2)
    assert language.last_input_shape == (2, 107, 896)
    assert isinstance(language.q_proj, LoRALinear)
    assert isinstance(language.v_proj, LoRALinear)
    assert torch.equal(logits, torch.zeros_like(logits))


def test_mamba_c_contract_declares_direct_token_projection() -> None:
    model, _ = build_model()

    contract = model.contract()

    assert contract["eegmamba"]["visual_tokens"] == 72
    assert contract["eegmamba"]["token_pooling"] == "none"
    assert contract["qwen"]["visual_projection"] == "Linear(200,896)"
    assert contract["qwen"]["sequence_tokens"] == 107
    assert contract["e2"]["features"] == 120


def test_mamba_c_requires_patient_baseline() -> None:
    model, _ = build_model()

    with pytest.raises(ValueError, match="baseline_log_magnitude"):
        model(torch.zeros(1, 1, 18, 1024))


def test_mamba_c_portable_state_excludes_frozen_qwen_base() -> None:
    model, _ = build_model()

    state = portable_mamba_c_state_dict(model)

    assert "language_model.embedding.weight" not in state
    assert "language_model.q_proj.base.weight" not in state
    assert "language_model.q_proj.lora_a" in state
    assert "language_model.q_proj.lora_b" in state
    assert "backbone.projection.weight" in state


def test_mamba_c_freezes_qwen_base_but_trains_lora_and_backbone() -> None:
    model, language = build_model()

    assert not language.embedding.weight.requires_grad
    assert not language.q_proj.base_layer.weight.requires_grad
    assert language.q_proj.lora_a.requires_grad
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
