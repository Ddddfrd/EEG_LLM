from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from ai.chbmit.eegvl_m9_model import LoRAConfig, LoRALinear
from good.e1_e2_e3_e4_modernbert_base.model import (
    DEFAULT_MODERNBERT_MODEL,
    ModernBERTBackboneAdapter,
    _inject_attention_output_lora,
    build_model,
)


from good.e1_e2_e3_e4_modernbert_base import (
    train_visual_mean_stft_s1 as s1_module,
)

class FakeModernBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 768)
        self.config = SimpleNamespace()
        self.received_kwargs: dict[str, object] = {}

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def gradient_checkpointing_enable(self, **_: object) -> None:
        return None

    def forward(self, **kwargs: object) -> object:
        self.received_kwargs = kwargs
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


def test_modernbert_model_identity_is_official_checkpoint() -> None:
    assert DEFAULT_MODERNBERT_MODEL == "answerdotai/ModernBERT-base"


def test_modernbert_builder_exposes_pooling_selection() -> None:
    parameter = inspect.signature(build_model).parameters["pooling"]

    assert parameter.default == "mean"


def test_adapter_drops_causal_use_cache_argument() -> None:
    backbone = FakeModernBERT()
    adapter = ModernBERTBackboneAdapter(backbone)
    embeddings = torch.randn(2, 67, 768)
    mask = torch.ones(2, 67, dtype=torch.long)

    output = adapter(
        inputs_embeds=embeddings,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )

    assert output.last_hidden_state is embeddings
    assert "use_cache" not in backbone.received_kwargs
    assert backbone.received_kwargs["attention_mask"] is mask


class FakeModernBERTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.Wo = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.Wo = nn.Linear(8, 8, bias=False)


def test_attention_output_lora_does_not_wrap_mlp_wo() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([FakeModernBERTBlock(), FakeModernBERTBlock()])
    config = LoRAConfig(rank=2, alpha=4.0, dropout=0.0, target_modules=("attn.Wo",))

    names = _inject_attention_output_lora(model, config=config)

    assert names == ["layers.0.attn.Wo", "layers.1.attn.Wo"]
    assert all(isinstance(block.attn.Wo, LoRALinear) for block in model.layers)
    assert all(isinstance(block.mlp.Wo, nn.Linear) for block in model.layers)


def test_visual_mean_s1_uses_matched_stft_and_training_defaults(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(s1_module, "run_experiment", fake_run_experiment)

    exit_code = s1_module.main(
        [
            "--reference-artifact",
            str(tmp_path / "reference.json"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--shared-cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    config = captured["config"]
    stft = captured["stft_config_override"]
    builder = captured["model_builder"]
    assert exit_code == 0
    assert builder.keywords["pooling"] == "visual_mean"
    assert (stft.n_fft, stft.win_length, stft.hop_length) == (128, 128, 32)
    assert stft.zscore_input is False
    assert config.max_epochs == 5
    assert config.micro_batch_size == 8
    assert config.effective_batch_size == 32
