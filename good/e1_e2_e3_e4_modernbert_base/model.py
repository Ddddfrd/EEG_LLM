"""ModernBERT-base adapter for the retained Scheme C EEG classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ai.chbmit.eeg_continual_pretrain_model import (
    SERVER_TASK_PROMPT,
    ServerSTFTConfig,
    ServerSTFTEfficientNetEncoder,
    _fixed_prompt_ids,
)
from ai.chbmit.eegvl_m9_model import LoRAConfig, LoRALinear
from ai.chbmit.eegvl_multibranch_model import EEGVLE1E2E3E4Classifier


DEFAULT_MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
MODERNBERT_SCHEME_C_VERSION = "scheme_c_modernbert_base_v1"


class ModernBERTBackboneAdapter(nn.Module):
    """Expose ModernBERT through the causal-backbone interface used by Scheme C."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        del use_cache
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )


class ModernBERTSchemeCClassifier(EEGVLE1E2E3E4Classifier):
    """Scheme C with bidirectional ModernBERT token interaction."""

    model_name = "scheme_c_modernbert_base_classifier"
    model_version = MODERNBERT_SCHEME_C_VERSION

    def contract(self) -> dict[str, Any]:
        contract = super().contract()
        language = contract.pop("qwen")
        language.update(
            {
                "architecture": "ModernBERT bidirectional encoder",
                "causal_attention": False,
                "attention_implementation": "sdpa",
            }
        )
        parameters = dict(contract["parameters"])
        parameters["language_lora"] = parameters.pop("qwen_lora")
        contract.update(
            {
                "architecture": (
                    "STFT EfficientNet + ModernBERT Wqkv/Wo LoRA + "
                    "E2/E3/E4 residuals"
                ),
                "language_backbone": language,
                "fusion": (
                    f"tanh(LayerNorm(ModernBERT {self.pooling})) + "
                    "e2_proj + e3_proj + e4_proj"
                ),
                "parameters": parameters,
            }
        )
        return contract


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("ModernBERT Scheme C requires transformers") from exc
    return AutoModel, AutoTokenizer


def _inject_attention_output_lora(
    model: nn.Module,
    *,
    config: LoRAConfig,
) -> list[str]:
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(".attn.Wo")
    ]
    if not replacements:
        raise ValueError("No ModernBERT attention output projections found for LoRA")
    names: list[str] = []
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
        names.append(name)
    return names

def build_model(
    *,
    qwen_model_name: str = DEFAULT_MODERNBERT_MODEL,
    local_files_only: bool = True,
    pretrained_visual_encoder: bool = True,
    stft_config_override: ServerSTFTConfig | None = None,
    pooling: str = "mean",
) -> ModernBERTSchemeCClassifier:
    """Build Scheme C with the official ModernBERT-base checkpoint."""
    auto_model, auto_tokenizer = _require_transformers()
    tokenizer = auto_tokenizer.from_pretrained(
        qwen_model_name,
        local_files_only=local_files_only,
    )
    backbone = auto_model.from_pretrained(
        qwen_model_name,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
    )
    revision = getattr(backbone.config, "_commit_hash", None)
    prompt_ids = _fixed_prompt_ids(tokenizer, SERVER_TASK_PROMPT, token_count=35)
    config = stft_config_override or ServerSTFTConfig(
        n_fft=64,
        win_length=64,
        hop_length=32,
        zscore_input=False,
    )
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.05,
        target_modules=("Wqkv", "attn.Wo"),
    )
    model = ModernBERTSchemeCClassifier(
        language_model=ModernBERTBackboneAdapter(backbone),
        prompt_input_ids=prompt_ids,
        visual_encoder=ServerSTFTEfficientNetEncoder(
            pretrained=pretrained_visual_encoder,
            config=config,
        ),
        lora_config=lora_config,

        qwen_model_name=qwen_model_name,
        qwen_revision=revision,
        pooling=pooling,
    )
    if model.hidden_size != 768:
        raise ValueError(f"Expected ModernBERT-base hidden size 768, got {model.hidden_size}")
    model.lora_module_names.extend(
        _inject_attention_output_lora(model.language_model, config=lora_config)
    )
    if len(model.lora_module_names) != 44:
        raise ValueError(
            f"Expected 44 ModernBERT attention LoRA modules, got "
            f"{len(model.lora_module_names)}"
        )
    return model
