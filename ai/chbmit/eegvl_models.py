"""EEG-VL model controls used by the CHB-MIT S0 experiment."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol

import torch
import torch.nn as nn


EEGVL_MODEL_VERSION = "eegvl_18_s0_v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_TASK_PROMPT = (
    "You are an AI system for EEG signal analysis. Analyze CNN-processed EEG "
    "features and classify the signal as Normal or Seizure."
)


class VisualTokenEncoder(Protocol):
    output_dim: int

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor: ...


def _require_torchvision() -> Any:
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise RuntimeError(
            "EEG-VL EfficientNet requires torchvision in the experiment environment"
        ) from exc
    return EfficientNet_B0_Weights, efficientnet_b0


def _single_channel_first_conv(model: nn.Module) -> None:
    first = model.features[0][0]
    if not isinstance(first, nn.Conv2d) or first.in_channels != 3:
        raise ValueError("Unexpected torchvision EfficientNet-B0 input convolution")
    replacement = nn.Conv2d(
        1,
        first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=first.bias is not None,
        padding_mode=first.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(first.weight.sum(dim=1, keepdim=True))
        if first.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(first.bias)
    model.features[0][0] = replacement


class EfficientNetVisualEncoder(nn.Module):
    output_dim = 1280

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights_type, factory = _require_torchvision()
        weights = weights_type.DEFAULT if pretrained else None
        backbone = factory(weights=weights)
        _single_channel_first_conv(backbone)
        self.features = backbone.features

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 4 or waveform.shape[1] != 1:
            raise ValueError("waveform must be shaped (batch, 1, channels, time)")
        feature_map = self.features(waveform)
        if feature_map.ndim != 4 or feature_map.shape[1] != self.output_dim:
            raise ValueError(
                "Unexpected EfficientNet feature map: "
                f"{tuple(feature_map.shape)}"
            )
        return feature_map.flatten(2).transpose(1, 2).contiguous()


class EfficientNetLinearClassifier(nn.Module):
    model_name = "m3_efficientnet_linear"

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder or EfficientNetVisualEncoder(pretrained=pretrained)
        output_dim = int(getattr(self.encoder, "output_dim", 1280))
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 2),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.encoder(waveform)
        return self.classifier(visual_tokens.mean(dim=1))


class EfficientNetTransformerClassifier(nn.Module):
    model_name = "m4_efficientnet_transformer"

    def __init__(
        self,
        *,
        encoder: nn.Module | None = None,
        pretrained: bool = True,
        hidden_size: int = 896,
        attention_heads: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder or EfficientNetVisualEncoder(pretrained=pretrained)
        output_dim = int(getattr(self.encoder, "output_dim", 1280))
        self.projector = nn.Linear(output_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(self.projector(self.encoder(waveform)))
        weights = torch.softmax(self.attention_score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        return self.classifier(pooled)


class FrozenQwenVisualClassifier(nn.Module):
    model_name = "m5_efficientnet_frozen_qwen"

    def __init__(
        self,
        *,
        language_model: nn.Module,
        prompt_input_ids: torch.Tensor,
        encoder: nn.Module | None = None,
        pretrained: bool = True,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("prompt_input_ids must be shaped (1, prompt_tokens)")
        self.encoder = encoder or EfficientNetVisualEncoder(pretrained=pretrained)
        self.language_model = language_model
        inferred_hidden = int(language_model.get_input_embeddings().embedding_dim)
        if hidden_size is not None and hidden_size != inferred_hidden:
            raise ValueError("Configured hidden size does not match language model")
        output_dim = int(getattr(self.encoder, "output_dim", 1280))
        self.projector = nn.Linear(output_dim, inferred_hidden)
        self.classifier = nn.Linear(inferred_hidden, 2)
        self.register_buffer(
            "prompt_input_ids",
            prompt_input_ids.to(dtype=torch.long),
            persistent=True,
        )
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        self.language_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        prompt: str = DEFAULT_TASK_PROMPT,
        encoder: nn.Module | None = None,
        pretrained_encoder: bool = True,
        local_files_only: bool = False,
    ) -> "FrozenQwenVisualClassifier":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "M5 requires transformers in the experiment environment"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        language_model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        return cls(
            language_model=language_model,
            prompt_input_ids=prompt_ids,
            encoder=encoder,
            pretrained=pretrained_encoder,
        )

    def train(self, mode: bool = True) -> "FrozenQwenVisualClassifier":
        super().train(mode)
        self.language_model.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        visual_embeddings = self.projector(self.encoder(waveform))
        batch = visual_embeddings.shape[0]
        prompt_ids = self.prompt_input_ids.expand(batch, -1)
        prompt_embeddings = self.language_model.get_input_embeddings()(prompt_ids)
        combined = torch.cat((prompt_embeddings, visual_embeddings), dim=1)
        attention_mask = torch.ones(
            combined.shape[:2],
            dtype=torch.long,
            device=combined.device,
        )
        output = self.language_model(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return self.classifier(output.last_hidden_state[:, -1])


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def smoke_language_model(
    *,
    vocabulary_size: int = 32,
    hidden_size: int = 896,
) -> nn.Module:
    """Small injectable language model used only by unit tests."""

    class _SmokeLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, hidden_size)
            self.projection = nn.Linear(hidden_size, hidden_size)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embedding

        def forward(self, *, inputs_embeds: torch.Tensor, **_: Any) -> Any:
            return SimpleNamespace(
                last_hidden_state=self.projection(inputs_embeds)
            )

    return _SmokeLanguageModel()
