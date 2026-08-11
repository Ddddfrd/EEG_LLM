"""EEGMamba visual tokens fused with Qwen LoRA and patient-relative E2.

Mamba-C keeps all 18 channel by 4 temporal EEGMamba tokens. Each 200-D
token is projected directly into Qwen's 896-D embedding space; no adaptive
token pooling is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .eeg_continual_pretrain_model import (
    E2_FREQUENCY_BANDS_HZ,
    PatientRelativeSpectrumEncoder,
    SERVER_TASK_PROMPT,
    ServerSTFTConfig,
    _fixed_prompt_ids,
    _require_transformers,
)
from .eegmamba_b import (
    EEGMambaInputConfig,
    E2LogMagnitudeFrontEnd,
    file_sha256,
    load_official_eegmamba_backbone,
    waveform_to_eegmamba_patches,
)
from .eegvl_m9_model import LoRAConfig, LoRALinear, inject_lora
from .eegvl_models import DEFAULT_QWEN_MODEL


EEGMAMBA_C_MODEL_VERSION = "eegmamba_c_qwen_lora_e2_v1"


class EEGMambaCQwenE2Classifier(nn.Module):
    """Direct 72-token EEGMamba to Qwen-LoRA classifier with E2 bypass."""

    model_name = "eegmamba_c_qwen_lora_e2_classifier"
    model_version = EEGMAMBA_C_MODEL_VERSION

    def __init__(
        self,
        *,
        backbone: nn.Module,
        language_model: nn.Module,
        prompt_input_ids: torch.Tensor,
        lora_config: LoRAConfig,
        qwen_model_name: str,
        qwen_revision: str | None,
        input_config: EEGMambaInputConfig | None = None,
        stft_config: ServerSTFTConfig | None = None,
        checkpoint_sha256: str | None = None,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be mean or last")
        if prompt_input_ids.shape != (1, 35):
            raise ValueError("Mamba-C prompt must contain exactly 35 tokens")

        self.input_config = input_config or EEGMambaInputConfig()
        self.input_config.validate()
        self.stft_config = stft_config or ServerSTFTConfig()
        self.stft_config.validate()
        self.backbone = backbone
        self.language_model = language_model
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        self.lora_module_names = inject_lora(
            self.language_model,
            config=lora_config,
        )

        hidden_size = int(
            self.language_model.get_input_embeddings().embedding_dim
        )
        if hidden_size != 896:
            raise ValueError(f"Expected Qwen hidden size 896, got {hidden_size}")
        with torch.no_grad():
            prompt_seed = self.language_model.get_input_embeddings()(
                prompt_input_ids
            ).detach().clone()
        self.prompt_embeddings = nn.Parameter(prompt_seed)

        self.channel_embeddings = nn.Parameter(
            torch.zeros(self.input_config.channels, 200)
        )
        self.patch_embeddings = nn.Parameter(
            torch.zeros(self.input_config.patch_count, 200)
        )
        nn.init.normal_(self.channel_embeddings, std=0.02)
        nn.init.normal_(self.patch_embeddings, std=0.02)
        self.visual_projection = nn.Linear(200, hidden_size)

        self.e2_frontend = E2LogMagnitudeFrontEnd(config=self.stft_config)
        self.e2_encoder = PatientRelativeSpectrumEncoder(config=self.stft_config)
        self.e2_projection = nn.Sequential(
            nn.LayerNorm(PatientRelativeSpectrumEncoder.output_features),
            nn.Linear(PatientRelativeSpectrumEncoder.output_features, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )
        self.head_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.pooling = pooling
        self.lora_config = lora_config
        self.qwen_model_name = qwen_model_name
        self.qwen_revision = qwen_revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.register_buffer(
            "prompt_input_ids",
            prompt_input_ids.to(dtype=torch.long),
            persistent=True,
        )
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model, "config"):
            self.language_model.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        official_checkpoint_path: Path,
        qwen_model_name: str = DEFAULT_QWEN_MODEL,
        prompt: str = SERVER_TASK_PROMPT,
        local_files_only: bool = True,
        lora_config: LoRAConfig | None = None,
        input_config: EEGMambaInputConfig | None = None,
        stft_config: ServerSTFTConfig | None = None,
        pooling: str = "mean",
    ) -> "EEGMambaCQwenE2Classifier":
        auto_model, auto_tokenizer = _require_transformers()
        tokenizer = auto_tokenizer.from_pretrained(
            qwen_model_name,
            local_files_only=local_files_only,
        )
        language_model = auto_model.from_pretrained(
            qwen_model_name,
            local_files_only=local_files_only,
        )
        prompt_ids = _fixed_prompt_ids(tokenizer, prompt, token_count=35)
        revision = getattr(language_model.config, "_commit_hash", None)
        return cls(
            backbone=load_official_eegmamba_backbone(official_checkpoint_path),
            language_model=language_model,
            prompt_input_ids=prompt_ids,
            lora_config=lora_config
            or LoRAConfig(target_modules=("q_proj", "v_proj")),
            qwen_model_name=qwen_model_name,
            qwen_revision=revision,
            input_config=input_config,
            stft_config=stft_config,
            checkpoint_sha256=file_sha256(official_checkpoint_path),
            pooling=pooling,
        )

    def train(self, mode: bool = True) -> "EEGMambaCQwenE2Classifier":
        super().train(mode)
        self.language_model.eval()
        for module in self.language_model.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def _visual_tokens(self, waveform: torch.Tensor) -> torch.Tensor:
        patches = waveform_to_eegmamba_patches(
            waveform,
            config=self.input_config,
        )
        representation = self.backbone(patches)
        expected = (
            waveform.shape[0],
            self.input_config.channels,
            self.input_config.patch_count,
            200,
        )
        if tuple(representation.shape) != expected:
            raise ValueError(
                f"EEGMamba backbone output must be shaped {expected}, "
                f"got {tuple(representation.shape)}"
            )
        positioned = (
            representation
            + self.channel_embeddings[None, :, None, :]
            + self.patch_embeddings[None, None, :, :]
        )
        return self.visual_projection(positioned.flatten(1, 2))

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        baseline_log_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if baseline_log_magnitude is None:
            raise ValueError("Mamba-C requires baseline_log_magnitude for E2")
        visual_tokens = self._visual_tokens(waveform)
        prompt = self.prompt_embeddings.expand(waveform.shape[0], -1, -1)
        multimodal = torch.cat([prompt, visual_tokens], dim=1)
        expected_tokens = 35 + self.input_config.channels * self.input_config.patch_count
        if multimodal.shape[1] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} multimodal tokens")
        attention_mask = torch.ones(
            multimodal.shape[:2],
            dtype=torch.long,
            device=multimodal.device,
        )
        outputs = self.language_model(
            inputs_embeds=multimodal,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        summary = hidden.mean(dim=1) if self.pooling == "mean" else hidden[:, -1]

        log_magnitude = self.e2_frontend.log_magnitude(waveform)
        e2_features = self.e2_encoder(
            log_magnitude,
            baseline_log_magnitude.float(),
        )
        summary = summary + self.e2_projection(e2_features)
        return self.classifier(torch.tanh(self.head_norm(summary)))

    def parameter_summary(self) -> dict[str, int]:
        named = list(self.named_parameters())
        return {
            "total": sum(parameter.numel() for _, parameter in named),
            "trainable": sum(
                parameter.numel()
                for _, parameter in named
                if parameter.requires_grad
            ),
            "eegmamba": sum(
                parameter.numel()
                for name, parameter in named
                if name.startswith("backbone.")
            ),
            "qwen_lora": sum(
                parameter.numel()
                for name, parameter in named
                if name.startswith("language_model.") and parameter.requires_grad
            ),
        }

    def contract(self) -> dict[str, Any]:
        visual_tokens = self.input_config.channels * self.input_config.patch_count
        return {
            "version": self.model_version,
            "architecture": "EEGMamba 72 tokens -> Qwen LoRA + E2 residual",
            "official_eegmamba_repository": "https://github.com/wjq-learning/EEGMamba",
            "official_checkpoint_sha256": self.checkpoint_sha256,
            "input": self.input_config.to_dict(),
            "eegmamba": {
                "d_model": 200,
                "layers": 12,
                "visual_tokens": visual_tokens,
                "token_pooling": "none",
                "token_order": "channel_major_then_patch",
                "channel_position_embedding": True,
                "patch_position_embedding": True,
                "fine_tuning": "all_backbone_parameters_trainable",
            },
            "qwen": {
                "model_name": self.qwen_model_name,
                "revision": self.qwen_revision,
                "hidden_size": 896,
                "prompt_tokens": 35,
                "visual_projection": "Linear(200,896)",
                "sequence_tokens": 35 + visual_tokens,
                "pooling": self.pooling,
            },
            "lora": {
                **self.lora_config.to_dict(),
                "injected_modules": list(self.lora_module_names),
            },
            "e2": {
                "features": PatientRelativeSpectrumEncoder.output_features,
                "bands_hz": [list(band) for band in E2_FREQUENCY_BANDS_HZ],
                "definition": "mean(log1p magnitude - patient baseline)",
                "stft": self.stft_config.to_dict(),
                "fusion": "120 -> 256 -> 896 additive residual",
            },
            "head": "LayerNorm -> tanh -> zero-initialized Linear(896,2)",
            "parameters": self.parameter_summary(),
        }


def portable_mamba_c_state_dict(
    model: EEGMambaCQwenE2Classifier,
) -> dict[str, torch.Tensor]:
    """Save all task weights while excluding the frozen Qwen base model."""
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if (
            not name.startswith("language_model.")
            or name.endswith(".lora_a")
            or name.endswith(".lora_b")
        ):
            state[name] = value.detach().cpu().clone()
    return state


def load_portable_mamba_c_state_dict(
    model: EEGMambaCQwenE2Classifier,
    state: Mapping[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(dict(state), strict=False)
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if (
            not name.startswith("language_model.")
            or name.endswith(".lora_a")
            or name.endswith(".lora_b")
        )
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise ValueError(
            "Portable Mamba-C state is incompatible: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
