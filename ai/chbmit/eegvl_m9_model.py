"""M9 STFT-EfficientNet-Qwen model with local LoRA and residual fusion."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from .eegvl_models import DEFAULT_QWEN_MODEL, DEFAULT_TASK_PROMPT, _require_torchvision


M9_MODEL_VERSION = "eegvl_m9_stft_qwen_lora_residual_v1"


@dataclass(frozen=True)
class M9STFTConfig:
    sampling_frequency_hz: int = 256
    n_fft: int = 256
    win_length: int = 128
    hop_length: int = 32
    low_frequency_hz: float = 1.0
    high_frequency_hz: float = 45.0
    eeg_channels: int = 18
    grid_rows: int = 6
    grid_columns: int = 3
    image_size: int = 160
    normalization_clip: float = 5.0

    def validate(self) -> None:
        if self.sampling_frequency_hz < 1:
            raise ValueError("sampling_frequency_hz must be positive")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be inside n_fft")
        if self.hop_length < 1:
            raise ValueError("hop_length must be positive")
        nyquist = self.sampling_frequency_hz / 2.0
        if not 0 <= self.low_frequency_hz < self.high_frequency_hz <= nyquist:
            raise ValueError("Invalid STFT frequency range")
        if self.grid_rows * self.grid_columns != self.eeg_channels:
            raise ValueError("STFT grid must contain every EEG channel")
        if self.image_size < 32 or self.normalization_clip <= 0:
            raise ValueError("Invalid STFT image settings")

    @property
    def first_frequency_bin(self) -> int:
        return int(round(
            self.low_frequency_hz
            * self.n_fft
            / self.sampling_frequency_hz
        ))

    @property
    def last_frequency_bin(self) -> int:
        return int(round(
            self.high_frequency_hz
            * self.n_fft
            / self.sampling_frequency_hz
        ))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "first_frequency_bin": self.first_frequency_bin,
            "last_frequency_bin": self.last_frequency_bin,
            "formula": "log1p(abs(STFT) ** 2)",
            "window": "periodic_hann",
            "center": False,
            "mosaic_order": "channel_index_row_major_6x3",
            "rgb_conversion": "repeat_grayscale_mosaic_three_times",
            "normalization": "per_window_global_zscore_clip_then_scale",
        }


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(
            rank,
            base_layer.in_features,
            dtype=base_layer.weight.dtype,
            device=base_layer.weight.device,
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base_layer.out_features,
            rank,
            dtype=base_layer.weight.dtype,
            device=base_layer.weight.device,
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(values)
        update = nn.functional.linear(
            nn.functional.linear(self.dropout(values), self.lora_a),
            self.lora_b,
        )
        return base + update * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    def validate(self) -> None:
        if self.rank < 1 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        if not self.target_modules:
            raise ValueError("LoRA requires at least one target module")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "target_modules": list(self.target_modules),
        }


def inject_lora(
    model: nn.Module,
    *,
    config: LoRAConfig,
) -> list[str]:
    config.validate()
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in config.target_modules
        ):
            replacements.append((name, module))
    if not replacements:
        raise ValueError("No matching linear layers found for LoRA")

    names: list[str] = []
    for name, module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
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


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


class M9STFTMosaicEfficientNet(nn.Module):
    """Create an RGB spectrogram mosaic and return EfficientNet visual tokens."""

    output_dim = 1280

    def __init__(
        self,
        *,
        pretrained: bool = True,
        config: M9STFTConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or M9STFTConfig()
        self.config.validate()
        weights_type, factory = _require_torchvision()
        weights = weights_type.DEFAULT if pretrained else None
        backbone = factory(weights=weights)
        self.features = backbone.features
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.config.win_length, periodic=True),
            persistent=True,
        )

    def spectrogram_mosaic(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be shaped (batch, 1, channels, time)")
        waveform = image[:, 0]
        batch, channels, timepoints = waveform.shape
        if channels != self.config.eeg_channels:
            raise ValueError(
                f"Expected {self.config.eeg_channels} channels, got {channels}"
            )
        if timepoints < self.config.n_fft:
            raise ValueError("EEG window is shorter than n_fft")
        flat = waveform.reshape(batch * channels, timepoints)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                flat.float(),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.stft_window.float(),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square()[
                :,
                self.config.first_frequency_bin
                : self.config.last_frequency_bin + 1,
            ]
            frequency_bins, frames = power.shape[-2:]
            log_power = torch.log1p(power).reshape(
                batch,
                channels,
                frequency_bins,
                frames,
            )
            global_log_power = torch.log1p(
                power.reshape(batch, -1).mean(dim=1)
            )
            center = log_power.mean(dim=(1, 2, 3), keepdim=True)
            scale = log_power.std(
                dim=(1, 2, 3),
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)
            normalized = (
                (log_power - center) / scale
            ).clamp(
                -self.config.normalization_clip,
                self.config.normalization_clip,
            ) / self.config.normalization_clip
            mosaic = normalized.reshape(
                batch,
                self.config.grid_rows,
                self.config.grid_columns,
                frequency_bins,
                frames,
            ).permute(0, 1, 3, 2, 4).reshape(
                batch,
                1,
                self.config.grid_rows * frequency_bins,
                self.config.grid_columns * frames,
            )
            resized = nn.functional.interpolate(
                mosaic,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
            rgb = resized.repeat(1, 3, 1, 1)
        return rgb, global_log_power

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mosaic, global_log_power = self.spectrogram_mosaic(image)
        feature_map = self.features(mosaic)
        if feature_map.ndim != 4 or feature_map.shape[1] != self.output_dim:
            raise ValueError(
                f"Unexpected EfficientNet feature map: {tuple(feature_map.shape)}"
            )
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        pooled = feature_map.mean(dim=(2, 3))
        return tokens, pooled, global_log_power


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("M9 requires transformers") from exc
    return AutoModel, AutoTokenizer


class M9STFTQwenLoRAResidual(nn.Module):
    """Fuse Qwen LoRA output with a direct EfficientNet residual."""

    model_name = "m9_stft_efficientnet_qwen_lora_residual"
    model_version = M9_MODEL_VERSION

    def __init__(
        self,
        *,
        language_model: nn.Module,
        prompt_input_ids: torch.Tensor,
        visual_encoder: M9STFTMosaicEfficientNet,
        lora_config: LoRAConfig,
        qwen_model_name: str,
        qwen_revision: str | None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("prompt_input_ids must be shaped (1, prompt_tokens)")
        for parameter in language_model.parameters():
            parameter.requires_grad = False
        self.lora_module_names = inject_lora(
            language_model,
            config=lora_config,
        )
        self.language_model = language_model
        self.visual_encoder = visual_encoder
        hidden_size = int(
            language_model.get_input_embeddings().embedding_dim
        )
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(visual_encoder.output_dim),
            nn.Linear(visual_encoder.output_dim, hidden_size),
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(visual_encoder.output_dim + 1),
            nn.Linear(visual_encoder.output_dim + 1, hidden_size),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        self.register_buffer(
            "prompt_input_ids",
            prompt_input_ids.to(dtype=torch.long),
            persistent=True,
        )
        self.lora_config = lora_config
        self.qwen_model_name = qwen_model_name
        self.qwen_revision = qwen_revision
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model, "config"):
            self.language_model.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        qwen_model_name: str = DEFAULT_QWEN_MODEL,
        prompt: str = DEFAULT_TASK_PROMPT,
        local_files_only: bool = True,
        pretrained_visual_encoder: bool = True,
        stft_config: M9STFTConfig | None = None,
        lora_config: LoRAConfig | None = None,
    ) -> "M9STFTQwenLoRAResidual":
        auto_model, auto_tokenizer = _require_transformers()
        tokenizer = auto_tokenizer.from_pretrained(
            qwen_model_name,
            local_files_only=local_files_only,
        )
        language_model = auto_model.from_pretrained(
            qwen_model_name,
            local_files_only=local_files_only,
        )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        revision = getattr(language_model.config, "_commit_hash", None)
        return cls(
            language_model=language_model,
            prompt_input_ids=prompt_ids,
            visual_encoder=M9STFTMosaicEfficientNet(
                pretrained=pretrained_visual_encoder,
                config=stft_config,
            ),
            lora_config=lora_config or LoRAConfig(),
            qwen_model_name=qwen_model_name,
            qwen_revision=revision,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        visual_tokens, pooled, global_log_power = self.visual_encoder(image)
        projected = self.visual_projector(visual_tokens)
        prompt = self.language_model.get_input_embeddings()(
            self.prompt_input_ids.expand(image.shape[0], -1)
        )
        multimodal = torch.cat([prompt, projected], dim=1)
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
        qwen_summary = outputs.last_hidden_state[:, -1]
        residual_input = torch.cat(
            [pooled, global_log_power[:, None].to(dtype=pooled.dtype)],
            dim=1,
        )
        residual = self.residual_projection(residual_input)
        gate = torch.sigmoid(self.residual_gate)
        fused = self.fusion_norm(qwen_summary + gate * residual)
        return self.classifier(self.dropout(fused))

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "qwen_lora": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name.startswith("language_model.") and parameter.requires_grad
            ),
            "visual_encoder": sum(
                parameter.numel()
                for parameter in self.visual_encoder.parameters()
            ),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "version": M9_MODEL_VERSION,
            "qwen_model_name": self.qwen_model_name,
            "qwen_revision": self.qwen_revision,
            "prompt_token_count": int(self.prompt_input_ids.shape[1]),
            "visual_token_count": (
                self.visual_encoder.config.image_size // 32
            ) ** 2,
            "hidden_size": int(
                self.language_model.get_input_embeddings().embedding_dim
            ),
            "stft": self.visual_encoder.config.to_dict(),
            "lora": {
                **self.lora_config.to_dict(),
                "injected_module_count": len(self.lora_module_names),
                "injected_modules": list(self.lora_module_names),
            },
            "fusion": (
                "layer_norm(qwen_last_token + "
                "sigmoid(gate) * residual(efficientnet_pool,global_log_power))"
            ),
            "parameters": self.parameter_summary(),
        }


def portable_state_dict(
    model: M9STFTQwenLoRAResidual,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if (
            not name.startswith("language_model.")
            or name.endswith(".lora_a")
            or name.endswith(".lora_b")
        ):
            state[name] = value.detach().cpu().clone()
    return state


def load_portable_state_dict(
    model: M9STFTQwenLoRAResidual,
    state: Mapping[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(dict(state), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if (
            not name.startswith("language_model.")
            or name.endswith(".lora_a")
            or name.endswith(".lora_b")
        )
    ]
    if unexpected or invalid_missing:
        raise ValueError(
            "Portable M9 state is incompatible: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
