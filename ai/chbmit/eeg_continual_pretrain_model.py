"""Server-document EEG-VL pretraining model reproduction.

The source document specifies a 20-channel STFT-EfficientNet-Qwen model but
does not define how CHB-MIT's heterogeneous EDF montages become 20 channels.
This module therefore keeps that adaptation explicit and versioned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .contracts import CANONICAL_BIPOLAR_CHANNELS
from .eegvl_m9_model import LoRAConfig, LoRALinear, inject_lora
from .eegvl_models import (
    DEFAULT_QWEN_MODEL,
    _require_torchvision,
    _single_channel_first_conv,
)


SERVER_PRETRAIN_MODEL_VERSION = "eeg_continual_pretrain_stft_qwen_v1"
SERVER_TASK_PROMPT = (
    "You are an AI system for EEG signal analysis. Analyze CNN-processed EEG "
    "features to classify signals as Normal or Seizure. Input features: The "
    "EEG is"
)
SERVER_SOURCE_CHANNELS = tuple(CANONICAL_BIPOLAR_CHANNELS)
SERVER_ADAPTED_CHANNELS = (
    *SERVER_SOURCE_CHANNELS,
    "P7-T7=-(T7-P7)",
    "T8-P8#2=(T8-P8)",
)

E2_FREQUENCY_BANDS_HZ = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 55.0),
    (55.0, 100.0),
)


@dataclass(frozen=True)
class ServerSTFTConfig:
    sampling_frequency_hz: int = 256
    input_samples: int = 1024
    source_channels: int = 18
    eeg_channels: int = 20
    n_fft: int = 64
    win_length: int = 64
    hop_length: int = 32
    center: bool = True
    visual_tokens: int = 32
    input_scale_uv: float = 1024.0
    zscore_input: bool = False

    def validate(self) -> None:
        if self.sampling_frequency_hz < 1 or self.input_samples < 1:
            raise ValueError("Invalid STFT sampling contract")
        if self.source_channels != 18 or self.eeg_channels != 20:
            raise ValueError("Server reproduction requires 18-to-20 channels")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be inside n_fft")
        if self.hop_length < 1 or self.visual_tokens < 1:
            raise ValueError("Invalid STFT hop/token count")
        if self.input_scale_uv <= 0:
            raise ValueError("input_scale_uv must be positive")

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def time_frames(self) -> int:
        if self.center:
            return self.input_samples // self.hop_length + 1
        return (self.input_samples - self.n_fft) // self.hop_length + 1

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.eeg_channels * self.frequency_bins, self.time_frames

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "frequency_bins": self.frequency_bins,
            "time_frames": self.time_frames,
            "image_shape": list(self.image_shape),
            "window": "periodic_hann",
            "magnitude": "log1p(abs(stft))",
            "layout": "channels_concatenated_along_frequency_axis",
        }


class Canonical18ToServer20(nn.Module):
    """Adapt the stable local montage to the PDF's undocumented 20 channels."""

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 4 or waveform.shape[1] != 1:
            raise ValueError(
                "waveform must be shaped (batch, 1, channels, time)"
            )
        channels = int(waveform.shape[2])
        if channels == 20:
            return waveform
        if channels != 18:
            raise ValueError(f"Expected 18 or 20 EEG channels, got {channels}")
        values = waveform[:, 0]
        p7_t7 = -values[:, 2:3]
        repeated_t8_p8 = values[:, 14:15]
        return torch.cat([values, p7_t7, repeated_t8_p8], dim=1).unsqueeze(1)

    def contract(self) -> dict[str, Any]:
        return {
            "source_channels": list(SERVER_SOURCE_CHANNELS),
            "output_channels": list(SERVER_ADAPTED_CHANNELS),
            "status": "local_reproduction_inference_not_specified_by_pdf",
            "rationale": (
                "The common 23-channel CHB-MIT EDF layout appends P7-T7 and "
                "a repeated T8-P8 to the stable 18-channel bipolar montage."
            ),
        }


class ServerSTFTEfficientNetEncoder(nn.Module):
    """Convert EEG to the configured STFT image and 32 visual tokens."""

    output_dim = 1280

    def __init__(
        self,
        *,
        pretrained: bool = True,
        config: ServerSTFTConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ServerSTFTConfig()
        self.config.validate()
        weights_type, factory = _require_torchvision()
        weights = weights_type.DEFAULT if pretrained else None
        backbone = factory(weights=weights)
        _single_channel_first_conv(backbone)
        self.features = backbone.features
        self.channel_adapter = Canonical18ToServer20()
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.config.win_length, periodic=True),
            persistent=True,
        )

    def log_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        adapted = self.channel_adapter(waveform)
        batch, _, channels, samples = adapted.shape
        if channels != self.config.eeg_channels:
            raise ValueError("Channel adapter did not produce 20 channels")
        if samples != self.config.input_samples:
            raise ValueError(
                f"Expected {self.config.input_samples} samples, got {samples}"
            )
        values = adapted[:, 0].float() * self.config.input_scale_uv
        if self.config.zscore_input:
            center = values.mean(dim=-1, keepdim=True)
            scale = values.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            values = (values - center) / scale
        flat = values.reshape(batch * channels, samples)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                flat,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.stft_window.float(),
                center=self.config.center,
                return_complex=True,
            )
            return torch.log1p(spectrum.abs()).reshape(
                batch,
                channels,
                self.config.frequency_bins,
                self.config.time_frames,
            )

    def spectrogram_image(self, waveform: torch.Tensor) -> torch.Tensor:
        log_magnitude = self.log_magnitude(waveform)
        batch = int(log_magnitude.shape[0])
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            image = log_magnitude.reshape(
                batch,
                1,
                self.config.eeg_channels * self.config.frequency_bins,
                self.config.time_frames,
            )
        return image

    def forward_log_magnitude(
        self,
        log_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(log_magnitude.shape[0])
        image = log_magnitude.reshape(
            batch,
            1,
            self.config.eeg_channels * self.config.frequency_bins,
            self.config.time_frames,
        )
        feature_map = self.features(image)
        if feature_map.ndim != 4 or feature_map.shape[1] != self.output_dim:
            raise ValueError(
                f"Unexpected EfficientNet feature map: {tuple(feature_map.shape)}"
            )
        sequence = feature_map.flatten(2)
        pooled = nn.functional.adaptive_avg_pool1d(
            sequence,
            self.config.visual_tokens,
        )
        return pooled.transpose(1, 2).contiguous()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward_log_magnitude(self.log_magnitude(waveform))


class PatientRelativeSpectrumEncoder(nn.Module):
    """Encode E2 patient-relative log-spectrum band differences."""

    output_features = 20 * len(E2_FREQUENCY_BANDS_HZ)

    def __init__(self, *, config: ServerSTFTConfig) -> None:
        super().__init__()
        frequencies = torch.fft.rfftfreq(
            config.n_fft,
            d=1.0 / config.sampling_frequency_hz,
        )
        masks = []
        for index, (lower, upper) in enumerate(E2_FREQUENCY_BANDS_HZ):
            lower_check = (
                frequencies >= lower if index == 0 else frequencies > lower
            )
            mask = lower_check & (frequencies <= upper)
            if not bool(mask.any()):
                raise ValueError(
                    f"STFT has no bin for E2 band {lower:g}-{upper:g} Hz"
                )
            masks.append(mask)
        self.register_buffer(
            "band_masks",
            torch.stack(masks),
            persistent=True,
        )
        self.config = config

    def forward(
        self,
        log_magnitude: torch.Tensor,
        baseline_log_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            log_magnitude.shape[0],
            self.config.eeg_channels,
            self.config.frequency_bins,
        )
        if baseline_log_magnitude.shape != expected:
            raise ValueError(
                "baseline_log_magnitude must be shaped "
                f"{expected}, got {tuple(baseline_log_magnitude.shape)}"
            )
        relative = log_magnitude - baseline_log_magnitude.unsqueeze(-1)
        features = [
            relative[:, :, mask, :].mean(dim=(-2, -1))
            for mask in self.band_masks
        ]
        return torch.stack(features, dim=-1).flatten(1)


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Server EEG-VL reproduction requires transformers") from exc
    return AutoModel, AutoTokenizer


def _fixed_prompt_ids(
    tokenizer: Any,
    prompt: str,
    *,
    token_count: int,
) -> torch.Tensor:
    ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    ids = ids[:, :token_count]
    if ids.shape[1] < token_count:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        padding = torch.full(
            (1, token_count - ids.shape[1]),
            int(pad_id or 0),
            dtype=torch.long,
        )
        ids = torch.cat([ids, padding], dim=1)
    return ids


class ServerEEGVLPretrainModel(nn.Module):
    """Document-faithful STFT-EfficientNet-Qwen-LoRA binary classifier."""

    model_name = "server_stft_efficientnet_qwen_lora"
    model_version = SERVER_PRETRAIN_MODEL_VERSION

    def __init__(
        self,
        *,
        language_model: nn.Module,
        prompt_input_ids: torch.Tensor,
        visual_encoder: ServerSTFTEfficientNetEncoder,
        lora_config: LoRAConfig,
        qwen_model_name: str,
        qwen_revision: str | None,
        pooling: str = "mean",
        visual_bypass: bool = False,
        relative_spectral_bypass: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be mean or last")
        if prompt_input_ids.shape != (1, 35):
            raise ValueError("Server prompt must contain exactly 35 tokens")
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
        if hidden_size != 896:
            raise ValueError(f"Expected Qwen hidden size 896, got {hidden_size}")
        with torch.no_grad():
            prompt_seed = language_model.get_input_embeddings()(
                prompt_input_ids
            ).detach().clone()
        self.prompt_embeddings = nn.Parameter(prompt_seed)
        self.visual_projection = nn.Linear(visual_encoder.output_dim, hidden_size)
        self.head_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.visual_bypass = bool(visual_bypass)
        self.bypass_projection = (
            nn.Linear(visual_encoder.output_dim, hidden_size)
            if self.visual_bypass
            else None
        )
        self.relative_spectral_bypass = bool(relative_spectral_bypass)
        self.relative_spectral_encoder = (
            PatientRelativeSpectrumEncoder(config=visual_encoder.config)
            if self.relative_spectral_bypass
            else None
        )
        self.relative_spectral_projection = (
            nn.Sequential(
                nn.LayerNorm(PatientRelativeSpectrumEncoder.output_features),
                nn.Linear(
                    PatientRelativeSpectrumEncoder.output_features,
                    256,
                ),
                nn.GELU(),
                nn.Linear(256, hidden_size),
            )
            if self.relative_spectral_bypass
            else None
        )
        self.pooling = pooling
        self.lora_config = lora_config
        self.qwen_model_name = qwen_model_name
        self.qwen_revision = qwen_revision
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
        qwen_model_name: str = DEFAULT_QWEN_MODEL,
        prompt: str = SERVER_TASK_PROMPT,
        local_files_only: bool = True,
        pretrained_visual_encoder: bool = True,
        stft_config: ServerSTFTConfig | None = None,
        lora_config: LoRAConfig | None = None,
        pooling: str = "mean",
        visual_bypass: bool = False,
        relative_spectral_bypass: bool = False,
    ) -> "ServerEEGVLPretrainModel":
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
            language_model=language_model,
            prompt_input_ids=prompt_ids,
            visual_encoder=ServerSTFTEfficientNetEncoder(
                pretrained=pretrained_visual_encoder,
                config=stft_config,
            ),
            lora_config=lora_config or LoRAConfig(
                target_modules=("q_proj", "v_proj"),
            ),
            qwen_model_name=qwen_model_name,
            qwen_revision=revision,
            pooling=pooling,
            visual_bypass=visual_bypass,
            relative_spectral_bypass=relative_spectral_bypass,
        )

    def train(self, mode: bool = True) -> "ServerEEGVLPretrainModel":
        super().train(mode)
        self.language_model.eval()
        for _, module in self.language_model.named_modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        baseline_log_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_magnitude = self.visual_encoder.log_magnitude(waveform)
        visual_tokens = self.visual_encoder.forward_log_magnitude(
            log_magnitude
        )
        projected = self.visual_projection(visual_tokens)
        prompt = self.prompt_embeddings.expand(waveform.shape[0], -1, -1)
        multimodal = torch.cat([prompt, projected], dim=1)
        if multimodal.shape[1] != 67:
            raise ValueError("Expected 35 prompt + 32 visual tokens")
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
        if self.bypass_projection is not None:
            summary = summary + self.bypass_projection(visual_tokens.mean(dim=1))
        if self.relative_spectral_projection is not None:
            if (
                self.relative_spectral_encoder is None
                or baseline_log_magnitude is None
            ):
                raise ValueError(
                    "E2 bypass requires baseline_log_magnitude"
                )
            e2_features = self.relative_spectral_encoder(
                log_magnitude,
                baseline_log_magnitude.float(),
            )
            summary = summary + self.relative_spectral_projection(e2_features)
        bounded = torch.tanh(self.head_norm(summary))
        return self.classifier(bounded)

    def parameter_summary(self) -> dict[str, int]:
        named = list(self.named_parameters())
        return {
            "total": sum(parameter.numel() for _, parameter in named),
            "trainable": sum(
                parameter.numel()
                for _, parameter in named
                if parameter.requires_grad
            ),
            "qwen_lora": sum(
                parameter.numel()
                for name, parameter in named
                if name.startswith("language_model.") and parameter.requires_grad
            ),
            "efficientnet": sum(
                parameter.numel()
                for parameter in self.visual_encoder.parameters()
            ),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "version": self.model_version,
            "source_document": "eeg_continual_learning_doc.pdf",
            "qwen_model_name": self.qwen_model_name,
            "qwen_revision": self.qwen_revision,
            "hidden_size": 896,
            "prompt_tokens": 35,
            "visual_tokens": 32,
            "sequence_tokens": 67,
            "pooling": self.pooling,
            "visual_bypass": self.visual_bypass,
            "relative_spectral_bypass": self.relative_spectral_bypass,
            "stft": self.visual_encoder.config.to_dict(),
            "channel_adapter": self.visual_encoder.channel_adapter.contract(),
            "lora": {
                **self.lora_config.to_dict(),
                "injected_modules": list(self.lora_module_names),
            },
            "e2": {
                "enabled": self.relative_spectral_bypass,
                "features": PatientRelativeSpectrumEncoder.output_features,
                "bands_hz": [
                    list(values) for values in E2_FREQUENCY_BANDS_HZ
                ],
                "definition": "mean(log1p magnitude - patient baseline)",
            },
            "head": "LayerNorm -> tanh -> zero-initialized Linear(896,2)",
            "parameters": self.parameter_summary(),
        }


def portable_pretrain_state_dict(
    model: ServerEEGVLPretrainModel,
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


def load_portable_pretrain_state_dict(
    model: ServerEEGVLPretrainModel,
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
            "Portable server-pretrain state is incompatible: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
