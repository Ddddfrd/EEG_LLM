"""STFT-EfficientNet-Qwen model with E2, E3, and E4 residual branches."""

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
    ServerSTFTEfficientNetEncoder,
    _fixed_prompt_ids,
    _require_transformers,
)
from .eegmamba_b import file_sha256
from .eegvl_m9_model import LoRAConfig, LoRALinear, inject_lora
from .eegvl_models import DEFAULT_QWEN_MODEL


MULTIBRANCH_MODEL_VERSION = "eegvl_e1_e2_e3_e4_residual_v1"
E3_HIGH_FREQUENCY_BAND_HZ = (20.0, 70.0)
E3_TOTAL_FREQUENCY_BAND_HZ = (0.5, 100.0)
E3_SPIKE_MAD_MULTIPLIER = 6.0
E4_FREQUENCY_BANDS_HZ = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _frequency_mask(
    frequencies: torch.Tensor,
    band: tuple[float, float],
) -> torch.Tensor:
    lower, upper = band
    return (frequencies >= lower) & (frequencies <= upper)


class E3E4PhysiologyEncoder(nn.Module):
    """Extract gain-robust transient and focal spectral-ratio features."""

    e3_output_features = 40
    e4_output_features = 80

    def __init__(self, *, config: ServerSTFTConfig) -> None:
        super().__init__()
        frequencies = torch.fft.rfftfreq(
            config.input_samples,
            d=1.0 / config.sampling_frequency_hz,
        )
        masks = {
            "high": _frequency_mask(frequencies, E3_HIGH_FREQUENCY_BAND_HZ),
            "total": _frequency_mask(frequencies, E3_TOTAL_FREQUENCY_BAND_HZ),
            **{
                name: _frequency_mask(frequencies, band)
                for name, band in E4_FREQUENCY_BANDS_HZ.items()
            },
        }
        if not all(bool(mask.any()) for mask in masks.values()):
            raise ValueError("E3/E4 frequency bands must contain FFT bins")
        for name, mask in masks.items():
            self.register_buffer(f"{name}_mask", mask, persistent=True)
        self.config = config

    @staticmethod
    def _band_power(power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return power[..., mask].mean(dim=-1)

    def forward(self, waveform_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.config.eeg_channels, self.config.input_samples)
        if waveform_uv.ndim != 3 or tuple(waveform_uv.shape[1:]) != expected:
            raise ValueError(f"E3/E4 waveform must be shaped (batch, {expected[0]}, {expected[1]})")
        values = waveform_uv.float()
        centered = values - values.mean(dim=-1, keepdim=True)
        window = torch.hann_window(
            self.config.input_samples,
            periodic=True,
            dtype=centered.dtype,
            device=centered.device,
        )
        spectrum = torch.fft.rfft(centered * window, dim=-1)
        power = spectrum.abs().square()
        epsilon = torch.finfo(power.dtype).eps

        high = self._band_power(power, self.high_mask)
        total = self._band_power(power, self.total_mask)
        high_ratio = high / total.clamp_min(epsilon)

        derivative = torch.diff(values, dim=-1).abs()
        median = derivative.median(dim=-1, keepdim=True).values
        mad = (derivative - median).abs().median(dim=-1, keepdim=True).values
        robust_scale = (1.4826 * mad).clamp_min(1e-6)
        spike_threshold = median + E3_SPIKE_MAD_MULTIPLIER * robust_scale
        spike_density = (derivative > spike_threshold).float().mean(dim=-1)
        e3 = torch.stack([high_ratio, spike_density], dim=-1).flatten(1)

        delta = self._band_power(power, self.delta_mask)
        theta = self._band_power(power, self.theta_mask)
        alpha = self._band_power(power, self.alpha_mask)
        beta = self._band_power(power, self.beta_mask)
        low_sum = delta + theta
        ratios = torch.stack(
            [
                theta / beta.clamp_min(epsilon),
                delta / alpha.clamp_min(epsilon),
                beta / low_sum.clamp_min(epsilon),
                alpha / low_sum.clamp_min(epsilon),
            ],
            dim=-1,
        )
        # Isolated near-zero denominators should not dominate the learned projection.
        e4 = ratios.clamp(max=100.0).flatten(1)
        return e3, e4

    def contract(self) -> dict[str, Any]:
        return {
            "e3": {
                "features": self.e3_output_features,
                "high_frequency_ratio": {
                    "numerator_hz": list(E3_HIGH_FREQUENCY_BAND_HZ),
                    "denominator_hz": list(E3_TOTAL_FREQUENCY_BAND_HZ),
                    "power": "mean(abs(rfft(hann(x - mean(x)))) ** 2)",
                },
                "spike_density": {
                    "definition": "mean(abs(diff(x)) > median + 6 * 1.4826 * MAD)",
                    "mad_multiplier": E3_SPIKE_MAD_MULTIPLIER,
                },
            },
            "e4": {
                "features": self.e4_output_features,
                "bands_hz": {name: list(band) for name, band in E4_FREQUENCY_BANDS_HZ.items()},
                "ratios": [
                    "theta/beta",
                    "delta/alpha",
                    "beta/(delta+theta)",
                    "alpha/(delta+theta)",
                ],
                "maximum_ratio": 100.0,
            },
        }


def _xavier_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class EEGVLE1E2E3E4Classifier(nn.Module):
    """EfficientNet visual tokens plus additive E2/E3/E4 residual features."""

    model_name = "eegvl_e1_e2_e3_e4_classifier"
    model_version = MULTIBRANCH_MODEL_VERSION

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
    ) -> None:
        super().__init__()
        supported_pooling = {
            "mean",
            "last",
            "visual_mean",
            "visual_attention",
            "summary_token",
        }
        if pooling not in supported_pooling:
            raise ValueError(
                f"pooling must be one of {sorted(supported_pooling)}, got {pooling!r}"
            )
        if prompt_input_ids.shape != (1, 35):
            raise ValueError("Prompt must contain exactly 35 tokens")
        if visual_encoder.config.zscore_input:
            raise ValueError("Multibranch model must preserve amplitude: zscore_input=False")

        for parameter in language_model.parameters():
            parameter.requires_grad = False
        self.lora_module_names = inject_lora(language_model, config=lora_config)
        self.language_model = language_model
        self.visual_encoder = visual_encoder
        hidden_size = int(language_model.get_input_embeddings().embedding_dim)
        if hidden_size < 1:
            raise ValueError("Language model hidden size must be positive")
        self.hidden_size = hidden_size
        with torch.no_grad():
            prompt_seed = language_model.get_input_embeddings()(prompt_input_ids).detach().clone()
        self.prompt_token_count = int(prompt_seed.shape[1])
        self.visual_token_count = 32
        self.prompt_embeddings = nn.Parameter(prompt_seed)
        self.visual_projection = nn.Linear(visual_encoder.output_dim, hidden_size)
        self.visual_attention: nn.Linear | None = None
        self.summary_embedding: nn.Parameter | None = None
        if pooling == "visual_attention":
            self.visual_attention = nn.Linear(hidden_size, 1, bias=False)
            _xavier_linear(self.visual_attention)
        elif pooling == "summary_token":
            initializer_range = float(
                getattr(getattr(language_model, "config", None), "initializer_range", 0.02)
            )
            self.summary_embedding = nn.Parameter(
                torch.empty(1, 1, hidden_size).normal_(mean=0.0, std=initializer_range)
            )

        self.e2_encoder = PatientRelativeSpectrumEncoder(config=visual_encoder.config)
        self.physiology_encoder = E3E4PhysiologyEncoder(config=visual_encoder.config)
        self.e2_proj = nn.Linear(PatientRelativeSpectrumEncoder.output_features, hidden_size)
        self.e3_proj = nn.Linear(E3E4PhysiologyEncoder.e3_output_features, hidden_size)
        self.e4_proj = nn.Linear(E3E4PhysiologyEncoder.e4_output_features, hidden_size)
        for projection in (self.e2_proj, self.e3_proj, self.e4_proj):
            _xavier_linear(projection)

        self.head_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
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
    ) -> "EEGVLE1E2E3E4Classifier":
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
        config = stft_config or ServerSTFTConfig(
            n_fft=256,
            win_length=128,
            hop_length=32,
            zscore_input=False,
        )
        return cls(
            language_model=language_model,
            prompt_input_ids=prompt_ids,
            visual_encoder=ServerSTFTEfficientNetEncoder(
                pretrained=pretrained_visual_encoder,
                config=config,
            ),
            lora_config=lora_config or LoRAConfig(target_modules=("q_proj", "v_proj")),
            qwen_model_name=qwen_model_name,
            qwen_revision=revision,
            pooling=pooling,
        )

    def train(self, mode: bool = True) -> "EEGVLE1E2E3E4Classifier":
        super().train(mode)
        self.language_model.eval()
        for module in self.language_model.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        baseline_log_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cls_repr = self.forward_fused_representation(
            waveform,
            baseline_log_magnitude=baseline_log_magnitude,
        )
        return self.classifier(cls_repr)

    def _pool_qwen_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        summary_tokens = 1 if self.pooling == "summary_token" else 0
        expected_tokens = (
            self.prompt_token_count + self.visual_token_count + summary_tokens
        )
        if hidden.ndim != 3 or hidden.shape[1] != expected_tokens:
            raise ValueError(
                f"Qwen hidden must contain {expected_tokens} tokens for "
                f"pooling={self.pooling!r}"
            )
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1]

        visual_start = self.prompt_token_count
        visual_end = visual_start + self.visual_token_count
        visual_hidden = hidden[:, visual_start:visual_end]
        if self.pooling == "visual_mean":
            return visual_hidden.mean(dim=1)
        if self.pooling == "visual_attention":
            if self.visual_attention is None:
                raise RuntimeError("visual_attention pooling layer is missing")
            scores = self.visual_attention(visual_hidden).squeeze(-1).float()
            weights = torch.softmax(scores, dim=1).to(dtype=visual_hidden.dtype)
            return torch.sum(visual_hidden * weights.unsqueeze(-1), dim=1)
        if self.pooling == "summary_token":
            return hidden[:, -1]
        raise RuntimeError(f"Unsupported pooling mode: {self.pooling!r}")

    def forward_fused_representation(
        self,
        waveform: torch.Tensor,
        *,
        baseline_log_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the frozen language-model representation used by the classifier."""
        if baseline_log_magnitude is None:
            raise ValueError("E1+E2+E3+E4 requires baseline_log_magnitude")
        log_magnitude = self.visual_encoder.log_magnitude(waveform)
        visual_tokens = self.visual_encoder.forward_log_magnitude(log_magnitude)
        projected = self.visual_projection(visual_tokens)
        prompt = self.prompt_embeddings.expand(waveform.shape[0], -1, -1)
        multimodal = torch.cat([prompt, projected], dim=1)
        if self.pooling == "summary_token":
            if self.summary_embedding is None:
                raise RuntimeError("summary_token embedding is missing")
            summary = self.summary_embedding.expand(waveform.shape[0], -1, -1)
            multimodal = torch.cat([multimodal, summary], dim=1)
        expected_tokens = (
            self.prompt_token_count
            + self.visual_token_count
            + (1 if self.pooling == "summary_token" else 0)
        )
        if multimodal.shape[1] != expected_tokens:
            raise ValueError(
                "Unexpected multimodal token count: "
                f"expected {expected_tokens}, got {multimodal.shape[1]}"
            )
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
        llm_summary = self._pool_qwen_hidden(hidden)
        llm_repr = torch.tanh(self.head_norm(llm_summary))

        e2 = self.e2_encoder(log_magnitude, baseline_log_magnitude.float())
        waveform_uv = (
            self.visual_encoder.channel_adapter(waveform)[:, 0].float()
            * self.visual_encoder.config.input_scale_uv
        )
        e3, e4 = self.physiology_encoder(waveform_uv)
        cls_repr = llm_repr + self.e2_proj(e2) + self.e3_proj(e3) + self.e4_proj(e4)
        return cls_repr

    def parameter_summary(self) -> dict[str, int]:
        groups = {
            "total": 0,
            "trainable": 0,
            "efficientnet": 0,
            "qwen_lora": 0,
            "e2": 0,
            "e3": 0,
            "e4": 0,
            "head": 0,
        }
        for name, parameter in self.named_parameters():
            groups["total"] += parameter.numel()
            if parameter.requires_grad:
                groups["trainable"] += parameter.numel()
            if name.startswith("visual_encoder.features."):
                groups["efficientnet"] += parameter.numel()
            elif name.startswith("language_model.") and parameter.requires_grad:
                groups["qwen_lora"] += parameter.numel()
            elif name.startswith("e2_proj."):
                groups["e2"] += parameter.numel()
            elif name.startswith("e3_proj."):
                groups["e3"] += parameter.numel()
            elif name.startswith("e4_proj."):
                groups["e4"] += parameter.numel()
            elif parameter.requires_grad:
                groups["head"] += parameter.numel()
        return groups

    def contract(self) -> dict[str, Any]:
        return {
            "version": self.model_version,
            "architecture": "STFT EfficientNet + Qwen Q/V LoRA + E2/E3/E4 residuals",
            "input": {
                "cache_shape": [1, 18, 1024],
                "model_shape_after_channel_adapter": [20, 1024],
                "sampling_frequency_hz": 256,
                "duration_seconds": 4,
                "normalization": "clip to +/-1024 uV then divide by 1024",
                "per_window_channel_zscore": False,
            },
            "e1": {
                "stft": self.visual_encoder.config.to_dict(),
                "encoder": "ImageNet EfficientNet-B0 features",
                "visual_tokens": 32,
                "visual_projection": f"Linear(1280,{self.hidden_size})",
                "fine_tuning": "end-to-end",
            },
            "e2": {
                "features": PatientRelativeSpectrumEncoder.output_features,
                "bands_hz": [list(band) for band in E2_FREQUENCY_BANDS_HZ],
                "definition": "mean(log1p magnitude - patient rest baseline)",
                "projection": f"Xavier Linear(120,{self.hidden_size})",
            },
            **self.physiology_encoder.contract(),
            "qwen": {
                "model_name": self.qwen_model_name,
                "revision": self.qwen_revision,
                "frozen_base": True,
                "prompt_tokens": self.prompt_token_count,
                "visual_tokens": self.visual_token_count,
                "summary_tokens": 1 if self.pooling == "summary_token" else 0,
                "sequence_tokens": (
                    self.prompt_token_count
                    + self.visual_token_count
                    + (1 if self.pooling == "summary_token" else 0)
                ),
                "hidden_size": self.hidden_size,
                "pooling": self.pooling,
            },
            "lora": {
                **self.lora_config.to_dict(),
                "injected_modules": list(self.lora_module_names),
            },
            "fusion": (
                f"tanh(LayerNorm(Qwen {self.pooling})) + "
                "e2_proj + e3_proj + e4_proj"
            ),
            "gate_fusion": False,
            "head": f"zero-initialized Linear({self.hidden_size},2)",
            "parameters": self.parameter_summary(),
        }


def portable_multibranch_state_dict(
    model: EEGVLE1E2E3E4Classifier,
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


def load_portable_multibranch_state_dict(
    model: EEGVLE1E2E3E4Classifier,
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
            "Portable multibranch state is incompatible: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )


def checkpoint_sha256(path: Path) -> str:
    return file_sha256(path)
