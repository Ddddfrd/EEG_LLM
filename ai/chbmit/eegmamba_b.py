"""EEGMamba plus patient-relative E2 features for seizure classification.

The backbone mirrors the official Wang et al. EEGMamba implementation while
keeping the optional ``mamba_ssm`` dependency behind model construction. The
official checkpoint expects 200 samples per patch. Local four-second CHB-MIT
windows are therefore Fourier-resampled from 256 Hz to 200 Hz and reshaped to
four patches per channel.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_continual_pretrain_model import (
    E2_FREQUENCY_BANDS_HZ,
    PatientRelativeSpectrumEncoder,
    ServerSTFTConfig,
    ServerSTFTEfficientNetEncoder,
)


EEGMAMBA_B_MODEL_VERSION = "eegmamba_b_e2_v1"
OFFICIAL_EEGMAMBA_CHECKPOINT_SHA256 = (
    "b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe"
)


@dataclass(frozen=True)
class EEGMambaInputConfig:
    source_sampling_frequency_hz: int = 256
    target_sampling_frequency_hz: int = 200
    input_samples: int = 1024
    channels: int = 18
    patch_samples: int = 200
    input_scale_uv: float = 1024.0
    official_amplitude_divisor_uv: float = 100.0

    def validate(self) -> None:
        if self.source_sampling_frequency_hz != 256:
            raise ValueError("Mamba-B currently expects 256 Hz source windows")
        if self.input_samples != 1024 or self.channels != 18:
            raise ValueError("Mamba-B requires (18, 1024) four-second windows")
        if self.target_samples % self.patch_samples:
            raise ValueError("Target samples must divide into complete patches")
        if self.input_scale_uv <= 0 or self.official_amplitude_divisor_uv <= 0:
            raise ValueError("Input amplitude scales must be positive")

    @property
    def target_samples(self) -> int:
        return int(
            round(
                self.input_samples
                * self.target_sampling_frequency_hz
                / self.source_sampling_frequency_hz
            )
        )

    @property
    def patch_count(self) -> int:
        return self.target_samples // self.patch_samples

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "target_samples": self.target_samples,
            "patch_count": self.patch_count,
            "resampling": "real_fft_fourier_resample",
            "output_shape": [self.channels, self.patch_count, self.patch_samples],
        }


def fourier_resample_real(values: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Match scipy.signal.resample for real-valued signals along the last axis."""
    source_samples = int(values.shape[-1])
    if source_samples < 1 or target_samples < 1:
        raise ValueError("Source and target sample counts must be positive")
    if source_samples == target_samples:
        return values
    spectrum = torch.fft.rfft(values.float(), dim=-1)
    output_bins = target_samples // 2 + 1
    resized = spectrum.new_zeros((*spectrum.shape[:-1], output_bins))
    shared_samples = min(source_samples, target_samples)
    shared_bins = shared_samples // 2 + 1
    resized[..., :shared_bins] = spectrum[..., :shared_bins]
    if shared_samples % 2 == 0:
        nyquist = shared_samples // 2
        if target_samples < source_samples:
            resized[..., nyquist] *= 2.0
        elif source_samples < target_samples:
            resized[..., nyquist] *= 0.5
    output = torch.fft.irfft(resized, n=target_samples, dim=-1)
    return output * (target_samples / source_samples)


def waveform_to_eegmamba_patches(
    waveform: torch.Tensor,
    *,
    config: EEGMambaInputConfig,
) -> torch.Tensor:
    config.validate()
    expected = (1, config.channels, config.input_samples)
    if waveform.ndim != 4 or tuple(waveform.shape[1:]) != expected:
        raise ValueError(
            f"waveform must be shaped (batch, 1, 18, 1024), got {tuple(waveform.shape)}"
        )
    microvolts = waveform[:, 0].float() * config.input_scale_uv
    official_scale = microvolts / config.official_amplitude_divisor_uv
    resampled = fourier_resample_real(official_scale, config.target_samples)
    return resampled.reshape(
        waveform.shape[0],
        config.channels,
        config.patch_count,
        config.patch_samples,
    )


def _require_mamba_components() -> tuple[Any, Any, Any, Any]:
    try:
        from mamba_ssm.modules.block import Block
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
    except ImportError as exc:
        raise RuntimeError(
            "EEGMamba requires Linux CUDA packages mamba-ssm and causal-conv1d"
        ) from exc
    return Block, Mamba2, RMSNorm, layer_norm_fn


class EEGMambaPatchEmbedding(nn.Module):
    """Official EEGMamba time, spectral, and 2D positional patch embedding."""

    def __init__(self, *, d_model: int = 200, patch_samples: int = 200) -> None:
        super().__init__()
        if d_model != 200 or patch_samples != 200:
            raise ValueError("The official checkpoint requires d_model=patch_samples=200")
        self.d_model = d_model
        self.mask_encoding = nn.Parameter(torch.zeros(patch_samples), requires_grad=False)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                groups=d_model,
                bias=False,
            )
        )
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                1,
                25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
                bias=False,
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model, bias=False),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, patch_count, patch_samples = patches.shape
        if patch_samples != 200:
            raise ValueError("Official EEGMamba patches must contain 200 samples")
        masked = patches
        if mask is not None:
            masked = patches.clone()
            masked[mask == 1] = self.mask_encoding
        time_sequence = masked.reshape(batch, channels * patch_count, patch_samples)
        time_embedding = self.proj_in(time_sequence.unsqueeze(1))
        time_embedding = (
            time_embedding.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, channels, patch_count, self.d_model)
        )
        spectral = torch.fft.rfft(masked, dim=-1, norm="forward").abs()
        patch_embedding = time_embedding + self.spectral_proj(spectral)
        positional = self.positional_encoding(patch_embedding.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        return patch_embedding + positional


class AlternatingMambaEncoder(nn.Module):
    """Official 12-layer encoder, reversing sequence direction after each layer."""

    def __init__(self, *, d_model: int = 200, n_layer: int = 12) -> None:
        super().__init__()
        block_type, mamba_type, rms_norm_type, layer_norm_fn = _require_mamba_components()
        norm_factory: Callable[..., nn.Module] = partial(rms_norm_type, eps=1e-5)
        self.layers = nn.ModuleList(
            [
                block_type(
                    d_model,
                    partial(
                        mamba_type,
                        layer_idx=index,
                        headdim=50,
                        d_state=64,
                    ),
                    nn.Identity,
                    norm_cls=norm_factory,
                    fused_add_norm=True,
                    residual_in_fp32=True,
                )
                for index in range(n_layer)
            ]
        )
        self.norm_f = rms_norm_type(d_model, eps=1e-5)
        self.layer_norm_fn = layer_norm_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
            hidden_states = hidden_states.flip(1)
            residual = residual.flip(1)
        return self.layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
            is_rms_norm=True,
        )


class EEGMambaBackbone(nn.Module):
    """Checkpoint-compatible official EEGMamba backbone."""

    output_dim = 200

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = EEGMambaPatchEmbedding()
        self.encoder = AlternatingMambaEncoder()
        self.proj_out = nn.Sequential(nn.Linear(200, 200))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, channels, patch_count, _ = patches.shape
        embedded = self.patch_embedding(patches)
        encoded = self.encoder(embedded.reshape(batch, channels * patch_count, 200))
        encoded = encoded.reshape(batch, channels, patch_count, 200)
        return self.proj_out(encoded)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_eegmamba_backbone(
    checkpoint_path: Path,
    *,
    expected_sha256: str = OFFICIAL_EEGMAMBA_CHECKPOINT_SHA256,
) -> EEGMambaBackbone:
    checkpoint_path = Path(checkpoint_path)
    actual_sha256 = file_sha256(checkpoint_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "Official EEGMamba checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Official EEGMamba checkpoint is not a state_dict")
    backbone = EEGMambaBackbone()
    incompatible = backbone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Official EEGMamba checkpoint is incompatible")
    backbone.proj_out = nn.Identity()
    return backbone


class E2LogMagnitudeFrontEnd(nn.Module):
    """Lightweight E2 STFT front end without constructing EfficientNet."""

    def __init__(self, *, config: ServerSTFTConfig) -> None:
        super().__init__()
        # Reuse the established, tested STFT implementation without its CNN.
        self.config = config
        self._log_magnitude = ServerSTFTEfficientNetEncoder.log_magnitude
        from .eeg_continual_pretrain_model import Canonical18ToServer20

        self.channel_adapter = Canonical18ToServer20()
        self.register_buffer(
            "stft_window",
            torch.hann_window(config.win_length, periodic=True),
            persistent=True,
        )

    def log_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._log_magnitude(self, waveform)


class EEGMambaBE2Classifier(nn.Module):
    """Mamba-B: official EEGMamba representation plus 120-dimensional E2."""

    model_name = "eegmamba_b_e2_classifier"
    model_version = EEGMAMBA_B_MODEL_VERSION

    def __init__(
        self,
        *,
        backbone: nn.Module,
        input_config: EEGMambaInputConfig | None = None,
        stft_config: ServerSTFTConfig | None = None,
        checkpoint_sha256: str | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_config = input_config or EEGMambaInputConfig()
        self.input_config.validate()
        self.stft_config = stft_config or ServerSTFTConfig()
        self.stft_config.validate()
        self.e2_frontend = E2LogMagnitudeFrontEnd(config=self.stft_config)
        self.e2_encoder = PatientRelativeSpectrumEncoder(config=self.stft_config)
        self.mamba_norm = nn.LayerNorm(200)
        self.e2_norm = nn.LayerNorm(PatientRelativeSpectrumEncoder.output_features)
        self.classifier = nn.Sequential(
            nn.Linear(200 + PatientRelativeSpectrumEncoder.output_features, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 2),
        )
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def from_official_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        input_config: EEGMambaInputConfig | None = None,
        stft_config: ServerSTFTConfig | None = None,
        dropout: float = 0.1,
    ) -> "EEGMambaBE2Classifier":
        digest = file_sha256(checkpoint_path)
        backbone = load_official_eegmamba_backbone(checkpoint_path)
        return cls(
            backbone=backbone,
            input_config=input_config,
            stft_config=stft_config,
            checkpoint_sha256=digest,
            dropout=dropout,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        baseline_log_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if baseline_log_magnitude is None:
            raise ValueError("Mamba-B requires baseline_log_magnitude for E2")
        patches = waveform_to_eegmamba_patches(
            waveform,
            config=self.input_config,
        )
        representation = self.backbone(patches)
        if representation.shape[-1] != 200:
            raise ValueError("EEGMamba backbone must return 200-dimensional patch features")
        mamba_features = representation.mean(dim=(1, 2))
        log_magnitude = self.e2_frontend.log_magnitude(waveform)
        e2_features = self.e2_encoder(
            log_magnitude,
            baseline_log_magnitude.float(),
        )
        fused = torch.cat(
            [
                self.mamba_norm(mamba_features),
                self.e2_norm(e2_features),
            ],
            dim=1,
        )
        return self.classifier(fused)

    def parameter_summary(self) -> dict[str, int]:
        named = list(self.named_parameters())
        return {
            "total": sum(parameter.numel() for _, parameter in named),
            "trainable": sum(
                parameter.numel() for _, parameter in named if parameter.requires_grad
            ),
            "backbone": sum(
                parameter.numel() for name, parameter in named if name.startswith("backbone.")
            ),
            "e2_and_head": sum(
                parameter.numel() for name, parameter in named if not name.startswith("backbone.")
            ),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "version": self.model_version,
            "architecture": "official EEGMamba pooled patches + E2 + MLP head",
            "official_repository": "https://github.com/wjq-learning/EEGMamba",
            "official_checkpoint_sha256": self.checkpoint_sha256,
            "input": self.input_config.to_dict(),
            "backbone": {
                "d_model": 200,
                "layers": 12,
                "mamba": "Mamba2",
                "d_state": 64,
                "head_dim": 50,
                "direction": "sequence_flip_after_each_layer",
                "pooling": "mean_over_channels_and_patches",
                "fine_tuning": "all_backbone_parameters_trainable",
            },
            "e2": {
                "features": PatientRelativeSpectrumEncoder.output_features,
                "bands_hz": [list(band) for band in E2_FREQUENCY_BANDS_HZ],
                "definition": "mean(log1p magnitude - patient baseline)",
                "stft": self.stft_config.to_dict(),
            },
            "fusion": "LayerNorm(EEGMamba 200) || LayerNorm(E2 120)",
            "head": "Linear(320,200) -> ELU -> Dropout -> Linear(200,2)",
            "parameters": self.parameter_summary(),
        }
