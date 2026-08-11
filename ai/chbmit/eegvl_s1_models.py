"""Model adapters used by the EEG-VL S1 experiment."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ai.v3.raw_tcn_model import (
    RAW_TCN_MODEL_VERSION,
    GatedChannelAttention,
    SeparableTemporalBlock,
    SharedChannelTCN,
)

from .eegvl_models import (
    EfficientNetLinearClassifier,
    EfficientNetVisualEncoder,
    _require_torchvision,
    count_parameters,
)


EEGVL_S1_MODEL_VERSION = "eegvl_18_s1_models_v3"
S1_MODEL_NAMES = ("m2", "m3", "m6", "m7")
S1_INPUT_MODES = ("single_sum", "rgb_repeat")


class SharedChannelTCNImageClassifier(nn.Module):
    """Adapt the common S1 image tensor to the existing raw TCN contract."""

    model_name = "m2_shared_channel_tcn"
    model_version = RAW_TCN_MODEL_VERSION

    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        embedding_dim: int = 96,
        dropout: float = 0.25,
        channel_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = backbone or SharedChannelTCN(
            embedding_dim=embedding_dim,
            dropout=dropout,
            channel_dropout=channel_dropout,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be shaped (batch, 1, channels, time)")
        waveform = image[:, 0]
        log_scale = torch.log1p(
            waveform.float().std(dim=-1, unbiased=False)
        ).to(dtype=waveform.dtype)
        channel_mask = torch.ones(
            waveform.shape[:2],
            dtype=torch.bool,
            device=waveform.device,
        )
        return self.backbone(waveform, log_scale, channel_mask)


class SharedSpectralTemporalEncoder(nn.Module):
    """Encode one channel's log-power spectrum along STFT time frames."""

    def __init__(self, *, frequency_bins: int, embedding_dim: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(frequency_bins, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            SeparableTemporalBlock(64, dilation=1),
            SeparableTemporalBlock(64, dilation=2),
            SeparableTemporalBlock(64, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)
        self.scale_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        spectrum: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.temporal(self.stem(spectrum)).squeeze(-1)
        combined = self.projection(encoded) + self.scale_projection(
            log_scale[:, None]
        )
        return self.output_norm(combined)


class SharedChannelSTFTTCN(nn.Module):
    """Apply one spectral-temporal encoder to every EEG channel."""

    model_name = "m6_shared_channel_stft_tcn"
    model_version = "shared_channel_stft_tcn_v1"

    def __init__(
        self,
        *,
        sampling_frequency_hz: int = 256,
        n_fft: int = 256,
        win_length: int = 128,
        hop_length: int = 32,
        low_frequency_hz: float = 1.0,
        high_frequency_hz: float = 45.0,
        embedding_dim: int = 96,
        dropout: float = 0.25,
        channel_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= channel_dropout < 1.0:
            raise ValueError("channel_dropout must be in [0, 1)")
        if not 0 < win_length <= n_fft or hop_length < 1:
            raise ValueError("Invalid STFT frame configuration")
        nyquist = sampling_frequency_hz / 2.0
        if not 0 <= low_frequency_hz < high_frequency_hz <= nyquist:
            raise ValueError("Invalid STFT frequency range")
        first_bin = int(round(low_frequency_hz * n_fft / sampling_frequency_hz))
        last_bin = int(round(high_frequency_hz * n_fft / sampling_frequency_hz))
        if first_bin < 0 or last_bin <= first_bin or last_bin > n_fft // 2:
            raise ValueError("STFT frequency range has no valid bins")
        self.sampling_frequency_hz = int(sampling_frequency_hz)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.first_frequency_bin = first_bin
        self.last_frequency_bin = last_bin
        self.channel_dropout = float(channel_dropout)
        self.register_buffer(
            "stft_window",
            torch.hann_window(win_length, periodic=True),
            persistent=True,
        )
        frequency_bins = last_bin - first_bin + 1
        self.encoder = SharedSpectralTemporalEncoder(
            frequency_bins=frequency_bins,
            embedding_dim=embedding_dim,
        )
        self.attention_pool = GatedChannelAttention(embedding_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )

    def _training_mask(self, channel_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.channel_dropout == 0.0:
            return channel_mask
        keep = (
            torch.rand(channel_mask.shape, device=channel_mask.device)
            >= self.channel_dropout
        )
        keep &= channel_mask
        missing = ~keep.any(dim=1)
        if missing.any():
            first_valid = channel_mask.float().argmax(dim=1)
            keep[missing, first_valid[missing]] = True
        return keep

    def _spectrogram(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, timepoints = waveform.shape
        flat = waveform.reshape(batch * channels, timepoints)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                flat.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.stft_window.float(),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square()[
                :,
                self.first_frequency_bin : self.last_frequency_bin + 1,
            ]
            log_scale = torch.log1p(power.mean(dim=(1, 2)))
            log_power = torch.log1p(power)
            center = log_power.mean(dim=(1, 2), keepdim=True)
            scale = log_power.std(
                dim=(1, 2),
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)
            normalized = (log_power - center) / scale
        return normalized, log_scale

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be shaped (batch, 1, channels, time)")
        waveform = image[:, 0]
        if waveform.shape[-1] < self.n_fft:
            raise ValueError("EEG window is shorter than n_fft")
        batch, channels, _ = waveform.shape
        spectrum, log_scale = self._spectrogram(waveform)
        embeddings = self.encoder(spectrum, log_scale).reshape(
            batch,
            channels,
            -1,
        )
        channel_mask = torch.ones(
            (batch, channels),
            dtype=torch.bool,
            device=waveform.device,
        )
        active_mask = self._training_mask(channel_mask)
        attention_pool, _ = self.attention_pool(embeddings, active_mask)
        masked = embeddings.masked_fill(
            ~active_mask.unsqueeze(-1),
            float("-inf"),
        )
        max_pool = masked.max(dim=1).values
        return self.classifier(torch.cat([attention_pool, max_pool], dim=1))


class STFTEfficientNetVisualEncoder(nn.Module):
    """Convert multichannel EEG to a log-power image for EfficientNet-B0."""

    output_dim = 1280

    def __init__(
        self,
        *,
        pretrained: bool = True,
        eeg_channels: int = 18,
        image_size: int = 128,
        sampling_frequency_hz: int = 256,
        n_fft: int = 256,
        win_length: int = 128,
        hop_length: int = 32,
        low_frequency_hz: float = 1.0,
        high_frequency_hz: float = 45.0,
    ) -> None:
        super().__init__()
        if eeg_channels < 1 or image_size < 32:
            raise ValueError("Invalid STFT EfficientNet input dimensions")
        if not 0 < win_length <= n_fft or hop_length < 1:
            raise ValueError("Invalid STFT frame configuration")
        nyquist = sampling_frequency_hz / 2.0
        if not 0 <= low_frequency_hz < high_frequency_hz <= nyquist:
            raise ValueError("Invalid STFT frequency range")
        first_bin = int(round(low_frequency_hz * n_fft / sampling_frequency_hz))
        last_bin = int(round(high_frequency_hz * n_fft / sampling_frequency_hz))
        if first_bin < 0 or last_bin <= first_bin or last_bin > n_fft // 2:
            raise ValueError("STFT frequency range has no valid bins")

        weights_type, factory = _require_torchvision()
        weights = weights_type.DEFAULT if pretrained else None
        backbone = factory(weights=weights)
        first = backbone.features[0][0]
        if not isinstance(first, nn.Conv2d) or first.in_channels != 3:
            raise ValueError("Unexpected torchvision EfficientNet-B0 stem")
        replacement = nn.Conv2d(
            eeg_channels,
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
            mean_weight = first.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(
                mean_weight.repeat(1, eeg_channels, 1, 1)
                * (3.0 / eeg_channels)
            )
            if first.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(first.bias)
        backbone.features[0][0] = replacement

        self.eeg_channels = int(eeg_channels)
        self.image_size = int(image_size)
        self.sampling_frequency_hz = int(sampling_frequency_hz)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.first_frequency_bin = first_bin
        self.last_frequency_bin = last_bin
        self.register_buffer(
            "stft_window",
            torch.hann_window(win_length, periodic=True),
            persistent=True,
        )
        self.features = backbone.features

    def _spectrogram_image(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, channels, timepoints = waveform.shape
        if channels != self.eeg_channels:
            raise ValueError(
                f"Expected {self.eeg_channels} EEG channels, got {channels}"
            )
        flat = waveform.reshape(batch * channels, timepoints)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                flat.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.stft_window.float(),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square()[
                :,
                self.first_frequency_bin : self.last_frequency_bin + 1,
            ]
            log_power = torch.log1p(power).reshape(
                batch,
                channels,
                power.shape[-2],
                power.shape[-1],
            )
            center = log_power.mean(dim=(1, 2, 3), keepdim=True)
            scale = log_power.std(
                dim=(1, 2, 3),
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)
            normalized = ((log_power - center) / scale).clamp(-5.0, 5.0) / 5.0
            resized = nn.functional.interpolate(
                normalized,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return resized

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be shaped (batch, 1, channels, time)")
        waveform = image[:, 0]
        if waveform.shape[-1] < self.n_fft:
            raise ValueError("EEG window is shorter than n_fft")
        feature_map = self.features(self._spectrogram_image(waveform))
        if feature_map.ndim != 4 or feature_map.shape[1] != self.output_dim:
            raise ValueError(
                "Unexpected EfficientNet feature map: "
                f"{tuple(feature_map.shape)}"
            )
        return feature_map.flatten(2).transpose(1, 2).contiguous()


class STFTEfficientNetClassifier(EfficientNetLinearClassifier):
    model_name = "m7_stft_efficientnet_linear"
    model_version = "stft_efficientnet_b0_v1"

    def __init__(
        self,
        *,
        pretrained: bool = True,
        image_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            encoder=STFTEfficientNetVisualEncoder(
                pretrained=pretrained,
                image_size=image_size,
            ),
            pretrained=pretrained,
            dropout=dropout,
        )


class RGBRepeatEfficientNetVisualEncoder(nn.Module):
    """Keep the native RGB stem and repeat the single EEG image three times."""

    output_dim = 1280

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights_type, factory = _require_torchvision()
        weights = weights_type.DEFAULT if pretrained else None
        backbone = factory(weights=weights)
        self.features = backbone.features

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 4 or waveform.shape[1] != 1:
            raise ValueError("waveform must be shaped (batch, 1, channels, time)")
        feature_map = self.features(waveform.repeat(1, 3, 1, 1))
        if feature_map.ndim != 4 or feature_map.shape[1] != self.output_dim:
            raise ValueError(
                "Unexpected EfficientNet feature map: "
                f"{tuple(feature_map.shape)}"
            )
        return feature_map.flatten(2).transpose(1, 2).contiguous()


def build_s1_model(
    name: str,
    *,
    pretrained_encoder: bool = True,
    input_mode: str = "single_sum",
) -> nn.Module:
    if name == "m2":
        if input_mode != "single_sum":
            raise ValueError("M2 supports only the common single-channel input")
        return SharedChannelTCNImageClassifier()
    if name == "m3":
        if input_mode == "single_sum":
            encoder: nn.Module = EfficientNetVisualEncoder(
                pretrained=pretrained_encoder
            )
        elif input_mode == "rgb_repeat":
            encoder = RGBRepeatEfficientNetVisualEncoder(
                pretrained=pretrained_encoder
            )
        else:
            raise ValueError(f"Unknown S1 input mode: {input_mode}")
        return EfficientNetLinearClassifier(
            encoder=encoder,
            pretrained=pretrained_encoder,
        )
    if name == "m6":
        if input_mode != "single_sum":
            raise ValueError("M6 supports only the common single-channel input")
        return SharedChannelSTFTTCN()
    if name == "m7":
        if input_mode != "single_sum":
            raise ValueError("M7 supports only the common single-channel input")
        return STFTEfficientNetClassifier(pretrained=pretrained_encoder)
    raise ValueError(f"Unknown S1 model: {name}")


def describe_s1_model(model: nn.Module) -> dict[str, Any]:
    return {
        "model_name": str(getattr(model, "model_name", type(model).__name__)),
        "model_class": type(model).__name__,
        "s1_model_version": EEGVL_S1_MODEL_VERSION,
        "parameters": count_parameters(model),
    }
