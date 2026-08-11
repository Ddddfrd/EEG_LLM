"""Permutation-invariant raw EEG model with shared temporal encoding."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


RAW_TCN_MODEL_VERSION = "shared_channel_multiscale_tcn_attention_v1"


class SeparableTemporalBlock(nn.Module):
    def __init__(self, channels: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = 4 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=9,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if stride == 1
            else nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        encoded = self.pointwise(self.depthwise(values))
        return self.activation(self.norm(encoded) + residual)


class SharedTemporalEncoder(nn.Module):
    def __init__(self, *, branch_width: int = 16, embedding_dim: int = 96) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        1,
                        branch_width,
                        kernel_size=kernel,
                        stride=2,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(4, branch_width),
                    nn.GELU(),
                )
                for kernel in (7, 25, 75)
            ]
        )
        channels = branch_width * len(self.branches)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            SeparableTemporalBlock(64, stride=2, dilation=1),
            SeparableTemporalBlock(64, stride=2, dilation=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)
        self.scale_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, waveform: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        branches = [branch(waveform) for branch in self.branches]
        encoded = self.fusion(torch.cat(branches, dim=1)).squeeze(-1)
        combined = self.projection(encoded) + self.scale_projection(log_scale[:, None])
        return self.output_norm(combined)


class GatedChannelAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Linear(embedding_dim, embedding_dim)
        self.score = nn.Linear(embedding_dim, 1, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if channel_mask.ndim != 2 or channel_mask.shape != embeddings.shape[:2]:
            raise ValueError("channel_mask must match the first two embedding dimensions")
        if not channel_mask.any(dim=1).all():
            raise ValueError("Every sample requires at least one valid channel")
        scores = self.score(
            torch.tanh(self.value(embeddings)) * torch.sigmoid(self.gate(embeddings))
        ).squeeze(-1)
        scores = scores.masked_fill(~channel_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * embeddings, dim=1)
        return pooled, attention


class SharedChannelTCN(nn.Module):
    """Encode each channel identically and pool channels without order dependence."""

    model_version = RAW_TCN_MODEL_VERSION

    def __init__(
        self,
        *,
        embedding_dim: int = 96,
        dropout: float = 0.25,
        channel_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= channel_dropout < 1.0:
            raise ValueError("channel_dropout must be in [0, 1)")
        self.channel_dropout = channel_dropout
        self.encoder = SharedTemporalEncoder(embedding_dim=embedding_dim)
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
        keep = torch.rand(channel_mask.shape, device=channel_mask.device) >= self.channel_dropout
        keep &= channel_mask
        missing = ~keep.any(dim=1)
        if missing.any():
            first_valid = channel_mask.float().argmax(dim=1)
            keep[missing, first_valid[missing]] = True
        return keep

    def forward(
        self,
        waveform: torch.Tensor,
        log_scale: torch.Tensor,
        channel_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if waveform.ndim != 3:
            raise ValueError("waveform must be shaped (batch, channels, time)")
        if log_scale.shape != waveform.shape[:2]:
            raise ValueError("log_scale must be shaped (batch, channels)")
        if channel_mask.shape != waveform.shape[:2]:
            raise ValueError("channel_mask must be shaped (batch, channels)")
        batch, channels, timepoints = waveform.shape
        flat_waveform = waveform.reshape(batch * channels, 1, timepoints)
        flat_scale = log_scale.reshape(batch * channels)
        embeddings = self.encoder(flat_waveform, flat_scale).reshape(
            batch, channels, -1
        )
        active_mask = self._training_mask(channel_mask)
        attention_pool, attention = self.attention_pool(embeddings, active_mask)
        masked = embeddings.masked_fill(~active_mask.unsqueeze(-1), float("-inf"))
        max_pool = masked.max(dim=1).values
        logits = self.classifier(torch.cat([attention_pool, max_pool], dim=1))
        if return_attention:
            return logits, {"attention": attention, "active_channel_mask": active_mask}
        return logits

    def count_parameters(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }
