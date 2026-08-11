"""Channel-order invariant feature classifier with legacy checkpoint migration."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from .feature_schema import (
    BAND_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    checkpoint_band_schema_version,
)
from .evaluation_protocol import (
    EVALUATION_PROTOCOL_VERSION,
    PERSONALIZED_MODEL_PROTOCOL,
    validate_checkpoint_scopes,
)


N_FEATURES = 10
LEGACY_CHANNELS = 16
MODEL_TYPE = "channel_set_stats_v1"
VALIDATION_PROTOCOL = "patient_disjoint_v2"
GROUP_PROVENANCE = "patient_supergroup_v1"


def is_deployable_invariant_checkpoint(checkpoint: dict) -> bool:
    core_contract = (
        checkpoint.get("model_type") == MODEL_TYPE
        and checkpoint.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and checkpoint_band_schema_version(checkpoint) == BAND_SCHEMA_VERSION
        and checkpoint.get("deployment_approved") is True
        and checkpoint.get("simulation_only") is not True
    )
    if not core_contract:
        return False
    if checkpoint.get("evaluation_protocol") == EVALUATION_PROTOCOL_VERSION:
        return (
            checkpoint.get("model_protocol") == PERSONALIZED_MODEL_PROTOCOL
            and validate_checkpoint_scopes(checkpoint)
        )
    # Explicit compatibility path for the already approved P7 legacy artifact.
    return (
        checkpoint.get("validation_protocol") == VALIDATION_PROTOCOL
        and checkpoint.get("group_provenance") == GROUP_PROVENANCE
    )


class FeatureMLP(nn.Module):
    """Classify a variable-size set of channel feature vectors.

    Each sample is shaped ``(C, 10)``. Channel mean, population standard
    deviation and maximum are concatenated, so channel order has no effect.
    A batch may be a padded-free list of tensors with different channel counts.
    """

    model_type = MODEL_TYPE

    def __init__(self, n_features=N_FEATURES, hidden=128, nb_classes=2, dropout=0.3):
        super().__init__()
        in_dim = n_features * 3
        # Identity preserves legacy layer indices (first Linear remains net.1).
        self.net = nn.Sequential(
            nn.Identity(),
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, nb_classes),
        )

    @staticmethod
    def _aggregate_one(sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 2 or sample.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected one sample shaped (channels, {N_FEATURES}), "
                f"got {tuple(sample.shape)}"
            )
        if sample.shape[0] < 1:
            raise ValueError("A sample must contain at least one channel")
        mean = sample.mean(dim=0)
        std = sample.std(dim=0, correction=0)
        maximum = sample.max(dim=0).values
        return torch.cat([mean, std, maximum], dim=0)

    def aggregate(self, x: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.ndim == 2:
                return self._aggregate_one(x).unsqueeze(0)
            if x.ndim == 3:
                mean = x.mean(dim=1)
                std = x.std(dim=1, correction=0)
                maximum = x.max(dim=1).values
                return torch.cat([mean, std, maximum], dim=1)
            raise ValueError(f"Expected a 2D or 3D tensor, got {tuple(x.shape)}")

        samples = list(x)
        if not samples:
            raise ValueError("A batch must contain at least one sample")
        return torch.stack([self._aggregate_one(sample) for sample in samples])

    def forward(self, x: torch.Tensor | Iterable[torch.Tensor]):
        return self.net(self.aggregate(x))

    def count_params(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        return total, trainable


def migrate_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the legacy flattened 16x10 first layer to invariant mean pooling.

    Summing the 16 channel-specific weight blocks exactly preserves the legacy
    first-layer output when every channel has the same feature vector. The new
    std/max branches start at zero and can be learned during subsequent updates.
    """
    first_weight = state_dict.get("net.1.weight")
    if first_weight is None:
        raise ValueError("Checkpoint is missing net.1.weight")
    if first_weight.shape[1] == N_FEATURES * 3:
        return dict(state_dict)
    expected = LEGACY_CHANNELS * N_FEATURES
    if first_weight.shape[1] != expected:
        raise ValueError(
            f"Unsupported first-layer width {first_weight.shape[1]}; "
            f"expected {expected} (legacy) or {N_FEATURES * 3} (current)"
        )

    migrated = dict(state_dict)
    mean_weight = first_weight.reshape(
        first_weight.shape[0], LEGACY_CHANNELS, N_FEATURES
    ).sum(dim=1)
    migrated_weight = first_weight.new_zeros((first_weight.shape[0], N_FEATURES * 3))
    migrated_weight[:, :N_FEATURES] = mean_weight
    migrated["net.1.weight"] = migrated_weight
    return migrated


def load_feature_model_checkpoint(
    checkpoint: dict,
    *,
    device: torch.device | str = "cpu",
) -> tuple[FeatureMLP, bool]:
    """Build a current model from current or legacy checkpoint data."""
    state_dict = checkpoint["model_state_dict"]
    first_width = state_dict["net.1.weight"].shape[1]
    migrated = first_width == LEGACY_CHANNELS * N_FEATURES
    model = FeatureMLP().to(device)
    model.load_state_dict(migrate_legacy_state_dict(state_dict), strict=True)
    return model, migrated


# Kept as an explicit alias for older imports; new code must use FeatureMLP.
ChannelInvariantFeatureMLP = FeatureMLP
