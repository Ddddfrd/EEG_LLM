"""Experimental direct 20-channel CHB-MIT montage for Scheme C comparisons."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Mapping

from .contracts import CANONICAL_BIPOLAR_CHANNELS
from .index import canonical_hash
from .montage import MontageRecipe, build_montage_recipes


DIRECT20_EXTRA_CHANNELS = ("T7-FT9", "FT10-T8")
DIRECT20_CHANNELS = (*CANONICAL_BIPOLAR_CHANNELS, *DIRECT20_EXTRA_CHANNELS)


def _recipe_payload(recipe: MontageRecipe) -> dict[str, Any]:
    return {
        "target_label": recipe.target_label,
        "mode": recipe.mode,
        "terms": [
            {
                "source_index": term.source_index,
                "source_label": term.source_label,
                "coefficient": term.coefficient,
            }
            for term in recipe.terms
        ],
    }


def build_direct20_index(index: Mapping[str, Any]) -> dict[str, Any]:
    """Return an index with two real temporal channels and explicit missing zeros.

    CHB-MIT has only 18 bipolar channels that are reconstructable in every EDF.
    T7-FT9 and FT10-T8 are present in 655/686 records. Missing channels are zero
    filled instead of being replaced by duplicated canonical channels.
    """
    body = copy.deepcopy({
        key: value for key, value in index.items() if key != "index_sha256"
    })
    body["target_montage"] = list(DIRECT20_CHANNELS)
    availability: Counter[str] = Counter()
    missing_by_subject: Counter[str] = Counter()
    for record in body["records"]:
        recipes: list[dict[str, Any]] = []
        modes: Counter[str] = Counter()
        for target in DIRECT20_CHANNELS:
            try:
                recipe = build_montage_recipes(
                    record["signal_labels"],
                    targets=(target,),
                )[0]
                payload = _recipe_payload(recipe)
                availability[target] += 1
            except ValueError:
                if target not in DIRECT20_EXTRA_CHANNELS:
                    raise
                payload = {
                    "target_label": target,
                    "mode": "missing_zero",
                    "terms": [],
                }
                missing_by_subject[str(record["subject_id"])] += 1
            recipes.append(payload)
            modes[str(payload["mode"])] += 1
        record["montage"] = recipes
        record["montage_modes"] = dict(sorted(modes.items()))
    body["direct20_contract"] = {
        "channels": list(DIRECT20_CHANNELS),
        "extra_channels": list(DIRECT20_EXTRA_CHANNELS),
        "missing_policy": "zero_fill",
        "availability_by_channel_records": dict(sorted(availability.items())),
        "missing_extra_channel_records_by_subject": dict(
            sorted(missing_by_subject.items())
        ),
        "status": "experimental_cross_patient_montage",
    }
    return {**body, "index_sha256": canonical_hash(body)}


__all__ = ["DIRECT20_CHANNELS", "DIRECT20_EXTRA_CHANNELS", "build_direct20_index"]
