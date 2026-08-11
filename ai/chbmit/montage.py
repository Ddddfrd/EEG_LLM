"""Resolve heterogeneous CHB-MIT signal labels to a canonical bipolar montage."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .contracts import CANONICAL_BIPOLAR_CHANNELS


_ELECTRODE_ALIASES = {"01": "O1"}
_IGNORED_LABELS = {"", "-", ".", "--"}
_UNIT_TO_MICROVOLTS = {
    "uv": 1.0,
    "mv": 1_000.0,
    "v": 1_000_000.0,
    "nv": 0.001,
}


class EdfSignalReader(Protocol):
    def readSignal(
        self, signalnum: int, start: int = 0, n: int | None = None, digital: bool = False
    ) -> np.ndarray: ...

    def getPhysicalDimension(self, channel: int) -> str: ...


@dataclass(frozen=True)
class MontageTerm:
    source_index: int
    source_label: str
    coefficient: float


@dataclass(frozen=True)
class MontageRecipe:
    target_label: str
    mode: str
    terms: tuple[MontageTerm, ...]


def normalize_electrode(value: str) -> str:
    normalized = value.strip().upper()
    return _ELECTRODE_ALIASES.get(normalized, normalized)


def normalize_signal_label(value: str) -> str:
    normalized = value.strip().upper().replace("–", "-").replace("—", "-")
    if normalized.startswith("EEG "):
        normalized = normalized[4:].strip()
    if normalized in _IGNORED_LABELS:
        return ""
    if "-" not in normalized:
        return normalize_electrode(normalized)
    left, right = normalized.split("-", maxsplit=1)
    left = normalize_electrode(left)
    right = normalize_electrode(right)
    if not left or not right:
        return ""
    return f"{left}-{right}"


def _split_bipolar(label: str) -> tuple[str, str] | None:
    if not label or "-" not in label:
        return None
    left, right = label.split("-", maxsplit=1)
    if left in _IGNORED_LABELS or right in _IGNORED_LABELS:
        return None
    return left, right


def _combine_terms(terms: Sequence[MontageTerm]) -> tuple[MontageTerm, ...]:
    coefficients: dict[int, float] = {}
    labels: dict[int, str] = {}
    order: list[int] = []
    for term in terms:
        if term.source_index not in coefficients:
            order.append(term.source_index)
            coefficients[term.source_index] = 0.0
            labels[term.source_index] = term.source_label
        coefficients[term.source_index] += term.coefficient
    return tuple(
        MontageTerm(index, labels[index], coefficients[index])
        for index in order
        if not np.isclose(coefficients[index], 0.0)
    )


def build_montage_recipes(
    signal_labels: Sequence[str],
    targets: Sequence[str] = CANONICAL_BIPOLAR_CHANNELS,
) -> tuple[MontageRecipe, ...]:
    """Build reproducible recipes for direct, common-reference, or monopolar signals."""
    normalized = [normalize_signal_label(label) for label in signal_labels]
    direct: dict[str, int] = {}
    monopolar: dict[str, int] = {}
    adjacency: dict[str, list[tuple[str, MontageTerm]]] = {}
    for index, label in enumerate(normalized):
        if not label:
            continue
        pair = _split_bipolar(label)
        if pair is None:
            monopolar.setdefault(label, index)
            continue
        direct.setdefault(label, index)
        left, right = pair
        forward = MontageTerm(index, signal_labels[index].strip(), 1.0)
        reverse = MontageTerm(index, signal_labels[index].strip(), -1.0)
        adjacency.setdefault(left, []).append((right, forward))
        adjacency.setdefault(right, []).append((left, reverse))

    recipes: list[MontageRecipe] = []
    for raw_target in targets:
        target = normalize_signal_label(raw_target)
        pair = _split_bipolar(target)
        if pair is None:
            raise ValueError(f"Target is not bipolar: {raw_target}")
        if target in direct:
            index = direct[target]
            recipes.append(
                MontageRecipe(
                    target_label=target,
                    mode="direct",
                    terms=(MontageTerm(index, signal_labels[index].strip(), 1.0),),
                )
            )
            continue

        left, right = pair
        queue: deque[tuple[str, tuple[MontageTerm, ...]]] = deque([(left, ())])
        visited = {left}
        path_terms: tuple[MontageTerm, ...] | None = None
        while queue:
            node, terms = queue.popleft()
            if node == right:
                path_terms = _combine_terms(terms)
                break
            for neighbor, term in adjacency.get(node, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, (*terms, term)))
        if path_terms is not None:
            recipes.append(
                MontageRecipe(
                    target_label=target,
                    mode="bipolar_reconstruction",
                    terms=path_terms,
                )
            )
            continue

        if left in monopolar and right in monopolar:
            left_index = monopolar[left]
            right_index = monopolar[right]
            recipes.append(
                MontageRecipe(
                    target_label=target,
                    mode="referential_reconstruction",
                    terms=(
                        MontageTerm(
                            left_index, signal_labels[left_index].strip(), 1.0
                        ),
                        MontageTerm(
                            right_index, signal_labels[right_index].strip(), -1.0
                        ),
                    ),
                )
            )
            continue
        raise ValueError(
            f"Cannot reconstruct {target} from labels: {list(signal_labels)}"
        )
    return tuple(recipes)


def unit_scale_to_microvolts(unit: str) -> float:
    normalized = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_MICROVOLTS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported EDF physical unit: {unit!r}") from exc


def read_montage_window(
    reader: EdfSignalReader,
    recipes: Sequence[MontageRecipe],
    *,
    start_sample: int,
    sample_count: int,
) -> np.ndarray:
    """Read one canonical window in microvolts with shape ``(channels, samples)``."""
    if start_sample < 0 or sample_count < 1:
        raise ValueError("Invalid EDF sample range")
    source_indices = sorted({
        term.source_index for recipe in recipes for term in recipe.terms
    })
    sources: dict[int, np.ndarray] = {}
    for index in source_indices:
        values = np.asarray(
            reader.readSignal(index, start=start_sample, n=sample_count),
            dtype=np.float32,
        )
        if values.shape != (sample_count,):
            raise ValueError(
                f"EDF signal {index} returned {values.shape}, expected {(sample_count,)}"
            )
        sources[index] = values * unit_scale_to_microvolts(
            reader.getPhysicalDimension(index)
        )
    output = np.empty((len(recipes), sample_count), dtype=np.float32)
    for channel, recipe in enumerate(recipes):
        output[channel] = sum(
            term.coefficient * sources[term.source_index] for term in recipe.terms
        )
    return output
