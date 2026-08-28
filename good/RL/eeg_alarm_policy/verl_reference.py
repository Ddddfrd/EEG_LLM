"""Load unmodified verl-agent advantage functions with minimal import stubs."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    mean = selected.mean()
    variance = selected.var(unbiased=False)
    return (values - mean) * torch.rsqrt(variance + 1e-8)


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_modules(
    root: Path | None = None,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Return core_algos and core_gigpo without importing the full verl runtime."""
    source_root = root or Path(__file__).resolve().parents[1] / "verl-agent-master"
    core_path = source_root / "verl" / "trainer" / "ppo" / "core_algos.py"
    gigpo_path = source_root / "gigpo" / "core_gigpo.py"
    if not core_path.is_file() or not gigpo_path.is_file():
        raise FileNotFoundError("Vendored verl-agent reference sources are unavailable")

    saved = {
        name: sys.modules.get(name)
        for name in ("verl", "verl.utils", "verl.utils.torch_functional")
    }
    verl = types.ModuleType("verl")
    verl.__path__ = []
    verl.DataProto = type("DataProto", (), {})
    utils = types.ModuleType("verl.utils")
    utils.__path__ = []
    functional = types.ModuleType("verl.utils.torch_functional")
    functional.masked_whiten = _masked_whiten
    verl.utils = utils
    utils.torch_functional = functional
    sys.modules["verl"] = verl
    sys.modules["verl.utils"] = utils
    sys.modules["verl.utils.torch_functional"] = functional
    try:
        core = _load_module("_eeg_g0_verl_core_algos", core_path)
        gigpo = _load_module("_eeg_g0_verl_core_gigpo", gigpo_path)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return core, gigpo


def reference_source_metadata(root: Path | None = None) -> dict[str, dict[str, Any]]:
    source_root = root or Path(__file__).resolve().parents[1] / "verl-agent-master"
    paths = {
        "grpo_rloo": source_root / "verl" / "trainer" / "ppo" / "core_algos.py",
        "gigpo": source_root / "gigpo" / "core_gigpo.py",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }
