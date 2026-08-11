"""Repository-wide filesystem configuration.

Machine-specific paths belong in environment variables or CLI arguments.  The
defaults below are stable project-relative locations and importing this module
never creates directories.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_path(variable: str, default: Path) -> Path:
    value = os.getenv(variable)
    path = Path(value).expanduser() if value else default
    return path.resolve()


DATA_ROOT = _configured_path("ASTAR_DATA_ROOT", PROJECT_ROOT / "data" / "clips")
CHECKPOINT_ROOT = _configured_path("ASTAR_CHECKPOINT_ROOT", PROJECT_ROOT)


def checkpoint_dir(version: str) -> Path:
    """Return the configured checkpoint directory for a model generation."""
    defaults = {
        "v0": PROJECT_ROOT / "v0" / "saved_models",
        "v1": PROJECT_ROOT / "ai" / "v1" / "checkpoints",
        "v2": PROJECT_ROOT / "ai" / "v2" / "checkpoints",
    }
    if version not in defaults:
        raise ValueError(f"Unknown model version: {version}")
    if os.getenv("ASTAR_CHECKPOINT_ROOT"):
        return (CHECKPOINT_ROOT / version).resolve()
    return defaults[version].resolve()


def resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a CLI path, using the repository root for relative values."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
