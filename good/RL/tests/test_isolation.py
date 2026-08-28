from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "eeg_alarm_policy"
FORBIDDEN_TOP_LEVEL_IMPORTS = {"ai", "good", "verl"}


def test_core_package_has_no_astar_or_verl_imports() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", maxsplit=1)[0] in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    violations.append(f"{path.name}:{node.lineno}: {name}")

    assert violations == []
