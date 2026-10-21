# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Locate paired, vendor-grouped profiles without changing quantization options."""

from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent


def available_recipes() -> list[str]:
    """Return vendor/name identifiers with both a configs and rules document."""
    return sorted(
        path.relative_to(TOOL_DIR / "configs").with_suffix("").as_posix()
        for path in (TOOL_DIR / "configs").glob("*/*.json")
        if (TOOL_DIR / "rules" / path.relative_to(TOOL_DIR / "configs")).is_file()
    )


def recipe_paths(recipe: str) -> tuple[Path, Path]:
    if recipe not in available_recipes():
        raise ValueError(f"Unknown recipe: {recipe!r}")
    relative = Path(recipe + ".json")
    return TOOL_DIR / "configs" / relative, TOOL_DIR / "rules" / relative


def legacy_profile_path(value: str, kind: str) -> Path:
    """Resolve a moved built-in flat path; leave custom paths untouched.

    Only paths immediately under this tool's configs/ or rules/ qualify.
    A real file always wins. An ambiguous basename requires an explicit vendor.
    """
    path = Path(value)
    root = TOOL_DIR / kind
    if path.exists() or path.parent.resolve() != root:
        return path
    matches = [p for p in root.glob("*/*.json") if p.name == path.name]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous legacy profile {value!r}; use a vendor path")
    return matches[0] if matches else path
