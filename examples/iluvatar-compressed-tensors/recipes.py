# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Locate paired, vendor-grouped profiles without changing quantization options."""

from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent

# Historical vendor/name identifiers, shared by both CLI path and recipe lookup.
LEGACY_RECIPES = {
    "deepseek/deepseek_r1": "deepseek/deepseek_r1_w4a8",
    "deepseek/deepseek_v32": "deepseek/deepseek_v3_2_w4a8",
    "deepseek/deepseek_v4": "deepseek/deepseek_v4_w4a8",
    "minimax/minimax_m25": "minimax/minimax_m2_5_w4a8",
    "minimax/minimax_m25_fp8": "minimax/minimax_m2_5_fp8",
    "minimax/minimax_m27": "minimax/minimax_m2_7_w4a8",
    "minimax/minimax_m3": "minimax/minimax_m3_w4a8",
    "moonshot/kimi_k25": "moonshot/kimi_k2_5_w4a8",
    "qwen/qwen36-35b-a3b": "qwen/qwen3_6_35b_a3b_w4a8",
    "qwen/qwen3_5": "qwen/qwen3_5_w4a8",
    "stepfun/step3p5_w4a8": "stepfun/step3_5_w4a8",
    "stepfun/step3p7_flash_fp8": "stepfun/step3_7_flash_fp8",
    "stepfun/step3p7_flash_w8a8": "stepfun/step3_7_flash_fp8_w8a8",
    "stepfun/step3p7_flash_w8a8_fp8_mix": "stepfun/step3_7_flash_mix_fp8_w8a8",
    "stepfun/step3p7_w4a8": "stepfun/step3_7_w4a8",
    "tencent/hy3": "tencent/hy3_w4a8",
    "zhipu/glm_52": "zhipu/glm5_2_w4a8",
}


def available_recipes() -> list[str]:
    """Return vendor/name identifiers with both a configs and rules document."""
    return sorted(
        path.relative_to(TOOL_DIR / "configs").with_suffix("").as_posix()
        for path in (TOOL_DIR / "configs").glob("*/*.json")
        if (TOOL_DIR / "rules" / path.relative_to(TOOL_DIR / "configs")).is_file()
    )


def canonical_recipe_name(recipe: str) -> str:
    """Normalize historical names before argparse validates canonical choices."""
    return LEGACY_RECIPES.get(recipe, recipe)


def recipe_paths(recipe: str) -> tuple[Path, Path]:
    recipe = canonical_recipe_name(recipe)
    if recipe not in available_recipes():
        raise ValueError(f"Unknown recipe: {recipe!r}")
    relative = Path(recipe + ".json")
    return TOOL_DIR / "configs" / relative, TOOL_DIR / "rules" / relative


def legacy_profile_path(value: str, kind: str) -> Path:
    """Resolve old built-in flat or vendor paths without redirecting custom files."""
    path = Path(value)
    root = TOOL_DIR / kind
    if path.exists():
        return path
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return path
    if relative.suffix != ".json":
        return path
    if len(relative.parts) == 2:
        name = canonical_recipe_name(relative.with_suffix("").as_posix())
        target = root / (name + ".json")
        return target if target.is_file() else path
    if len(relative.parts) != 1:
        return path

    # Deduplicate canonical and historical matches pointing to the same profile.
    candidates = {
        p.relative_to(root).with_suffix("").as_posix()
        for p in root.glob("*/*.json")
        if p.stem == relative.stem
    }
    candidates.update(
        new
        for old, new in LEGACY_RECIPES.items()
        if Path(old).name == relative.stem and (root / (new + ".json")).is_file()
    )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous legacy profile {value!r}; use a vendor path")
    return root / (next(iter(candidates)) + ".json") if candidates else path
