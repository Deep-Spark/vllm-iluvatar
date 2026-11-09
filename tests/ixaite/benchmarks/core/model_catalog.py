# SPDX-License-Identifier: Apache-2.0
"""Repository model profiles shared by L2 and end-to-end benchmarks."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _profile_paths() -> dict[str, Path]:
    profiles: dict[str, Path] = {}
    for path in sorted(_MODEL_DIR.glob("*.yaml")):
        slug = path.stem
        if slug in profiles:
            raise ValueError(f"duplicate benchmark model profile {slug!r}")
        profiles[slug] = path
    return profiles


def list_models() -> list[str]:
    """Return model slugs discovered from repository profiles."""
    return sorted(_profile_paths())


def list_e2e_models() -> list[str]:
    """Return profiles that define an end-to-end benchmark or correctness probe."""
    return [
        model for model in list_models() if load_model(model).get("e2e") is not None
    ]


@cache
def load_model(slug: str) -> dict[str, Any]:
    """Load and minimally validate one repository model profile."""
    paths = _profile_paths()
    try:
        path = paths[slug]
    except KeyError as error:
        known = ", ".join(sorted(paths)) or "(none)"
        raise KeyError(f"unknown benchmark model {slug!r}; known: {known}") from error

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: profile must be an object")
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if data.get("slug") != slug:
        raise ValueError(f"{path}: slug must match filename {slug!r}")
    if not isinstance(data.get("l2"), dict):
        raise ValueError(f"{path}: l2 must be an object")
    e2e = data.get("e2e")
    if e2e is not None:
        if not isinstance(e2e, dict):
            raise ValueError(f"{path}: e2e must be an object or null")
        ops = e2e.get("ops")
        if not isinstance(ops, list) or not ops:
            raise ValueError(f"{path}: e2e.ops must be a non-empty list")
        if not all(isinstance(op, str) and op for op in ops):
            raise ValueError(f"{path}: e2e.ops must be a list of non-empty strings")
    return data


def l2_spec(model: str, suite: str) -> dict[str, Any]:
    """Return one suite's model specification."""
    specs = load_model(model)["l2"]
    try:
        spec = specs[suite]
    except KeyError as error:
        raise KeyError(f"no {suite} specs for model {model!r}") from error
    if not isinstance(spec, dict):
        raise ValueError(f"l2.{suite} for model {model!r} must be an object")
    return spec


def e2e_spec(model: str) -> dict[str, Any]:
    """Return the e2e section for one model."""
    spec = load_model(model).get("e2e")
    if not isinstance(spec, dict):
        raise ValueError(f"model {model!r} is L2-only and has no e2e profile")
    return spec


def e2e_ops(model: str) -> tuple[str, ...]:
    """Return plugin op names this model's serve path is expected to cover."""
    return tuple(e2e_spec(model)["ops"])


def listed_e2e_ops() -> set[str]:
    """Return the union of ``e2e.ops`` across discovered model profiles."""
    ops: set[str] = set()
    for model in list_e2e_models():
        ops.update(e2e_ops(model))
    return ops
