# SPDX-License-Identifier: Apache-2.0
"""Suite registry for ``python -m tests.ixaite.benchmarks``."""

from __future__ import annotations

from tests.ixaite.benchmarks.core.types import Suite

_SUITES: dict[str, Suite] = {}


def register(suite: Suite) -> Suite:
    """Register a suite instance under ``suite.name``."""
    name = suite.name
    if name in _SUITES:
        raise ValueError(f"suite {name!r} is already registered")
    _SUITES[name] = suite
    return suite


def get(name: str) -> Suite:
    """Return a registered suite or raise KeyError."""
    try:
        return _SUITES[name]
    except KeyError as error:
        known = ", ".join(sorted(_SUITES)) or "(none)"
        raise KeyError(f"unknown suite {name!r}; known: {known}") from error


def list_suites() -> list[str]:
    """Return registered suite names sorted."""
    return sorted(_SUITES)
