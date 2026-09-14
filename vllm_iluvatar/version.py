# SPDX-License-Identifier: Apache-2.0
"""Expose the package version, which is defined in the repo-root version.json.

``version.json`` is the single source of truth mandated by the release spec, so
it must not be duplicated here. It ships outside the package though, and is
therefore unavailable once the wheel is installed -- in that case the version
is read back from the installed distribution metadata, which setup.py filled in
from the very same file at build time.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path

_VERSION_JSON = Path(__file__).resolve().parent.parent / "version.json"


def _resolve_version() -> str:
    try:
        return _distribution_version("vllm_iluvatar")
    except PackageNotFoundError:
        pass
    # Not installed: running straight from a source checkout.
    if _VERSION_JSON.is_file():
        return str(json.loads(_VERSION_JSON.read_text(encoding="utf-8"))["version"])
    raise RuntimeError(
        "cannot determine the vllm_iluvatar version: the package is not "
        f"installed and {_VERSION_JSON} is missing"
    )


__version__ = _resolve_version()


try:
    from ._git_id import __git_id__
except ImportError:
    __git_id__ = "unknown"


def format_identity() -> str:
    """Release version plus baked git id, for logs and collect-env."""
    if __git_id__ and __git_id__ != "unknown":
        return f"{__version__} {__git_id__}"
    return str(__version__)
