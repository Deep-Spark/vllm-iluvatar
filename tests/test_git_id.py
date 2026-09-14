"""Installed-package git identity baked at wheel build time."""

import re

from vllm_iluvatar.version import __git_id__

_SELF_ID_RE = re.compile(r"^\[.+\] [0-9a-f]{40}$")


def test_git_id_defined():
    assert __git_id__ is not None
    if __git_id__ == "unknown":
        return
    assert _SELF_ID_RE.match(__git_id__), __git_id__
