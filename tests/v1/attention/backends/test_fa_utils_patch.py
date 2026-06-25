# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fa_utils general patch (capability overrides).

The core test (propagating rebind) is pure Python and needs neither a GPU nor
ixformer: it is the highest-value check because the community attention modules
bind these helpers via ``from fa_utils import <name>`` (so a plain ``setattr``
on the fa_utils module would miss them).

The behavioural checks for the ILU override values run against the impl module
with a fake ``vllm.platforms`` injected, so they too stay CPU-only.
"""

from __future__ import annotations

import sys
import types

import pytest


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def test_rebind_module_func_propagates_to_from_import_refs():
    """rebind must update both the source module and already-copied refs,
    while leaving homonymous-but-distinct symbols untouched (identity match)."""
    from vllm_iluvatar.ilu_patch import general_patch as p

    names = [
        "ilu_fake_fa_pkg",
        "ilu_fake_fa_pkg.src",
        "ilu_fake_fa_pkg.consumer",
        "ilu_fake_fa_pkg.sibling",
    ]
    _make_module("ilu_fake_fa_pkg")
    src = _make_module("ilu_fake_fa_pkg.src")
    consumer = _make_module("ilu_fake_fa_pkg.consumer")
    sibling = _make_module("ilu_fake_fa_pkg.sibling")
    try:

        def original():
            return "orig"

        def other():
            return "other"

        src.func = original
        # consumer did ``from src import func`` before the patch.
        consumer.func = original
        # sibling has a homonymous but DISTINCT object -> must not be touched.
        sibling.func = other

        def replacement():
            return "ilu"

        p.rebind_module_func(src, "func", replacement)

        assert src.func is replacement
        assert consumer.func is replacement  # propagated to copied reference
        assert sibling.func is other  # identity-safe: untouched
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_rebind_module_func_injects_new_symbol():
    """When the symbol does not exist yet, rebind just setattr's it (no walk)."""
    from vllm_iluvatar.ilu_patch import general_patch as p

    src = _make_module("ilu_fake_fa_pkg2.src")
    try:

        def added():
            return 1

        p.rebind_module_func(src, "added", added)
        assert src.added is added
    finally:
        sys.modules.pop("ilu_fake_fa_pkg2.src", None)


def test_ilu_override_values_are_unconditional():
    """The overrides deliberately decouple from FA version dispatch:
    get_flash_attn_version -> None and flash_attn_supports_sinks -> True,
    unconditionally (no platform/vllm import needed)."""
    import importlib

    impl = importlib.import_module("vllm_iluvatar.v1.attention.backends.fa_utils")
    assert impl.get_flash_attn_version() is None
    assert impl.get_flash_attn_version(has_sinks=True, head_size=128) is None
    assert impl.flash_attn_supports_sinks() is True


def test_vllm_flash_attn_shim_is_importable_without_ixformer():
    """The import_shims stand-in must build an importable module even when
    ixformer is absent (flash_attn_varlen_func degrades to a lazy-raising
    placeholder; get_scheduler_metadata returns None)."""
    from vllm_iluvatar.ilu_patch.import_shims import vllm_flash_attn as shim

    module = shim.build_module()
    assert shim.SHIM_KEY == "vllm.vllm_flash_attn"
    assert hasattr(module, "flash_attn_varlen_func")
    assert module.get_scheduler_metadata() is None
    try:
        from ixformer.contrib.vllm_flash_attn import (  # noqa: F401
            flash_attn_varlen_func,
        )

        has_ixformer = True
    except Exception:
        has_ixformer = False

    if not has_ixformer:
        with pytest.raises(RuntimeError):
            module.flash_attn_varlen_func()
