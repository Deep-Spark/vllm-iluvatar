# SPDX-License-Identifier: Apache-2.0
"""Tests for the FlyDSL jit_monitor compile hook."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.utils import jit_monitor

from vllm_iluvatar.backends.ixaite.patches import (
    jit_monitor as patch_jit_monitor_flydsl,
)


def _activate_without_other_backends(**kwargs):
    with (
        mock.patch.object(jit_monitor, "_setup_triton_autotuning_print"),
        mock.patch.object(jit_monitor, "_setup_triton_jit_hook"),
        mock.patch.object(jit_monitor, "_setup_cutedsl_jit_hook"),
        mock.patch.object(jit_monitor, "_setup_tilelang_jit_hook"),
    ):
        jit_monitor.activate(**kwargs)


@pytest.fixture
def flyc_compile():
    import flydsl.compiler as flyc

    inner = mock.Mock(return_value="compiled")
    orig = flyc.compile
    flyc.compile = inner
    jit_monitor._active = False
    jit_monitor._mode = "warn"
    jit_monitor._verbose = False
    jit_monitor._flydsl_hook_installed = False
    jit_monitor._flydsl_compile_count = 0
    yield inner
    flyc.compile = orig
    jit_monitor._active = False
    jit_monitor._mode = "warn"
    jit_monitor._verbose = False
    jit_monitor._flydsl_hook_installed = False
    jit_monitor._flydsl_compile_count = 0


def test_compile_before_activate_is_silent(flyc_compile):
    patch_jit_monitor_flydsl._apply()

    import flydsl.compiler as flyc

    flyc.compile(lambda: None, 1)
    flyc_compile.assert_called_once()
    assert getattr(jit_monitor, "_flydsl_compile_count", 0) == 0


def test_compile_after_activate_warns_with_shapes(flyc_compile):
    patch_jit_monitor_flydsl._apply()
    _activate_without_other_backends()

    import flydsl.compiler as flyc

    def _launch():
        return None

    _launch.__qualname__ = "compile_small_m.<locals>._launch"
    a = SimpleNamespace(shape=(16, 4096))
    w = SimpleNamespace(shape=(10240, 4096))
    out = SimpleNamespace(shape=(8, 10240))

    with mock.patch.object(jit_monitor.logger, "warning_once") as warning_once:
        result = flyc.compile(_launch, a, w, out, 8)

    assert result == "compiled"
    flyc_compile.assert_called_once()
    assert jit_monitor._flydsl_compile_count == 1
    warning_once.assert_called_once()
    args, _kwargs = warning_once.call_args
    assert args[1] == "FlyDSL"
    assert args[2] == "flyc.compile"
    assert "compile_small_m" in args[3]
    assert "operands=[(16, 4096), (10240, 4096), (8, 10240)]" in args[4]
    assert "scalars=[8]" in args[4]


def test_error_mode_raises_before_compile(flyc_compile):
    patch_jit_monitor_flydsl._apply()
    _activate_without_other_backends(mode="error")

    import flydsl.compiler as flyc

    with pytest.raises(RuntimeError, match="FlyDSL flyc.compile"):
        flyc.compile(lambda: None)

    flyc_compile.assert_not_called()
    assert jit_monitor._flydsl_compile_count == 1


def test_activate_is_idempotent(flyc_compile):
    patch_jit_monitor_flydsl._apply()
    patch_jit_monitor_flydsl._apply()
    _activate_without_other_backends()
    _activate_without_other_backends()

    import flydsl.compiler as flyc

    with mock.patch.object(jit_monitor.logger, "warning_once"):
        flyc.compile(lambda: None)
        flyc.compile(lambda: None)

    assert jit_monitor._flydsl_compile_count == 2
    assert flyc_compile.call_count == 2
