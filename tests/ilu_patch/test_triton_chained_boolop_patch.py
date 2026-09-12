# SPDX-License-Identifier: Apache-2.0
"""Tests for the narrowly scoped CoreX Triton boolean-chain patch."""

from __future__ import annotations

import ast
from types import SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_triton_chained_boolop


def test_rewrite_splits_chained_boolean_operator() -> None:
    jit_fn = SimpleNamespace(
        src="def kernel(a, b, c):\n    value = a or b or c\n    return value\n",
        hash="old",
    )

    assert patch_triton_chained_boolop._rewrite_kernel(jit_fn) == 1
    assert jit_fn.hash is None
    tree = ast.parse(jit_fn.src)
    outer = tree.body[0].body[0].value
    assert isinstance(outer, ast.BoolOp)
    assert len(outer.values) == 2
    assert isinstance(outer.values[0], ast.BoolOp)
    assert len(outer.values[0].values) == 2
    assert patch_triton_chained_boolop._rewrite_kernel(jit_fn) == 0


def test_patch_is_limited_to_sampler_penalties() -> None:
    assert (
        patch_triton_chained_boolop._TARGET_MODULE
        == "vllm.v1.worker.gpu.sample.penalties"
    )
    assert patch_triton_chained_boolop._TARGET_ATTR == "_penalties_kernel"
