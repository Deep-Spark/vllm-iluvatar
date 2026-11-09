# SPDX-License-Identifier: Apache-2.0
"""Plugin-facing checks for ixaite linear TP-local weight binding."""

from __future__ import annotations

import importlib

import pytest
import torch


def _load_ixaite_linear():
    import ixaite

    ixaite.initialize("ivcore11")
    return importlib.import_module("ixaite.ops.linear")


def test_tp4_qkv_weight_shape_binds_ixaite_linear() -> None:
    op = _load_ixaite_linear()
    n, k = 2560, 4096
    config = op._select_config(32, n, k, torch.bfloat16)
    assert config.kernel_id in {"small_m", "hgemm", "narrow_n_gemv"}
    assert config.kernel_id != "naive"


def test_tp2_vision_qkv_weight_shape_binds_ixaite_linear() -> None:
    op = _load_ixaite_linear()
    config = op._select_config(32, 1728, 1152, torch.bfloat16)
    assert config.kernel_id == "hgemm"
    assert config.kernel_id != "naive"


def test_tp4_vision_qkv_weight_shape_binds_ixaite_linear() -> None:
    op = _load_ixaite_linear()
    config = op._select_config(32, 864, 1152, torch.bfloat16)
    assert config.kernel_id == "hgemm"


def test_tp4_row_parallel_projection_shape_binds_ixaite_linear() -> None:
    op = _load_ixaite_linear()
    config = op._select_config(32, 4096, 1024, torch.bfloat16)
    assert config.kernel_id == "hgemm"


def test_unsupported_linear_shape_surfaces_from_ixaite() -> None:
    op = _load_ixaite_linear()
    with pytest.raises(ValueError, match="unsupported Qwen3.5 linear"):
        op._select_config(4, 32, 64)
