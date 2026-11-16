# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the offline e8m0 block dequantization layout dispatch.

``fp8_block_e8m0`` serves two incompatible scale layouts:

* 1-D, one scale per ``block_size`` elements of the last dim only
  (HY4 MXFP8, ``[N, K/32]``), and
* 2-D ``block_size x block_size`` tiles (DeepSeek-V4, ``[N/128, K/128]``).

Running a 1-D scale through the 2-D path reshapes cleanly but applies row
``r // block_size``'s exponent to row ``r``, so only 1 row in ``block_size``
keeps its own scale. That silently corrupted a whole W4A8 conversion, hence
these tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_METHODS_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "iluvatar-compressed-tensors"
    / "methods.py"
)


def _load_methods():
    spec = importlib.util.spec_from_file_location(
        "ilu_ct_methods_for_test", _METHODS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


methods = _load_methods()


def _e8m0(exponents: torch.Tensor) -> torch.Tensor:
    """Encode integer exponents as e8m0 bytes (value == 2 ** exponent)."""
    return (exponents + 127).to(torch.uint8)


def test_one_dim_layout_dispatches_to_block_dequant_1d():
    weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    scale = _e8m0(torch.tensor([[0, 1], [1, 0], [2, 0], [0, 2]]))

    got = methods.DEQUANT_METHODS["fp8_block_e8m0"](weight, scale, {"block_size": 4})
    want = methods._block_dequant_1d(weight, scale, block_size=4, decode_e8m0=True)
    assert torch.equal(got, want)

    # Every row must use its own scale, not row // block_size's.
    expected = weight * torch.tensor(
        [
            [1.0] * 4 + [2.0] * 4,
            [2.0] * 4 + [1.0] * 4,
            [4.0] * 4 + [1.0] * 4,
            [1.0] * 4 + [4.0] * 4,
        ]
    )
    assert torch.equal(got, expected)


def test_two_dim_layout_dispatches_to_block_dequant():
    weight = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    scale = _e8m0(torch.tensor([[0, 1], [2, 0]]))

    got = methods.DEQUANT_METHODS["fp8_block_e8m0"](weight, scale, {"block_size": 4})
    want = methods._block_dequant(weight, scale, block_size=4, decode_e8m0=True)
    assert torch.equal(got, want)

    expected = weight.clone()
    expected[:4, 4:] *= 2.0
    expected[4:, :4] *= 4.0
    assert torch.equal(got, expected)


def test_two_dim_path_rejects_a_one_dim_scale():
    """The exact shape that silently corrupted the HY4 W4A8 conversion."""
    weight = torch.zeros(4096, 6144, dtype=torch.float32)
    scale = torch.zeros(4096, 192, dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"2-D block dequant expects a scale"):
        methods._block_dequant(weight, scale, block_size=32, decode_e8m0=True)


def test_one_dim_path_rejects_a_two_dim_scale():
    weight = torch.zeros(1024, 4096, dtype=torch.float32)
    scale = torch.zeros(8, 32, dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"1-D block dequant"):
        methods._block_dequant_1d(weight, scale, block_size=128, decode_e8m0=True)


def test_fused_moe_three_dim_dispatches_per_expert():
    weight = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    scale = _e8m0(torch.zeros(2, 4, 2, dtype=torch.long))

    got = methods.DEQUANT_METHODS["fp8_block_e8m0"](weight, scale, {"block_size": 4})
    assert got.shape == weight.shape
    assert torch.equal(got, weight)


def test_two_dim_path_still_allows_ragged_last_block():
    weight = torch.ones(300, 500, dtype=torch.float32)
    scale = torch.ones(3, 4, dtype=torch.float32)

    got = methods._block_dequant(weight, scale, block_size=128, decode_e8m0=False)
    assert got.shape == (300, 500)
    assert torch.equal(got, weight)
