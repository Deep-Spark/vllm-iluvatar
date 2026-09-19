# SPDX-License-Identifier: Apache-2.0
"""CPU tests for GLM-5.2 official-FP8 weight streaming helpers."""

import pytest
import torch

from vllm_iluvatar.models.glm_moe_dsa.fp8_model import dequant_indexer_wk

_WK = "model.layers.0.self_attn.indexer.wk.weight"
_SCALE = "model.layers.0.self_attn.indexer.wk.weight_scale_inv"


def _fp8_ones(n: int = 128, k: int = 256) -> torch.Tensor:
    return torch.ones(n, k, dtype=torch.float8_e4m3fn)


def _block_scale(n: int = 128, k: int = 256, value: float = 0.5) -> torch.Tensor:
    return torch.full(((n + 127) // 128, (k + 127) // 128), value, dtype=torch.float32)


def test_dequant_indexer_wk_folds_scale_in_either_order():
    weight = _fp8_ones()
    scale = _block_scale()
    other = torch.tensor(1.0)

    weight_first = dict(
        dequant_indexer_wk(
            [
                (_WK, weight),
                (_SCALE, scale),
                ("other", other),
            ]
        )
    )
    scale_first = dict(
        dequant_indexer_wk(
            [
                (_SCALE, scale),
                (_WK, weight),
                ("other", other),
            ]
        )
    )

    assert set(weight_first) == {_WK, "other"}
    assert set(scale_first) == {_WK, "other"}
    assert weight_first["other"] is other
    torch.testing.assert_close(weight_first[_WK], scale_first[_WK])
    torch.testing.assert_close(
        weight_first[_WK],
        torch.full((128, 256), 0.5, dtype=torch.bfloat16),
    )


def test_dequant_indexer_wk_passes_through_bf16_wk():
    bf16 = torch.ones(4, 4, dtype=torch.bfloat16)
    out = dict(dequant_indexer_wk([(_WK, bf16)]))
    assert out[_WK] is bf16


def test_dequant_indexer_wk_rejects_unpaired_scale():
    with pytest.raises(ValueError, match="unpaired indexer.wk"):
        list(dequant_indexer_wk([(_SCALE, _block_scale())]))
