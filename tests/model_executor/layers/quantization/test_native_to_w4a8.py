# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_iluvatar.model_executor.layers.quantization.backends.native.moe import (
    _pack_weight,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["NN", "TN"])
def test_tp_down_projection_preserves_full_k_channel_scale(dtype, layout):
    # Rank 0 sees only a small value. Recomputing its scale locally would
    # quantize it to code 7, while a full-K scale rounds it to code 0.
    weight = torch.ones(32, 64, dtype=dtype)
    weight[:, 32:] = 127
    weight[0].zero_()

    first, first_scale = _pack_weight(weight, layout, k_start=0, k_size=32)
    second, second_scale = _pack_weight(weight, layout, k_start=32, k_size=32)
    full, full_scale = _pack_weight(weight, layout)
    assert torch.count_nonzero(first).item() == 0
    assert torch.count_nonzero(second).item() > 0
    assert torch.equal(first_scale, torch.ones_like(first_scale))
    assert torch.equal(first_scale, second_scale)
    assert torch.equal(first_scale, full_scale)
    assert torch.equal(torch.cat((first, second), dim=0 if layout == "NN" else 1), full)


@pytest.mark.parametrize("layout", ["NN", "TN"])
def test_gate_up_shards_pack_into_fused_weight(layout):
    gate = torch.arange(32 * 64, dtype=torch.float16).reshape(32, 64) % 13
    up = -(gate + 3)
    gate_pack, gate_scale = _pack_weight(gate, layout)
    up_pack, up_scale = _pack_weight(up, layout)
    packed, scale = _pack_weight(torch.cat((gate, up)), layout)
    assert torch.equal(torch.cat((gate_scale, up_scale), dim=1), scale)
    axis = 1 if layout == "NN" else 0
    assert torch.equal(torch.cat((gate_pack, up_pack), dim=axis), packed)
