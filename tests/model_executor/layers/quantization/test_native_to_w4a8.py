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


@pytest.mark.parametrize("layout,half,down_k", [("TN", 296, 384), ("NN", 320, 320)])
def test_native_loader_pads_logical_tp_shards(layout, half, down_k):
    from types import SimpleNamespace

    from tests.mxfp4_reference import pack_ixinfer
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEWeights,
        make_moe_compute,
        select_moe_backend,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.native.moe import (
        IluNativeToW4A8MoEMethod,
    )

    hidden, intermediate = 603, 289
    config = SimpleNamespace(
        has_bias=False,
        is_act_and_mul=True,
        intermediate_size_per_partition=intermediate,
        intermediate_size_per_partition_unpadded=intermediate,
        hidden_dim=hidden,
        moe_parallel_config=SimpleNamespace(tp_size=2),
        tp_rank=0,
        ep_rank=0,
    )
    layer = torch.nn.Module()
    layer.moe_config = config
    layer.local_num_experts = layer.global_num_experts = 1
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer._map_global_expert_id_to_local_expert_id = lambda index: index
    selection = select_moe_backend("ivcore11", name="ixinfer_w4a8", layout=layout)
    method = IluNativeToW4A8MoEMethod(config, selection)
    method.create_weights(layer, 1, hidden, intermediate, torch.bfloat16)
    compute = make_moe_compute(method._compute_config(layer), selection)
    assert compute.intermediate == half
    assert compute.config.w2_k == down_k

    gate = torch.ones(2 * intermediate, hidden, dtype=torch.bfloat16)
    up = -gate
    down = torch.ones(hidden, 2 * intermediate, dtype=torch.bfloat16)
    down[:, intermediate:] = 127  # Full checkpoint K controls the local codes.
    for name, shard, value in (
        ("w13", "w1", gate),
        ("w13", "w3", up),
        ("w2", "w2", down),
    ):
        weight = getattr(layer, name + "_weight")
        assert weight.weight_loader(weight, value, name, shard, 0, True)

    codes = torch.zeros(2 * half, compute.config.w13_k, dtype=torch.int8)
    codes[:intermediate, :hidden] = 7
    codes[half : half + intermediate, :hidden] = -7
    assert torch.equal(layer.w13_weight[0], pack_ixinfer(codes, layout))
    assert torch.count_nonzero(layer.w2_weight) == 0
    assert torch.equal(layer.w2_weight_scale, torch.ones_like(layer.w2_weight_scale))
    compute.validate_weights(MoEWeights.from_layer(layer, compute.weight_spec))
