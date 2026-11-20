# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for serialized F4F8 expert loading and execution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import requires_corex30
from ixformer.inference.functions.ops.gemm.fp4.gemm.gemm import _pack_nibbles_ix30
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_iluvatar.model_executor.layers.quantization.ilu_f4f8 import (
    IluF4F8Config,
    IluF4F8MoEMethod,
)

_GROUP = 256
_HIDDEN = 512
_INTERMEDIATE = 512
_LOCAL_EXPERTS = 4


def _config():
    return IluF4F8Config.from_config(
        {
            "quant_method": "ilu_f4f8",
            "format": "ilu-mixed-f4f8",
            "version": 1,
            "routed_experts": {
                "quant_dtype": "fp4_e2m1",
                "group_size": _GROUP,
                "storage_dtype": "uint8",
                "weight_layout": "ix30_kmajor_pack32_v1",
                "scale_dtype": "float32",
                "scale_form": "absolute",
                "input_activations": {
                    "quant_dtype": "fp8_e4m3fn",
                    "strategy": "token",
                    "dynamic": True,
                },
            },
        }
    )


def _projection(experts, k, n):
    nibbles = torch.randint(0, 16, (experts, k, n), dtype=torch.uint8)
    scales = torch.rand(experts, k // _GROUP, n) * 0.008 + 0.004
    levels = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    reference = levels[nibbles.long()] * scales.repeat_interleave(_GROUP, dim=1)
    return _pack_nibbles_ix30(nibbles), scales, reference


def _fp8_roundtrip(value):
    scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return (value / scale).to(torch.float8_e4m3fn).float() * scale


@requires_corex30
@pytest.mark.parametrize(
    "tokens,tp_size,tp_rank,use_ep,limit",
    [
        (1, 1, 0, False, None),
        (7, 2, 1, False, 0.2),
        (32, 1, 0, True, None),
        (64, 1, 0, False, 0.2),
        (65, 1, 0, True, 0.2),
        (192, 1, 0, False, 10.0),
        (384, 1, 0, True, 10.0),
    ],
)
@torch.inference_mode()
def test_f4f8_loader_and_local_experts_match_reference(
    cuda_device, monkeypatch, tokens, tp_size, tp_rank, use_ep, limit
):
    """Cover TP slices, remote/empty experts, GEMV/GEMM and clamp semantics."""
    torch.manual_seed(813)
    device = torch.device(cuda_device)
    global_experts = _LOCAL_EXPERTS * (2 if use_ep else 1)
    first_expert = _LOCAL_EXPERTS if use_ep else 0
    local_intermediate = _INTERMEDIATE // tp_size
    parallel = FusedMoEParallelConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        pcp_size=1,
        pcp_rank=0,
        dp_size=2 if use_ep else 1,
        dp_rank=1 if use_ep else 0,
        ep_size=2 if use_ep else 1,
        ep_rank=1 if use_ep else 0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=global_experts,
        experts_per_token=2,
        hidden_dim=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_local_experts=_LOCAL_EXPERTS,
        num_logical_experts=global_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        swiglu_limit=limit,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.swiglu_limit = limit
    layer.apply_router_weight_on_input = False
    mapping = torch.full((global_experts,), -1, dtype=torch.int32)
    mapping[first_expert : first_expert + _LOCAL_EXPERTS] = torch.arange(
        _LOCAL_EXPERTS, dtype=torch.int32
    )
    layer.expert_map = mapping.to(device) if use_ep else None
    layer._map_global_expert_id_to_local_expert_id = lambda i: int(mapping[i])
    method = IluF4F8MoEMethod(_config(), layer)
    with torch.device(device):
        method.create_weights(
            layer, _LOCAL_EXPERTS, _HIDDEN, local_intermediate, torch.bfloat16
        )
    checkpoint = {
        "w1": _projection(global_experts, _HIDDEN, _INTERMEDIATE),
        "w3": _projection(global_experts, _HIDDEN, _INTERMEDIATE),
        "w2": _projection(global_experts, _INTERMEDIATE, _HIDDEN),
    }
    for shard, (packed, scales, _) in checkpoint.items():
        weight = layer.w2_weight if shard == "w2" else layer.w13_weight
        scale = layer.w2_weight_scale if shard == "w2" else layer.w13_weight_scale
        for expert in range(global_experts):
            for param, tensor, suffix in [
                (weight, packed[expert], "weight"),
                (scale, scales[expert], "weight_scale"),
            ]:
                param.weight_loader(param, tensor, f"{shard}_{suffix}", shard, expert)
    raw13 = layer.w13_weight_scale.detach().clone()
    raw2 = layer.w2_weight_scale.detach().clone()
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    for raw, actual in [(raw13, layer.w13_weight_scale), (raw2, layer.w2_weight_scale)]:
        expected = raw.clone()
        expected[:, :-1] = raw[:, :-1] / raw[:, 1:]
        torch.testing.assert_close(actual, expected)

    x = (torch.randn(tokens, _HIDDEN) * 0.3).to(torch.bfloat16)
    ids = torch.stack(
        [
            torch.full((tokens,), first_expert),
            torch.arange(tokens) % 2 + (first_expert + 1 if not use_ep else 0),
        ],
        dim=1,
    ).to(torch.int32)
    if use_ep:
        ids[::3, 1] = first_expert + 1
    weights = torch.rand(tokens, 2)
    weights /= weights.sum(-1, keepdim=True)
    shared = (torch.randn_like(x.float()) * 0.01).to(torch.bfloat16)
    expected = shared.float().clone()
    qx = _fp8_roundtrip(x.float())
    lo, hi = tp_rank * local_intermediate, (tp_rank + 1) * local_intermediate
    for expert in range(first_expert, first_expert + _LOCAL_EXPERTS):
        selected = (ids == expert).nonzero()
        if selected.numel() == 0:
            continue
        rows, slots = selected.unbind(1)
        gate = (
            (qx[rows] @ checkpoint["w1"][2][expert, :, lo:hi])
            .to(torch.bfloat16)
            .float()
        )
        up = (
            (qx[rows] @ checkpoint["w3"][2][expert, :, lo:hi])
            .to(torch.bfloat16)
            .float()
        )
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        act = _fp8_roundtrip(F.silu(gate) * up)
        routed = (act @ checkpoint["w2"][2][expert, lo:hi]).to(torch.bfloat16).float()
        expected.index_add_(0, rows, routed * weights[rows, slots, None])
    init_workspace_manager(device)
    # Unused compact rows are deliberately NaN: downstream kernels must mask
    # them, including partial expert tiles and experts with no routed tokens.
    import ixformer.inference.functions as functions

    expand = functions.moe_expand_input_dynamic_scaled_fp8

    def expand_with_poisoned_tail(hidden_states, dst_to_src, top_k, **kwargs):
        rows, hidden = dst_to_src.numel(), hidden_states.shape[1]
        fp8 = torch.empty(rows, hidden, dtype=torch.float8_e4m3fn, device=device)
        fp8.view(torch.uint8).fill_(0x7F)
        scales = torch.full((rows,), float("nan"), device=device)
        return expand(
            hidden_states,
            dst_to_src,
            top_k,
            fp8_output=fp8,
            output_scales=scales,
            **kwargs,
        )

    monkeypatch.setattr(
        functions, "moe_expand_input_dynamic_scaled_fp8", expand_with_poisoned_tail
    )
    actual = method.apply(
        layer,
        x.to(device),
        weights.to(device),
        ids.to(device),
        None,
        shared.to(device),
    )
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.06, atol=0.004)


@requires_corex30
@pytest.mark.parametrize("tokens", [384, 512, 513])
def test_f4f8_sorted_experts_match_reference(cuda_device, monkeypatch, tokens):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "_LOCAL_EXPERTS", 36)
    test_f4f8_loader_and_local_experts_match_reference(
        cuda_device, monkeypatch, tokens, 1, 0, True, 10.0
    )
