# SPDX-License-Identifier: Apache-2.0
"""Shared native NN MoE fixture; unchanged from the main quantization tests."""

from types import SimpleNamespace

import ixformer.inference.functions as ixf_ops
import torch

from vllm_iluvatar.model_executor.layers.quantization import ilu_w4a8_nn as w4a8_moe
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
)


def make_nn_graph_method(experts=4, hidden=256, intermediate=256):
    rng = torch.Generator().manual_seed(71)
    cls = w4a8_moe.IluW4A8NNMoEMethod
    method = cls.__new__(cls)
    method.group_size = -1
    method.gemm_format = "NN"
    method.gemm1_info = W4A8GroupGemmInfo("NN", hidden, 2 * intermediate)
    method.gemm2_info = W4A8GroupGemmInfo("NN", intermediate, hidden)
    layer = torch.nn.Module()
    raw = []
    scales = []
    for prefix, n, k in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        w = torch.randint(-7, 8, (experts, n, k), generator=rng, dtype=torch.int8)
        scale = torch.rand(experts, 1, n, generator=rng) * 0.002
        # Construct an NN checkpoint directly, independent of plugin packing.
        # Force every output channel's q8 amax=126, hence the NN scale is 18.
        w[..., 0] = 7
        q8 = (w.to(torch.int16) * 18).clamp(-127, 127).to(torch.int8).cuda()
        packed, extra, _ = ixf_ops.quant_repack_int4(q8, -1, 1, "NN", False)
        torch.testing.assert_close(extra, torch.full_like(extra, 18))
        layer.register_parameter(
            prefix + "_weight", torch.nn.Parameter(packed, requires_grad=False)
        )
        layer.register_parameter(
            prefix + "_weight_scale",
            torch.nn.Parameter((scale * 18).cuda(), requires_grad=False),
        )
        raw.append(w)
        scales.append(scale)
        for suffix in ("i8_weight_scale", "i8_weight_zero", "input_scale", "bias"):
            setattr(layer, prefix + "_" + suffix, None)
    layer.moe_config = SimpleNamespace(ep_rank=1)
    layer.local_num_experts = experts
    layer.global_num_experts = experts * 2
    layer.top_k = 2
    layer.expert_map = torch.arange(experts, device="cuda")
    layer.activation = SimpleNamespace(value="silu")
    layer._ilu_output_scale = 0.375
    return method, layer, raw, scales
