# SPDX-License-Identifier: Apache-2.0
"""Device checks for runtime weight contracts, independent of model smoke tests."""

from types import SimpleNamespace

import ixformer.inference.functions as ops
import pytest
import torch

from tests.mxfp4_reference import mxfp4_channel_int4
from vllm_iluvatar.ilu_util.arch import cuda_arch
from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEComputeSelection,
    get_moe_backend,
    select_moe_backend,
)
from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
    Mxfp4LoadSpec,
    make_mxfp4_loader,
)
from vllm_iluvatar.model_executor.layers.quantization.common.paired_moe_loader import (
    ExpertUploadBuffer,
)


@pytest.mark.parametrize(
    "backend,layout,source",
    [
        (backend, layout, source)
        for source in ("mxfp4", "block_fp8")
        for backend, layout in (
            ("ixinfer_w4a8", "TN"),
            ("ixinfer_w4a8", "NN"),
            ("ixmm_w4a8", "NN"),
        )
    ]
    + [("ixinfer_w4a8_nn_v1", "NN", "block_fp8")],
)
def test_w4a8_runtime_weight_contract(backend, layout, source):
    arch = cuda_arch()
    ixmm_supported = arch == "ivcore30" or (
        arch == "ivcore11" and source == "block_fp8"
    )
    if (backend == "ixmm_w4a8" and not ixmm_supported) or (
        backend.startswith("ixinfer_w4a8") and layout == "NN" and arch != "ivcore11"
    ):
        pytest.skip("Not a supported architecture/layout combination")
    if backend == "ixmm_w4a8":
        from ixformer.inference.ixmm_backend import ixmm_available

        if not ixmm_available():
            pytest.skip("ixformer was built without the ixmm backend")
    torch.manual_seed(41)
    e, n, k = 2, 512, 512
    if source == "mxfp4":
        raw = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        scales = torch.randint(
            119, 123, (e, n, k // 32), device="cuda", dtype=torch.uint8
        )
        pairs = [mxfp4_channel_int4(raw[j], scales[j]) for j in range(e)]
        logical = torch.stack([p[0] for p in pairs])
        channel = torch.stack([p[1] for p in pairs])
        scale_rows, weight_cols, scale_cols = 1, 2, 32
    else:
        # Independent decoded source and two-stage channel quantization oracle.
        # The GEMM consumes the real FP8 loader's native output below.
        raw = (torch.randn(e, n, k) * 30).to(torch.float8_e4m3fn)
        scales = (torch.rand(e, n // 128, k // 128) + 0.1).to(torch.bfloat16)
        decoded = raw.float() * scales.float().repeat_interleave(
            128, 1
        ).repeat_interleave(128, 2)
        channel = decoded.abs().amax(-1) / 127
        channel = torch.where(channel == 0, 1.0, channel)
        logical = (
            (decoded / channel.unsqueeze(-1)).round().clamp(-127, 127) / 18
        ).round()
        logical, channel = logical.cuda(), channel.cuda()
        scale_rows, weight_cols, scale_cols = 128, 1, 128
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda expert: expert
    selection = select_moe_backend(arch, name=backend, layout=layout)
    spec = get_moe_backend(selection).weight_specs[layout]
    if source == "mxfp4":
        loader = make_mxfp4_loader(spec, Mxfp4LoadSpec(e, k, n // 2))
    else:
        from vllm_iluvatar.model_executor.layers.quantization.common.block_fp8_loader import (
            BlockFp8LoadSpec,
            make_block_fp8_loader,
        )

        loader = make_block_fp8_loader(spec, BlockFp8LoadSpec(e, k, n // 2, (128, 128)))
    with torch.device("cuda"):
        loader.create_weights(layer, {})
    buffer = ExpertUploadBuffer(
        loader.staging_bytes,
        layer.w13_weight.device,
        conversion_capacity=loader.conversion_bytes,
    )
    loader.bind_upload_buffer(buffer)
    addresses = (buffer.host.data_ptr(), buffer.gpu.data_ptr())
    for expert in range(e):
        for shard, w, sc in (
            ("w1", raw[expert, : n // 2], scales[expert, : n // 2 // scale_rows]),
            ("w3", raw[expert, n // 2 :], scales[expert, n // 2 // scale_rows :]),
            (
                "w2",
                raw[expert, :k, : n // 2 // weight_cols],
                scales[expert, : k // scale_rows, : n // 2 // scale_cols],
            ),
        ):
            group = "w2" if shard == "w2" else "w13"
            loader.load_shard(group, False, w.cpu(), shard, expert)
            loader.load_shard(group, True, sc.cpu(), shard, expert)
            assert addresses == (buffer.host.data_ptr(), buffer.gpu.data_ptr())
    loader.finalize()
    buffer.close()
    weight, runtime_scale = layer.w13_weight, layer.w13_weight_scale
    x = torch.randn((33, k), device="cuda", dtype=torch.bfloat16)
    a, sa = ops.scaled_int8_quant(x)
    out = torch.empty((33, n), device="cuda", dtype=torch.bfloat16)
    if backend == "ixmm_w4a8":
        ops.moe_w4a8_group_gemm_ixmm(
            a,
            weight,
            sa.reshape(-1),
            runtime_scale.flatten(1),
            torch.tensor([[0, 17], [17, 33]], device="cuda", dtype=torch.int32),
            out,
            backend="ixmm",
        )
    elif backend == "ixinfer_w4a8_nn_v1":
        from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
            MoEComputeConfig,
            MoEWeights,
        )

        compute = get_moe_backend(selection)(
            MoEComputeConfig(k, n // 2, e, e, limit=10), selection
        )
        compute.gemm(
            MoEWeights.from_layer(layer, spec),
            a,
            sa,
            "w13",
            torch.tensor([17, 16], dtype=torch.int32, device="cuda"),
            out.dtype,
            decode=True,
            output=out,
        )
    else:
        ops.moe_w4a8_group_gemm(
            a,
            weight,
            sa,
            runtime_scale,
            torch.bfloat16,
            torch.tensor([17, 16], dtype=torch.int32),
            format=layout,
            version=2,
            group_size=-1,
            output=out,
            backend="ixinfer",
        )
    expected = torch.empty_like(out)
    for j, (start, end) in enumerate(((0, 17), (17, 33))):
        # Same quantized operands; only FP32 matmul/scaling form the oracle.
        expected[start:end] = (
            (a[start:end].float() @ logical[j].float().T)
            * sa.reshape(-1, 1)[start:end]
            * channel[j]
            * 18
        ).to(torch.bfloat16)
    # BF16 epilogues may round scaling intermediates; budget two BF16 ULPs.
    torch.testing.assert_close(out, expected, rtol=0.016, atol=0.016)


@pytest.mark.parametrize(
    "activation,variant,limit",
    [
        ("silu", "dsv4_clamp", 10.0),
        ("swiglustep", "step3_swiglustep", 3.0),
        ("silu", "step3_swiglustep", None),
    ],
)
def test_fp4fp8_independent_forward_matches_existing_contract(
    activation, variant, limit
):
    if cuda_arch() != "ivcore30":
        pytest.skip("FP4FP8 requires ivcore30")
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
        _ilu_mxfp4_fused_experts_ixformer,
    )

    torch.manual_seed(52)
    h, i, e = 512, 256, 2
    weights = []
    for n, k in ((2 * i, h), (h, i)):
        w = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        s = torch.full((e, n, k // 32), 120, device="cuda", dtype=torch.uint8)
        weights.append(
            ops.preprocess_weight_for_ixmm(
                w.view(torch.int8), s.view(torch.float8_e8m0fnu)
            )
        )
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(hidden_dim=h, intermediate_size_per_partition=i),
        local_num_experts=e,
        activation=SimpleNamespace(value=activation),
        swiglu_limit=limit,
        apply_router_weight_on_input=False,
        expert_map=torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int32),
        w13_weight=weights[0][0],
        w13_weight_scale=weights[0][1],
        w2_weight=weights[1][0],
        w2_weight_scale=weights[1][1],
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        MoEWeights,
    )

    layer.global_num_experts = 4
    layer.moe_config.ep_rank = 0
    compute = IxMMFP4FP8Backend(
        MoEComputeConfig(h, i, e, 4, activation=activation, limit=limit),
        MoEComputeSelection("ixmm_fp4fp8", "fp4fp8", "ivcore30"),
    )
    runtime = MoEWeights.from_layer(layer, compute.weight_spec)
    compute.validate_weights(runtime)

    def forward(x, scores, ids, output=None):
        return compute.run_standard(
            x,
            runtime,
            scores,
            ids,
            layer.expert_map,
            output=output,
            workspace=compute.workspace_spec(ids.numel(), x.dtype).allocate(x.device),
        )

    x = torch.randn((33, h), device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (33, 2), device="cuda", dtype=torch.int32)
    scores = torch.rand((33, 2), device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    destination = torch.empty_like(x)
    actual = forward(x, scores, ids, destination)
    assert actual is destination
    reference = _ilu_mxfp4_fused_experts_ixformer(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        scores,
        ids,
        swiglu_limit=limit if limit is not None else 0.0,
        expert_map=layer.expert_map,
        output_scale=1.0,
        extra_residual=None,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixmm",
        gemm_mode=None,
        swiglu_variant=variant,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Exercise the Method's actual backend binding, late runner scaling and
    # shared residual through the same public apply entry used by model runners.
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        IluFp8MoEMethod,
    )

    method = object.__new__(IluFp8MoEMethod)
    method.selection = compute.selection
    method._compute = None
    layer._ilu_output_scale = 2.5
    method._bind_compute(layer, runtime)
    residual = torch.randn_like(x) * 0.25
    merged = method.apply(layer, x, scores, ids, None, residual)
    expected_merged = _ilu_mxfp4_fused_experts_ixformer(
        x,
        runtime.w13_weight,
        runtime.w13_weight_scale,
        runtime.w2_weight,
        runtime.w2_weight_scale,
        scores,
        ids,
        swiglu_limit=limit if limit is not None else 0.0,
        expert_map=layer.expert_map,
        output_scale=2.5,
        extra_residual=residual,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixmm",
        gemm_mode=None,
        swiglu_variant=variant,
    )
    torch.testing.assert_close(merged, expected_merged, rtol=0, atol=0)
    # Exercise the strided AllGather packet through the same Method/compute.
    packet, _, _ = ops.moe_ep_pack_fp8_routed(x, scores, ids)
    unpacked_scores, unpacked_ids = ops.moe_ep_unpack_fp8_routed_meta(packet, h, 2)
    quantized = packet[:, :h].view(torch.float8_e4m3fn)
    scales = packet[:, h : h + 4].view(torch.float32)[:, 0]
    assert method.supports_decode_packed_fp8_dispatch
    packed_result = method.apply(
        layer,
        (quantized, scales, x.dtype),
        unpacked_scores,
        unpacked_ids,
        None,
        residual,
    )
    torch.testing.assert_close(packed_result, merged, rtol=0.016, atol=0.016)
    assert method.supports_prequant_fp8_dispatch
    # Prefill AllGather returns separate contiguous q/scale tensors. Its native
    # local-route expansion must preserve the strided gather's exact operands.
    gathered_result = method.apply(
        layer,
        (quantized.contiguous(), scales.contiguous(), x.dtype),
        unpacked_scores,
        unpacked_ids,
        None,
        residual,
    )
    torch.testing.assert_close(gathered_result, packed_result, rtol=0, atol=0)

    inline_input = (packet.view(torch.float8_e4m3fn), h, x.dtype)
    inline_result = method.apply(
        layer, inline_input, unpacked_scores, unpacked_ids, None, residual
    )
    torch.testing.assert_close(inline_result, packed_result, rtol=0, atol=0)
    inline_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(inline_graph):
        inline_captured = method.apply(
            layer, inline_input, unpacked_scores, unpacked_ids, None, residual
        )
    next_packet, _, _ = ops.moe_ep_pack_fp8_routed(
        x * 0.5, scores.flip(1).contiguous(), (ids + 1) % 4
    )
    packet.copy_(next_packet)
    next_scores, next_ids = ops.moe_ep_unpack_fp8_routed_meta(packet, h, 2)
    unpacked_scores.copy_(next_scores)
    unpacked_ids.copy_(next_ids)
    inline_graph.replay()
    # Separate q/scale views see the updated packet, independently exercising
    # the existing gather path as the byte-preserving expansion oracle.
    replay_expected = method.apply(
        layer,
        (quantized, scales, x.dtype),
        unpacked_scores,
        unpacked_ids,
        None,
        residual,
    )
    torch.testing.assert_close(inline_captured, replay_expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="already been bound"):
        method._bind_compute(layer, runtime)
    # Replay with changed routing, so capture cannot freeze expert metadata.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward(x, scores, ids, destination)
    ids.copy_((ids + 1) % 4)
    graph.replay()
    torch.testing.assert_close(captured, forward(x, scores, ids), rtol=0, atol=0)

    # Batched kernels have their own quantization/rounding path. Compare with
    # the previous batched implementation, without loosening bitwise parity.
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.batch_experts import (
        IluMoEBatchedExpertsFp8,
    )

    legacy = object.__new__(IluMoEBatchedExpertsFp8)
    legacy.out_dtype = torch.bfloat16
    legacy.swiglu_limit = (
        None if variant == "step3_swiglustep" and activation != "swiglustep" else limit
    )
    legacy.quant_config = SimpleNamespace(
        w1_scale=runtime.w13_weight_scale, w2_scale=runtime.w2_weight_scale
    )
    batched_x = torch.randn((e, 4, h), device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    batched_out = torch.empty_like(batched_x)
    compute.run_batched(
        batched_x,
        runtime,
        None,
        counts,
        output=batched_out,
        workspace=compute.workspace_spec(e * 4, batched_x.dtype, batched=True).allocate(
            batched_x.device
        ),
    )
    expected = torch.empty_like(batched_out)
    legacy.apply(
        expected,
        batched_x,
        runtime.w13_weight,
        runtime.w2_weight,
        None,
        None,
        SimpleNamespace(value=activation),
        4,
        layer.expert_map,
        None,
        None,
        None,
        None,
        SimpleNamespace(expert_num_tokens=counts),
    )
    torch.testing.assert_close(batched_out[0, :3], expected[0, :3], rtol=0, atol=0)
    torch.testing.assert_close(batched_out[1, :1], expected[1, :1], rtol=0, atol=0)


@pytest.mark.parametrize("limit", [None, 10.0])
def test_w4a8_prequantized_input_packet_and_norm_tuple(limit):
    if cuda_arch() != "ivcore30":
        pytest.skip("ixmm_w4a8 and its ixmm_nn weight format require ivcore30")
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        MoEWeights,
        make_moe_compute,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_int8_router_pack import (
        pack_router_input,
        unpack_router_input,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        IluFp8MoEMethod,
    )

    torch.manual_seed(91)
    h, intermediate, experts, rows, topk = 512, 256, 2, 17, 2
    compute = make_moe_compute(
        MoEComputeConfig(h, intermediate, experts, experts, limit=limit),
        MoEComputeSelection("ixmm_w4a8", "NN", cuda_arch()),
    )
    assert compute.supports_batched_experts == (cuda_arch() == "ivcore30")
    tensors = []
    for n, k in ((2 * intermediate, h), (h, intermediate)):
        w = torch.empty((experts, k, n // 2), device="cuda", dtype=torch.int8)
        s = torch.empty((experts, 1, n), device="cuda", dtype=torch.float32)
        for expert in range(experts):
            raw = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            scale = torch.full((n // 128, k // 128), 0.01, device="cuda")
            ops.fp8_block_to_w4a8(
                raw, scale, w[expert], s[expert], weight_format="ixmm_nn"
            )
        tensors.extend((w, s))
    runtime = MoEWeights(compute.weight_spec, *tensors)
    compute.validate_weights(runtime)
    method = object.__new__(IluFp8MoEMethod)
    method.selection = compute.selection
    method._compute, method._runtime_weights = compute, runtime
    layer = SimpleNamespace(expert_map=None)
    assert method.supports_prequantized_input
    assert method.supports_packed_int8_input
    assert not method.supports_decode_packed_fp8_dispatch
    x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
    q, scale = ops.scaled_int8_quant(x)
    ids = torch.randint(0, experts, (rows, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(rows, topk, device="cuda")
    residual = torch.randn_like(x) * 0.01
    expected = method.apply(layer, (q, scale, x.dtype), weights, ids, None, residual)
    if cuda_arch() == "ivcore11":
        plain = method.apply(layer, x, weights, ids, None, residual)
        torch.testing.assert_close(plain, expected, rtol=0, atol=0)
    logits = torch.randn(rows, experts, device="cuda")
    packet, layout = pack_router_input(q, scale, logits, None)
    packed, unpacked_logits, _ = unpack_router_input(packet, layout, x.dtype)
    assert torch.equal(logits, unpacked_logits)
    actual = method.apply(layer, packed, weights, ids, None, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
