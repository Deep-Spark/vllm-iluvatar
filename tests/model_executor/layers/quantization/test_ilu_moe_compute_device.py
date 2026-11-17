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
    Mxfp4UploadBuffer,
    make_mxfp4_loader,
)


@pytest.mark.parametrize(
    "backend,layout",
    [("ixinfer_w4a8", "TN"), ("ixinfer_w4a8", "NN"), ("ixmm_w4a8", "NN")],
)
def test_w4a8_runtime_weight_contract(backend, layout):
    arch = cuda_arch()
    if (backend == "ixmm_w4a8" and arch != "ivcore30") or (
        backend == "ixinfer_w4a8" and layout == "NN" and arch != "ivcore11"
    ):
        pytest.skip("Not a supported architecture/layout combination")
    torch.manual_seed(41)
    e, n, k = 2, 512, 512
    raw = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(119, 123, (e, n, k // 32), device="cuda", dtype=torch.uint8)
    pairs = [mxfp4_channel_int4(raw[j], scales[j]) for j in range(e)]
    logical = torch.stack([p[0] for p in pairs])
    channel = torch.stack([p[1] for p in pairs])
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda expert: expert
    selection = select_moe_backend(arch, name=backend, layout=layout)
    spec = get_moe_backend(selection).weight_specs[layout]
    loader = make_mxfp4_loader(spec, Mxfp4LoadSpec(e, k, n // 2))
    with torch.device("cuda"):
        loader.create_weights(layer, {})
    buffer = Mxfp4UploadBuffer(loader.staging_bytes, layer.w13_weight.device)
    loader.bind_upload_buffer(buffer)
    addresses = (buffer.host.data_ptr(), buffer.gpu.data_ptr())
    for expert in range(e):
        for shard, w, sc in (
            ("w1", raw[expert, : n // 2], scales[expert, : n // 2]),
            ("w3", raw[expert, n // 2 :], scales[expert, n // 2 :]),
            ("w2", raw[expert, :k, : n // 4], scales[expert, :k, : n // 64]),
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


def test_fp4fp8_independent_forward_matches_existing_contract():
    activation, variant, limit = "silu", "dsv4_clamp", 10.0
    if cuda_arch() != "ivcore30":
        pytest.skip("FP4FP8 requires ivcore30")
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
        ilu_mxfp4_fused_experts,
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
    reference = ilu_mxfp4_fused_experts(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        scores,
        ids,
        swiglu_limit=limit,
        expert_map=layer.expert_map,
        backend="ixmm",
        swiglu_variant=variant,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
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
