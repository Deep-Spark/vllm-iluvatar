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
    + [("ixinfer_w4a8_nn_v1", "NN", "block_fp8"), ("ixmm_w4a8", "NN", "mxfp8")],
)
def test_w4a8_runtime_weight_contract(backend, layout, source):
    arch = cuda_arch()
    ixmm_supported = arch in ("ivcore11", "ivcore30")
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
        scale_rows, weight_cols, scale_cols = (
            (1, 1, 32) if source == "mxfp8" else (128, 1, 128)
        )
        if source == "mxfp8":
            scales = torch.randint(119, 123, (e, n, k // 32), dtype=torch.uint8)
            block_scale = scales.view(torch.float8_e8m0fnu).float()
        else:
            scales = (torch.rand(e, n // 128, k // 128) + 0.1).to(torch.bfloat16)
            block_scale = scales.float()
        decoded = raw.float() * block_scale.repeat_interleave(
            scale_rows, 1
        ).repeat_interleave(scale_cols, 2)
        channel = decoded.abs().amax(-1) / 127
        channel = torch.where(channel == 0, 1.0, channel)
        logical = (
            (decoded / channel.unsqueeze(-1)).round().clamp(-127, 127) / 18
        ).round()
        logical, channel = logical.cuda(), channel.cuda()
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

        factory = make_block_fp8_loader
        if source == "mxfp8":
            from vllm_iluvatar.model_executor.layers.quantization.common.mxfp8_loader import (
                make_mxfp8_loader,
            )

            factory = make_mxfp8_loader
        loader = factory(spec, BlockFp8LoadSpec(e, k, n // 2, (scale_rows, scale_cols)))
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


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("source", ["MXFP4", "block-FP8", "MXFP8"])
def test_ixinfer_online_padding_and_graph_replay(monkeypatch, layout, source):
    if cuda_arch() != "ivcore11":
        pytest.skip("BI150 TN/NN v2 required")

    from dataclasses import replace

    from vllm_iluvatar.model_executor.layers.fused_moe.experts import ilu_ixinfer_w4a8
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        make_moe_compute,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        IluFp8MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
        IluMxfp4MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.moe import (
        IluMxFp8MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
        MoEComputeBindingMixin,
    )

    torch.manual_seed(61)
    hidden, intermediate, experts = 608, 288, 2
    config = MoEComputeConfig(hidden, intermediate, experts, experts)
    monkeypatch.setattr(MoEComputeBindingMixin, "_compute_config", lambda *_: config)
    selection = select_moe_backend("ivcore11", name="ixinfer_w4a8", layout=layout)
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        intermediate_size_per_partition_unpadded=intermediate,
        hidden_dim_unpadded=hidden,
        has_bias=False,
        is_act_and_mul=True,
        tp_size=2,
        tp_rank=1,
        max_capture_size=16,
    )
    layer.expert_map = None
    layer._map_global_expert_id_to_local_expert_id = lambda index: index
    if source == "MXFP4":
        method = IluMxfp4MoEMethod(layer.moe_config, selection)
    elif source == "MXFP8":
        method = IluMxFp8MoEMethod(None, layer.moe_config)
        method.selection = selection
    else:
        method = object.__new__(IluFp8MoEMethod)
        method.moe = layer.moe_config
        method.weight_block_size = (128, 128)
        method._checkpoint_loader = None
        method.selection = selection
    with torch.device("cuda"):
        method.create_weights(layer, experts, hidden, intermediate, torch.bfloat16)
    loader = method._checkpoint_loader

    compute = make_moe_compute(method._compute_config(layer), selection)
    w13_k, w2_n = 640, (640 if layout == "NN" else hidden)
    assert compute.config.w13_k == w13_k
    assert compute.config.w2_n == w2_n
    runtime_i = 320 if layout == "NN" else intermediate
    w2_k = 320 if layout == "NN" else 384
    assert compute.intermediate == runtime_i
    assert compute.config.w2_k == w2_k
    loader.bind_upload_buffer(
        ExpertUploadBuffer(loader.staging_bytes, layer.w2_weight.device)
    )

    # Independently pack the zero-extended source and compare the runtime bytes.
    from tests.mxfp4_reference import pack_ixinfer

    for expert in range(experts):
        expected_w13, expected_w13_scale = [], []
        for shard in ("w1", "w3", "w2"):
            group = "w2" if shard == "w2" else "w13"
            rows = (
                slice(intermediate, 2 * intermediate) if group == "w13" else slice(None)
            )
            cols = (
                slice(intermediate, 2 * intermediate) if group == "w2" else slice(None)
            )
            ws, ss = (
                loader.source_shape(group, is_scale) for is_scale in (False, True)
            )
            if source == "MXFP4":
                raw = torch.randint(0, 256, ws, dtype=torch.uint8)
                scales = torch.randint(119, 124, ss, dtype=torch.uint8)
                packed_cols = (
                    slice(intermediate // 2, intermediate)
                    if group == "w2"
                    else slice(None)
                )
                scale_cols = (
                    slice(intermediate // 32, intermediate // 16)
                    if group == "w2"
                    else slice(None)
                )
                codes, channels = mxfp4_channel_int4(
                    raw[rows, packed_cols], scales[rows, scale_cols]
                )
            else:
                raw = torch.randn(ws).to(torch.float8_e4m3fn)
                exponents = torch.randint(119, 124, ss, dtype=torch.uint8)
                block_scales = (exponents.to(torch.int32) << 23).view(torch.float32)
                scales = block_scales if source == "block-FP8" else exponents
                bn, bk = (128, 128) if source == "block-FP8" else (1, 32)
                dense_scales = block_scales.repeat_interleave(bn, 0).repeat_interleave(
                    bk, 1
                )
                dense = (
                    raw.float()[rows, cols] * dense_scales[: ws[0], : ws[1]][rows, cols]
                )
                channels = dense.abs().amax(dim=-1, keepdim=True) / 127
                channels = torch.where(channels == 0, 1.0, channels)
                codes = (
                    ((dense / channels).round().clamp(-127, 127) / 18)
                    .round()
                    .clamp(-8, 7)
                    .to(torch.int8)
                )
                channels = channels.T
            for is_scale, tensor in ((False, raw), (True, scales)):
                assert loader.load_shard(group, is_scale, tensor, shard, expert)
            if shard != "w2":
                expected_w13.append(
                    torch.nn.functional.pad(
                        codes, (0, w13_k - hidden, 0, runtime_i - intermediate)
                    )
                )
                expected_w13_scale.append(
                    torch.nn.functional.pad(
                        channels, (0, runtime_i - intermediate), value=1.0
                    )
                )
                continue
            codes = torch.nn.functional.pad(
                codes, (0, w2_k - intermediate, 0, w2_n - hidden)
            )
            expected = pack_ixinfer(codes, layout).cuda()
            torch.testing.assert_close(
                layer.w2_weight[expert], expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                getattr(layer, "w2_" + loader.scale_parameter_name)[expert, :, :hidden],
                channels.cuda(),
            )
        expected = pack_ixinfer(torch.cat(expected_w13), layout).cuda()
        torch.testing.assert_close(layer.w13_weight[expert], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            getattr(layer, "w13_" + loader.scale_parameter_name)[expert],
            torch.cat(expected_w13_scale, dim=-1).cuda(),
        )

    weights = loader.finalize()
    compute.validate_weights(weights)
    reference = make_moe_compute(replace(compute.config, hidden=w2_n), selection)
    ctx = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(ilu_ixinfer_w4a8, "get_forward_context", lambda: ctx)

    x = torch.randn((16, hidden), device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(experts, (16, 2), device="cuda", dtype=torch.int32)
    scores = torch.full((16, 2), 0.5, device="cuda")

    def padded_x(value):
        return torch.nn.functional.pad(value, (0, w13_k - hidden))

    def standard(backend, inputs):
        return backend.run_standard(
            inputs,
            weights,
            scores,
            ids,
            None,
            workspace=backend.workspace_spec(ids.numel(), x.dtype).allocate(x.device),
        )

    actual = standard(compute, x)
    assert actual.shape == x.shape
    torch.testing.assert_close(
        actual, standard(reference, padded_x(x))[:, :hidden], rtol=0, atol=0
    )

    ctx.attn_metadata = SimpleNamespace(num_prefills=0, num_decodes=x.shape[0])
    standard(compute, x)  # Warm up before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = standard(compute, x)
    x.mul_(0.5)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    torch.testing.assert_close(
        captured, standard(reference, padded_x(x))[:, :hidden], rtol=0, atol=0
    )

    counts = torch.tensor([8, 8], device="cuda", dtype=torch.int32)
    batched_x = x.reshape(experts, 8, hidden)

    def batched(backend, inputs):
        out = torch.empty((experts, 8, backend.hidden), device=x.device, dtype=x.dtype)
        return backend.run_batched(
            inputs,
            weights,
            None,
            counts,
            output=out,
            workspace=backend.workspace_spec(16, x.dtype, batched=True).allocate(
                x.device
            ),
        )

    torch.testing.assert_close(
        batched(compute, batched_x),
        batched(reference, padded_x(batched_x))[..., :hidden],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "source,activation",
    [
        ("MXFP4", "swiglustep"),
        ("block-FP8", "swiglustep"),
        ("MXFP8", "swiglustep"),
        ("MXFP8", "swigluoai_uninterleave"),
    ],
)
def test_ixmm_online_padding_and_activation_graph(source, activation, monkeypatch):
    from ixformer.inference.ixmm_backend import ixmm_available

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        make_moe_compute,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_w4a8 import (
        IxMMW4A8Backend,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        IluFp8MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.moe import (
        IluMxFp8MoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
        MoEComputeBindingMixin,
    )

    if not ixmm_available():
        pytest.skip("IxMM bridge required")
    torch.manual_seed(73)
    hidden, intermediate, experts = 608, 288, 2
    config = IxMMW4A8Backend.padded_config(
        MoEComputeConfig(
            hidden,
            intermediate,
            experts,
            experts,
            activation=activation,
            limit=3.0 if activation == "swiglustep" else 7.0,
        )
    )
    assert (config.w13_k, config.w2_n, config.w2_k) == (640, 640, 320)
    compute = make_moe_compute(
        config, MoEComputeSelection("ixmm_w4a8", "NN", cuda_arch())
    )
    layer = torch.nn.Module()
    layer._map_global_expert_id_to_local_expert_id = lambda index: index
    layer.expert_map = None
    if source == "MXFP4":
        loader = make_mxfp4_loader(
            compute.weight_spec,
            Mxfp4LoadSpec(
                experts,
                hidden,
                config.intermediate,
                2,
                1,
                intermediate,
                w13_k_padded=config.w13_k,
                w2_n_padded=config.w2_n,
            ),
        )
        with torch.device("cuda"):
            loader.create_weights(layer, {})
    else:
        layer.moe_config = SimpleNamespace(
            intermediate_size_per_partition_unpadded=intermediate,
            hidden_dim_unpadded=hidden,
            has_bias=False,
            is_act_and_mul=True,
            tp_size=2,
            tp_rank=1,
        )
        # vLLM advertises the padded width, while source tensors remain logical.
        monkeypatch.setattr(
            MoEComputeBindingMixin, "_compute_config", lambda *_: config
        )
        if source == "MXFP8":
            method = IluMxFp8MoEMethod(None, layer.moe_config)
        else:
            method = object.__new__(IluFp8MoEMethod)
            method.moe = layer.moe_config
            method.weight_block_size = (128, 128)
            method._checkpoint_loader = None
        method.selection = compute.selection
        with torch.device("cuda"):
            method.create_weights(
                layer, experts, hidden, config.intermediate, torch.bfloat16
            )
        loader = method._checkpoint_loader
        assert loader.source_shape("w2", False) == (hidden, 2 * intermediate)
    loader.bind_upload_buffer(
        ExpertUploadBuffer(loader.staging_bytes, layer.w2_weight.device)
    )
    for expert in range(experts):
        for shard in ("w1", "w3", "w2"):
            group = "w2" if shard == "w2" else "w13"
            ws = loader.source_shape(group, False)
            ss = loader.source_shape(group, True)
            if source == "MXFP4":
                raw = torch.randint(256, ws, dtype=torch.uint8)
            else:
                raw = torch.randn(ws).to(torch.float8_e4m3fn)
            scale = (
                torch.full(ss, 120, dtype=torch.uint8)
                if source != "block-FP8"
                else torch.full(ss, 0.01, dtype=torch.float32)
            )
            assert loader.load_shard(group, False, raw, shard, expert)
            assert loader.load_shard(group, True, scale, shard, expert)
    weights = loader.finalize()
    compute.validate_weights(weights)

    a = (
        torch.randn(16, 2 * config.intermediate, device="cuda", dtype=torch.bfloat16)
        * 8
    )
    gate, up = a.float().chunk(2, dim=-1)
    if activation == "swiglustep":
        expected = torch.nn.functional.silu(gate).clamp(max=3) * up.clamp(-3, 3)
        q, s = compute._swiglustep_quant(a)
    else:
        gate = gate.clamp(max=7)
        expected = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
        q, s = compute._swigluoai_quant(a)
    tolerance = 2 * expected.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((q.float() * s.reshape(-1, 1) - expected).abs() <= tolerance)

    x = torch.randn(16, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(experts, (16, 2), device="cuda", dtype=torch.int32)
    scores = torch.full((16, 2), 0.5, device="cuda")

    def run():
        return compute.run_standard(
            x,
            weights,
            scores,
            ids,
            None,
            workspace=compute.workspace_spec(ids.numel(), x.dtype).allocate(x.device),
        )

    eager = run()
    assert eager.shape == x.shape
    assert eager.isfinite().all()
    assert eager.abs().max() > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    x.mul_(0.5)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    torch.testing.assert_close(captured, run(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "activation,variant,limit",
    [
        ("silu", "dsv4_clamp", 10.0),
        ("swiglustep", "step3_swiglustep", 3.0),
        ("silu", "step3_swiglustep", None),
    ],
)
@pytest.mark.parametrize("intermediate", [160, 256, 384])
def test_fp4fp8_independent_forward_matches_existing_contract(
    activation, variant, limit, intermediate
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
    h, i, e = 512, intermediate, 2
    weights = []
    w2_k = (i + 255) // 256 * 256
    for n, k in ((2 * i, h), (h, w2_k)):
        w = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        s = torch.full((e, n, k // 32), 120, device="cuda", dtype=torch.uint8)
        if weights:
            w[..., i // 2 :] = 0
            s[..., i // 32 :] = 127
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

    # The legacy batched implementation only accepts an aligned GEMM2 K.
    # Padded widths above exercise Standard and its CUDA Graph replay.
    if i != w2_k:
        return

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
