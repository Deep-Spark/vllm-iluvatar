# SPDX-License-Identifier: Apache-2.0
"""MXFP4 load-time conversion, source ownership and exact shard assembly."""

import ast
from pathlib import Path

import pytest
import torch

from tests.mxfp4_reference import mxfp4_channel_int4, pack_ixinfer
from vllm_iluvatar.ilu_util.arch import cuda_arch
from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    get_moe_backend,
    select_moe_backend,
)
from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
    Mxfp4LoadSpec,
    Mxfp4UploadBuffer,
    make_mxfp4_loader,
)


@pytest.mark.parametrize("n,k", [(128, 256), (256, 512)])
def test_mxfp4_quantization_and_tn_payload(n, k):
    gen = torch.Generator().manual_seed(23)
    raw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    exponents = torch.randint(120, 130, (n, k // 32), dtype=torch.uint8, generator=gen)
    raw[0] = 0
    q, scale = mxfp4_channel_int4(raw, exponents)
    # Independently decode interleaved E2M1 values, including signed zero.
    code = torch.empty((n, k), dtype=torch.int64)
    code[:, ::2] = raw & 15
    code[:, 1::2] = raw >> 4
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)[
        code & 7
    ]
    values = (
        magnitude
        * torch.where(code < 8, 1.0, -1.0)
        * (2.0 ** (exponents.double() - 127)).repeat_interleave(32, 1)
    )
    expected_scale = values.abs().amax(1, keepdim=True) / 127
    expected_scale[expected_scale == 0] = 1
    # Quantization arithmetic is explicitly FP32, not a BF16 intermediate.
    expected_scale = expected_scale.float()
    expected_q = (
        ((values.float() / expected_scale).round().clamp(-127, 127) / 18)
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
    )
    torch.testing.assert_close(scale, expected_scale.T, rtol=0, atol=0)
    assert torch.equal(q, expected_q)
    packed = pack_ixinfer(q, "TN").reshape(n, k // 32, 16)
    decoded = (
        torch.cat((packed & 15, (packed >> 4) & 15), -1).reshape(n, k).to(torch.int16)
    )
    decoded = torch.where(decoded >= 8, decoded - 16, decoded).to(torch.int8)
    assert torch.equal(decoded, q)


def test_nan_scale_rejected():
    with pytest.raises(ValueError, match="NaN"):
        mxfp4_channel_int4(
            torch.zeros((128, 128), dtype=torch.uint8),
            torch.full((128, 8), 255, dtype=torch.uint8),
        )


def test_compute_does_not_delegate_to_old_moe_forward():
    root = (
        Path(__file__).parents[4]
        / "vllm_iluvatar/model_executor/layers/fused_moe/experts"
    )
    forbidden = (
        "ilu_mxfp4_fused_moe",
        "ilu_fp8_batch_experts",
        "ilu_w4a8_nn",
        "ilu_compressed_tensors_moe_w4a8_int8",
        "ilu_moe_batch_experts",
        "ilu_moe_standard_experts",
    )
    paths = list(root.glob("ilu_*.py"))
    assert len(paths) >= 4
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    (node.module or "").endswith(name) for name in forbidden
                ), path


class ExpertLayer(torch.nn.Module):
    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return {4: 0, 5: 1}.get(expert_id, -1)


def deliver(layer, shard, expert, tensor, scale=False):
    name = ("w2" if shard == "w2" else "w13") + "_weight" + ("_scale" if scale else "")
    param = getattr(layer, name)
    return param.weight_loader(param, tensor, name, shard, expert, return_success=True)


def loader_for(layer, backend, layout, device="cpu", tp=1, rank=0):
    arch = "ivcore11" if backend == "ixinfer_w4a8" else "ivcore30"
    selection = select_moe_backend(arch, name=backend, layout=layout)
    spec = get_moe_backend(selection).weight_specs[layout]
    loader = make_mxfp4_loader(spec, Mxfp4LoadSpec(2, 512, 256, tp, rank))
    with torch.device(device):
        loader.create_weights(layer, {})
    loader.bind_upload_buffer(
        Mxfp4UploadBuffer(loader.staging_bytes, layer.w13_weight.device)
    )
    if device == "cpu":
        # CPU cases isolate checkpoint ownership/assembly from native kernels.
        # Production always invokes the native converter on the target device.
        def reference_convert(weight, scale, target, target_scale):
            q, channel = mxfp4_channel_int4(weight.view(torch.uint8), scale)
            target.copy_(pack_ixinfer(q, layout))
            target_scale.copy_(channel)

        loader.convert_shard = reference_convert
    return loader


@pytest.mark.parametrize(
    "backend,layout,device",
    [
        ("ixinfer_w4a8", "TN", "cpu"),
        ("ixinfer_w4a8", "NN", "cpu"),
        ("ixinfer_w4a8", "TN", "cuda"),
        ("ixinfer_w4a8", "NN", "cuda"),
        ("ixmm_w4a8", "NN", "cuda"),
        ("ixmm_fp4fp8", "fp4fp8", "cuda"),
    ],
)
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1)])
def test_streamed_shards_match_whole_expert(backend, layout, device, tp, rank):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        if (backend.startswith("ixmm") and cuda_arch() != "ivcore30") or (
            backend == "ixinfer_w4a8" and layout == "NN" and cuda_arch() != "ivcore11"
        ):
            pytest.skip("Unsupported device/backend combination")
    torch.manual_seed(43)
    layer = ExpertLayer()
    loader = loader_for(layer, backend, layout, device, tp, rank)
    pointers = {k: v.data_ptr() for k, v in layer.named_parameters()}
    sources = {}
    for shard, n, k in (
        ("w1", 256 * tp, 512),
        ("w3", 256 * tp, 512),
        ("w2", 512, 256 * tp),
    ):
        sources[shard] = (
            torch.randint(0, 256, (2, n, k // 2), dtype=torch.uint8),
            torch.randint(119, 126, (2, n, k // 32), dtype=torch.uint8),
        )
    assert not deliver(layer, "w1", 99, sources["w1"][0][0])
    for expert in (1, 0):
        for shard in ("w3", "w2", "w1"):
            for scale in (True, False) if expert else (False, True):
                src = sources[shard][int(scale)][expert].clone()
                assert deliver(
                    layer,
                    shard,
                    expert + 4,
                    src.view(torch.float8_e8m0fnu) if scale else src,
                    scale,
                )
                src.zero_()  # Loader must own the first pending shard.
    weights = loader.finalize()
    assert not loader._pending
    assert pointers == {k: v.data_ptr() for k, v in layer.named_parameters()}
    assert weights.w13_weight is layer.w13_weight
    for group in ("w13", "w2"):
        local = []
        for scale in (False, True):
            if group == "w13":
                tensor = torch.cat(
                    [
                        sources[shard][int(scale)][:, rank * 256 : (rank + 1) * 256]
                        for shard in ("w1", "w3")
                    ],
                    1,
                )
            else:
                length = 256 // (32 if scale else 2)
                tensor = sources["w2"][int(scale)][
                    :, :, rank * length : (rank + 1) * length
                ]
            local.append(tensor.to(device).contiguous())
        w, s = local
        if backend == "ixmm_fp4fp8":
            from ixformer.inference import functions as ops

            expected, scales = ops.preprocess_weight_for_ixmm(
                w.view(torch.int8), s.view(torch.float8_e8m0fnu)
            )
        else:
            pairs = [mxfp4_channel_int4(w[e], s[e]) for e in range(2)]
            if backend == "ixmm_w4a8":
                from ixmm.gemm.i4_i8_fp16 import preswizzle_w4a8_weight

                expected = preswizzle_w4a8_weight(
                    torch.stack([q.T.contiguous() for q, _ in pairs])
                )
                scales = torch.stack([s * 18 for _, s in pairs])
            else:
                expected = torch.stack([pack_ixinfer(q, layout) for q, _ in pairs])
                scales = torch.stack([s for _, s in pairs])
        torch.testing.assert_close(
            getattr(layer, group + "_weight"), expected, rtol=0, atol=0
        )
        actual_scales = getattr(layer, group + "_weight_scale")
        if device == "cuda" and backend != "ixmm_fp4fp8":
            # Native division is correctly rounded; Torch may use reciprocal
            # multiplication. IxMM adds a multiply by 18 (one more rounding).
            # Packed INT4 bytes above must still match exactly.
            ulps = (actual_scales.view(torch.int32) - scales.view(torch.int32)).abs()
            assert (ulps <= (2 if backend == "ixmm_w4a8" else 1)).all()
        elif backend == "ixmm_fp4fp8":
            torch.testing.assert_close(actual_scales, scales, rtol=2e-7, atol=0)
        else:
            torch.testing.assert_close(actual_scales, scales, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="finalized"):
        loader.finalize()


def test_missing_duplicate_and_failed_loads():
    layer = ExpertLayer()
    loader = loader_for(layer, "ixinfer_w4a8", "TN")
    w = torch.zeros((256, 256), dtype=torch.uint8)
    s = torch.full((256, 16), 120, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="Missing"):
        loader.finalize()
    deliver(layer, "w1", 4, w)
    with pytest.raises(ValueError, match="Duplicate"):
        deliver(layer, "w1", 4, w)
    with pytest.raises(ValueError, match="expected"):
        deliver(layer, "w2", 4, w)
    s.fill_(255)
    with pytest.raises(ValueError, match="NaN"):
        deliver(layer, "w1", 4, s, True)
    with pytest.raises(RuntimeError, match="failed"):
        loader.finalize()


def test_loader_rejects_different_scale_contract():
    from dataclasses import replace

    layer = ExpertLayer()
    loader = loader_for(layer, "ixinfer_w4a8", "NN")
    with pytest.raises(ValueError, match="contract"):
        make_mxfp4_loader(
            replace(loader.weight_spec, code_multiplier=1), loader.load_spec
        )


def test_native_converter_does_not_fall_back_to_torch(monkeypatch):
    from ixformer.inference.backend_registry import registry

    from vllm_iluvatar.model_executor.layers.quantization.common.mxfp4_loader import (
        _native_converter,
    )

    _native_converter.cache_clear()
    monkeypatch.setattr(registry, "available_backends", lambda name: ["torch"])
    with pytest.raises(RuntimeError, match="requires native"):
        _native_converter("mxfp4_to_w4a8")
    _native_converter.cache_clear()
