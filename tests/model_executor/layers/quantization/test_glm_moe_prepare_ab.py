# SPDX-License-Identifier: Apache-2.0
"""R2: same NN checkpoint/input/routing, opt-in fusion off versus on.

GPU suite; Producer only syntax-checks it. The independent CPU dense reference
uses separate old-BF16 and new-FP32 input-quantization algorithms, each at
atol=rtol=0.03. Legacy drift is a separate, unchanged-budget assessment;
algorithm correctness does not approve compatibility or model quality.
Run the optional compatibility diagnostic with --run-legacy-drift -m legacy_drift.
"""

import ixformer.inference.functions as F
import pytest
import torch
from vllm import envs as opt

from tests.model_executor.layers.quantization.moe_test_utils import make_nn_graph_method
from vllm_iluvatar.ilu_util import arch

# 本文件 4 个测试全部在函数体首行断言 is_ivcore30()（NN prepare 融合仅
# 在 ivcore30/CQ 机器可用），没有架构无关用例，故模块级整体跳过。
pytestmark = pytest.mark.skipif(
    not arch.is_ivcore30(),
    reason="requires ivcore30 (CQ); GLM MoE NN prepare fusion suite",
)


def dense_reference(q, scale, raw, scales, ids, weights, layer):
    q, scale = q.float().cpu(), scale.reshape(-1, 1).cpu()
    ids, weights = ids.cpu(), weights.cpu()
    out = torch.zeros(q.shape, dtype=torch.float32)
    start = layer.moe_config.ep_rank * layer.local_num_experts
    for local in range(layer.local_num_experts):
        rows, slots = torch.where(ids == start + local)
        z = (
            (
                (q[rows] @ raw[0][local].float().T)
                * scale[rows]
                * (scales[0][local] * 18)
            )
            .bfloat16()
            .float()
        )
        gate, up = z.chunk(2, -1)
        act = torch.nn.functional.silu(gate) * up
        sa = act.abs().amax(-1, keepdim=True) / 127
        inv = torch.where(sa > 0, sa.reciprocal(), 0)
        qa = (act.bfloat16().float() * inv).round().clamp(-128, 127)
        routed = (
            ((qa @ raw[1][local].float().T) * sa * (scales[1][local] * 18))
            .bfloat16()
            .float()
        )
        out.index_add_(0, rows, routed * weights[rows, slots, None])
    return (out * layer._ilu_output_scale).bfloat16()


def independent_input_quant(x, *, legacy):
    """CPU arithmetic oracle, never derived from the tested fused output."""
    value = x.detach().cpu().float()
    maximum = value.abs().amax(-1, keepdim=True)
    scale = maximum / 127
    inverse = 127 / (maximum + 1e-6)
    if legacy:
        # Native old BF16 quant: inverse and product each round to BF16.
        product = (value.bfloat16() * inverse.bfloat16()).float()
    else:
        product = value * inverse
    return product.round().clamp(-127, 127).to(torch.int8), scale


def input_references(x, native_q, native_scale, *, prequantized=False):
    if prequantized:
        return [(native_q.cpu(), native_scale.cpu())] * 2
    old = independent_input_quant(x, legacy=True)
    new = independent_input_quant(x, legacy=False)
    torch.testing.assert_close(native_q.cpu(), old[0], atol=0, rtol=0)
    torch.testing.assert_close(
        native_scale.cpu().reshape(-1, 1), old[1], atol=1e-12, rtol=2e-6
    )
    return [old, new]


def assess_outputs(outputs, references, assessment, record_property, label):
    import json

    y0, y1 = [output.detach().float().cpu() for output in outputs]
    r0, r1 = [reference.float().cpu() for reference in references]
    observed = y1 - y0
    predicted = r1 - r0
    error = observed.abs()
    # Preserve the exact original pointwise compatibility budget. It no
    # longer follows from separate references; this is an independent gate.
    legacy_bound = 0.06 + 0.06 * r0.abs()
    metrics = {
        "label": label,
        "assessment": assessment,
        "legacy_budget": "0.06 + 0.06 * abs(old_reference)",
        "legacy_budget_violations": int((error > legacy_bound).sum()),
        "observed_max_abs": float(error.max()),
        "predicted_max_abs": float(predicted.abs().max()),
        "observed_rmse": float(observed.square().mean().sqrt()),
        "normalized_l2": float(observed.norm() / y0.norm().clamp_min(1e-30)),
        "drift_residual_max_abs": float((observed - predicted).abs().max()),
        "old_oracle_max_residual": float((y0 - r0).abs().max()),
        "new_oracle_max_residual": float((y1 - r1).abs().max()),
    }
    record_property("drift_" + label, json.dumps(metrics, sort_keys=True))
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if assessment == "legacy_drift":
        assert torch.all(error <= legacy_bound), metrics
    else:
        for enabled, (output, reference) in enumerate(zip(outputs, references)):
            torch.testing.assert_close(
                output.cpu(),
                reference,
                atol=0.03,
                rtol=0.03,
                msg=lambda msg, enabled=enabled: f"{label} enabled={enabled}: {msg}",
            )
        # Consistency envelope includes algorithm drift, not a new
        # compatibility tolerance or a replacement for legacy_drift.
        consistency = predicted.abs() + 0.06 + 0.03 * (r0.abs() + r1.abs())
        assert torch.all(error <= consistency)


@pytest.mark.parametrize(
    "assessment",
    ["algorithm", pytest.param("legacy_drift", marks=pytest.mark.legacy_drift)],
)
@pytest.mark.parametrize(
    "hidden,topk,tokens,kind",
    [
        (256, 6, 17, "random"),
        (3584, 6, 9, "random"),
        (4096, 8, 17, "random"),
        (6144, 8, 3, "random"),
        (256, 8, 4, "half_bin"),
        (256, 8, 4, "small"),
        (256, 8, 4, "zero"),
        (256, 8, 4, "no_local"),
        (256, 8, 4, "prequantized"),
    ],
)
@torch.inference_mode()
def test_full_nn_prepare_off_on(
    monkeypatch, hidden, topk, tokens, kind, assessment, record_property
):
    assert arch.is_ivcore30(), "Ruler must load paired CQ ixformer"
    method, layer, raw, scales = make_nn_graph_method(hidden=hidden)
    layer.top_k = topk
    rng = torch.Generator().manual_seed(309)
    x = torch.randn(tokens, hidden, generator=rng).bfloat16().cuda()
    if kind == "half_bin":
        # BF16 values on/around dynamic quantization half bins; first value
        # fixes amax so this does not become a different normalization case.
        values = (torch.arange(hidden).float().remainder(127) + 0.5) / 127
        x.copy_(values.bfloat16().cuda().expand_as(x))
        x[:, 0] = 1
    elif kind == "small":
        x.mul_(1e-4)
    elif kind == "zero":
        x.zero_()
    ids = torch.randint(8, (tokens, topk), generator=rng, dtype=torch.int32).cuda()
    weights = torch.rand(tokens, topk, generator=rng).cuda()
    if kind == "no_local":
        ids.zero_()  # ep_rank=1 owns experts 4..7.
    q, scale = F.scaled_int8_quant(x)
    quantized = input_references(x, q, scale, prequantized=kind == "prequantized")
    references = [
        dense_reference(qr, sr, raw, scales, ids, weights, layer)
        for qr, sr in quantized
    ]
    record_property(
        "changed_input_bins",
        int(torch.count_nonzero(quantized[0][0] != quantized[1][0])),
    )
    native = F.moe_expand_input_ep_dynamic_scaled_int8
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs["topk"])
        return native(*args, **kwargs)

    monkeypatch.setattr(F, "moe_expand_input_ep_dynamic_scaled_int8", checked)
    inputs = (q, scale, x.dtype) if kind == "prequantized" else x
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(opt, "VLLM_ILU_GLM_MOE_PREPARE_FUSION", enabled)
        output = method.apply(layer, inputs, weights, ids, None, None)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        outputs.append(output.cpu().clone())
    assert calls == ([] if kind == "prequantized" else [topk])
    assess_outputs(
        outputs,
        references,
        assessment,
        record_property,
        f"full_{hidden}_{topk}_{tokens}_{kind}",
    )
    if kind in ("zero", "no_local"):
        assert torch.count_nonzero(outputs[0]) == 0
        assert torch.count_nonzero(outputs[1]) == 0


def standard_experts(method, layer, monkeypatch):
    from types import SimpleNamespace

    from vllm_iluvatar.model_executor.layers.quantization.backends.w4a8 import (
        nn_experts as std,
    )

    experts = std.IluMoEStandardExpertsW4A8NN.__new__(std.IluMoEStandardExpertsW4A8NN)
    experts.local_num_experts = layer.local_num_experts
    experts.format, experts.group_size = "NN", -1
    experts.gemm1_info, experts.gemm2_info = method.gemm1_info, method.gemm2_info
    experts.w13_input_scale = experts.w2_input_scale = experts.swiglu_limit = None
    experts.runner_scaling_factor = layer._ilu_output_scale
    experts.quant_config = SimpleNamespace(
        w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale
    )
    monkeypatch.setattr(std, "get_ep_group", lambda: SimpleNamespace(rank=1))
    return experts


@pytest.mark.parametrize(
    "assessment",
    ["algorithm", pytest.param("legacy_drift", marks=pytest.mark.legacy_drift)],
)
@pytest.mark.parametrize("hidden,topk", [(256, 8), (3584, 6)])
@pytest.mark.parametrize("caller", ["monolithic", "standard"])
@torch.inference_mode()
def test_nn_prepare_ab_dynamic_routing_graph(
    monkeypatch, hidden, topk, caller, assessment, record_property
):
    """Both captured paths consume changed input, routing and weights in-place."""
    from vllm.model_executor.layers.fused_moe import MoEActivation

    assert arch.is_ivcore30(), "requires paired CQ candidate"
    method, layer, raw, scales = make_nn_graph_method(hidden=hidden)
    layer.top_k = topk
    experts = standard_experts(method, layer, monkeypatch)
    rng = torch.Generator().manual_seed(715)
    x = torch.randn(17, hidden, generator=rng).bfloat16().cuda()
    ids = torch.randint(8, (17, topk), generator=rng, dtype=torch.int32).cuda()
    weights = torch.rand(17, topk, generator=rng).cuda()
    native = F.moe_expand_input_ep_dynamic_scaled_int8
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs["topk"])
        return native(*args, **kwargs)

    monkeypatch.setattr(F, "moe_expand_input_ep_dynamic_scaled_int8", checked)
    graphs, outputs = [], []
    for enabled in (False, True):
        monkeypatch.setattr(opt, "VLLM_ILU_GLM_MOE_PREPARE_FUSION", enabled)
        output = torch.empty_like(x)

        def launch(output=output):
            if caller == "monolithic":
                return method.apply(layer, x, weights, ids, None, None)
            return experts.apply(
                output,
                x,
                layer.w13_weight,
                layer.w2_weight,
                weights,
                ids,
                MoEActivation.SILU,
                layer.global_num_experts,
                layer.expert_map,
                None,
                None,
                None,
                None,
            )

        for _ in range(3):
            launch()
        if not enabled:
            assert calls == []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch()
        graphs.append(graph)
        outputs.append(captured)
    assert calls and set(calls) == {topk}
    for routing in ("random", "one_local", "no_local"):
        x.normal_()
        weights.uniform_(0.05, 0.95)
        if routing == "random":
            ids.copy_(torch.randint(8, ids.shape, device=ids.device, dtype=ids.dtype))
        else:
            ids.fill_(4 if routing == "one_local" else 0)
        q, scale = F.scaled_int8_quant(x)
        quantized = input_references(x, q, scale)
        references = [
            dense_reference(qr, sr, raw, scales, ids, weights, layer)
            for qr, sr in quantized
        ]
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        for output in outputs:
            assert torch.isfinite(output).all()
        assess_outputs(
            outputs,
            references,
            assessment,
            record_property,
            f"graph_{caller}_{hidden}_{topk}_{routing}",
        )
        if routing == "no_local":
            assert all(torch.count_nonzero(output) == 0 for output in outputs)


@pytest.mark.parametrize("caller", ["monolithic", "standard"])
@torch.inference_mode()
def test_unpadded_k_retains_legacy_rejection(monkeypatch, caller):
    """hidden != pad_k must bypass fusion; old expand rejects this contract.

    This is not a claim that NN supports automatic input padding. The loaded
    candidate's old non-packed expand requires equal input/output hidden sizes.
    """
    from vllm.model_executor.layers.fused_moe import MoEActivation

    assert arch.is_ivcore30(), "requires paired CQ candidate"
    method, layer, _, _ = make_nn_graph_method(hidden=256)
    layer.top_k = 8
    experts = standard_experts(method, layer, monkeypatch)
    x = torch.randn(2, 192, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(2, 8, device="cuda") / 8
    ids = torch.full((2, 8), 4, device="cuda", dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("hidden != pad_k selected CQ fused quant")

    monkeypatch.setattr(F, "moe_expand_input_ep_dynamic_scaled_int8", forbidden)
    for enabled in (False, True):
        monkeypatch.setattr(opt, "VLLM_ILU_GLM_MOE_PREPARE_FUSION", enabled)
        with pytest.raises(
            RuntimeError, match="non-packed input/output hidden sizes must match"
        ):
            if caller == "monolithic":
                method.apply(layer, x, weights, ids, None, None)
            else:
                output = torch.empty(2, 256, device="cuda", dtype=x.dtype)
                experts.apply(
                    output,
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    weights,
                    ids,
                    MoEActivation.SILU,
                    layer.global_num_experts,
                    layer.expert_map,
                    None,
                    None,
                    None,
                    None,
                )


@pytest.mark.parametrize("hidden", [256, 6144])
@torch.inference_mode()
def test_half_bin_primitive_error_measurement(hidden, record_property):
    """Measure the rounding delta against old BF16 quant, not a shared oracle."""
    assert arch.is_ivcore30(), "requires paired CQ candidate"
    base = ((torch.arange(hidden).float().remainder(127) + 0.5) / 127).bfloat16()
    x = torch.stack((base, base * 1e-6, torch.zeros_like(base))).cuda()
    x[0, 0] = 1
    old_q, old_scale = F.scaled_int8_quant(x)
    mapping = torch.arange(3, device="cuda", dtype=torch.int32)
    q = torch.empty_like(old_q)
    scale = torch.empty(3, device="cuda")
    F.moe_expand_input_ep_dynamic_scaled_int8(x, mapping, q, scale, topk=1)
    delta = (q.int() - old_q.int()).abs()
    maximum = int(delta.max())
    count = int(torch.count_nonzero(delta))
    record_property("max_int8_bin_delta", maximum)
    record_property("changed_int8_elements", count)
    print(f"hidden={hidden} max_int8_bin_delta={maximum} changed_elements={count}")
    assert maximum <= 1
    torch.testing.assert_close(scale, old_scale.reshape(-1), rtol=2e-6, atol=1e-12)
