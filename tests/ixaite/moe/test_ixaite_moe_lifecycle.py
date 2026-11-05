# SPDX-License-Identifier: Apache-2.0
"""Check constructor failures, expert loading, numerical output and graph replay."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import MoEActivation

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite.moe import unquantized, w4a8, w8a8


# Keep the fixed expert contract small while exercising empty experts and shared output.
def _config():
    return SimpleNamespace(
        has_bias=False,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        is_lora_enabled=False,
        is_act_and_mul=True,
    )


# Unsupported storage must fail before the constructor binds kernels or allocates weights.
@pytest.mark.parametrize(
    "params",
    [
        {"gemm_format": "NN", "pack_version": 2},
        {"gemm_format": "TN", "pack_version": 1},
    ],
)
def test_w4a8_rejects_missing_storage_contract(monkeypatch, params):
    def unexpected_bind():
        raise AssertionError("unsupported format reached kernel binding")

    monkeypatch.setattr(w4a8, "configure_moe_providers", unexpected_bind)
    with pytest.raises(RuntimeError, match="TN, pack_version=2"):
        w4a8.IluCompressedTensorsW4A8Int8MoEMethod(
            None, None, _config(), ilu_params=params
        )


# The library has no W8A8 grouped compute; partial routing support is insufficient.
def test_w8a8_missing_compute_fails_before_parent_init(monkeypatch):
    monkeypatch.setattr(
        w8a8, "load_ixaite_python_op", lambda *args: (None, "missing test kernel")
    )
    with pytest.raises(RuntimeError, match="moe_w8a8_group_gemm.*missing test kernel"):
        w8a8.IluCompressedTensorsW8A8Int8MoEMethod(None, None, _config())


# Fixed semantic differences cannot be silently discarded by the SiLU pipeline.
@pytest.mark.parametrize(
    "field,value",
    [("has_bias", True), ("is_lora_enabled", True), ("in_dtype", torch.float32)],
)
def test_unquantized_rejects_unsupported_config(field, value):
    config = _config()
    setattr(config, field, value)
    with pytest.raises(RuntimeError, match="FP16/BF16, SiLU"):
        unquantized.IluUnquantizedFusedMoEMethod(config)


# Independent fp32 per-expert projections check routing weights and the shared residual.
def _reference(layer, x, weights, ids, shared):
    output = torch.zeros_like(x, dtype=torch.float32)
    intermediate = layer.w2_weight.shape[-1]
    # 1. Select each expert from the route table and compute its gate/up projection.
    # 2. Apply SiLU, down-project, and accumulate using the corresponding route weight.
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = (x[token].float() @ layer.w13_weight[expert].float().T).split(
                intermediate
            )
            activated = torch.nn.functional.silu(gate) * up
            output[token] += (activated @ layer.w2_weight[expert].float().T) * weights[
                token, slot
            ]
    return output + shared.float()


# Exercise the real inherited create/load lifecycle and capture twice with changed inputs.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("tokens", [1, 4])
def test_unquantized_experts_and_graph_match_torch(tokens, default_vllm_config):
    torch.manual_seed(3829)
    with forbid_ixformer_calls(), torch.device("cuda"):
        method = unquantized.IluUnquantizedFusedMoEMethod(_config())
        layer = torch.nn.Module()
        layer.apply_router_weight_on_input = False
        layer.swiglu_limit = None
        method.create_weights(layer, 8, 256, 128, torch.bfloat16)
        with torch.no_grad():
            layer.w13_weight.copy_(torch.randn_like(layer.w13_weight) * 0.1)
            layer.w2_weight.copy_(torch.randn_like(layer.w2_weight) * 0.1)
        method.process_weights_after_loading(layer)
        x = torch.randn(tokens, 256, dtype=torch.bfloat16) * 0.2
        # Six experts remain empty; both routes repeatedly target the other two.
        ids = torch.tensor([[0, 3]] * tokens, dtype=torch.int32)
        weights = torch.tensor([[0.3, 0.7]] * tokens, dtype=torch.float32)
        shared = torch.randn_like(x) * 0.1

        def run():
            return method.apply(layer, x, weights, ids, None, shared)

        expected = _reference(layer, x, weights, ids, shared)
        output = run()
        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(2):
            x.mul_(0.7)
            expected = _reference(layer, x, weights, ids, shared)
            graph.replay()
            torch.testing.assert_close(captured.float(), expected, rtol=2e-2, atol=2e-2)


# Verify the actual gate class uses its selected fp32 GEMM for the shared-expert batch.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_gate_class_single_token_matches_torch(weight_dtype):
    from vllm_iluvatar.backends.ixaite.moe.gate_linear import IluGateLinear

    with forbid_ixformer_calls():
        gate = object.__new__(IluGateLinear)
        torch.nn.Module.__init__(gate)
        gate.weight = torch.nn.Parameter(
            torch.randn(256, 2048, device="cuda", dtype=weight_dtype)
        )
        gate.bias = None
        gate.skip_bias_add = True
        gate.out_dtype = torch.float32
        x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        output, bias = gate.forward(x)
        # Pure ixaite: bf16 act x bf16 weight -> fp32 (fp32 checkpoint weights
        # are cast once; do not upcast activations and call F.linear/ixblas).
        w_bf16 = gate.weight.detach().to(torch.bfloat16)
        torch.testing.assert_close(
            output, (x.float() @ w_bf16.float().T).to(torch.float32), rtol=2e-2, atol=2e-2
        )
        assert output.dtype == torch.float32
        assert bias is None


def _bare_fp32_gate(n: int, k: int):
    from vllm_iluvatar.backends.ixaite.moe.gate_linear import IluGateLinear

    gate = object.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.randn(n, k, device="cuda", dtype=torch.float32)
    )
    gate.bias = None
    gate.skip_bias_add = True
    gate.out_dtype = torch.float32
    gate.return_bias = True
    return gate


def _bf16_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1]).float()
    ref = flat @ weight.detach().to(torch.bfloat16).float().T
    return ref.reshape(x.shape[:-1] + (weight.shape[0],))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_gate_token_count_uses_numel_not_leading_dim():
    """shape[0] <= 8 must not enter the router GEMM when numel//K is wider.

    T=2..8 is the phys-16 MMA. T>8 hits the IXCC-rejected epilogue, so a
    rank-3 prefill has to fall through to the replicated linear.
    """
    from vllm_iluvatar.backends.ixaite.moe.gate_linear import _router_tokens

    gate = _bare_fp32_gate(256, 2048)
    seen = {"n": 0}

    class _Quant:
        def apply(self, layer, x, bias):
            seen["n"] += 1
            flat = x.reshape(-1, x.shape[-1])
            y = torch.nn.functional.linear(flat, layer.weight)
            return y.reshape(x.shape[:-1] + (layer.weight.shape[0],))

    gate.quant_method = _Quant()
    wide = torch.randn(8, 8, 2048, device="cuda", dtype=torch.bfloat16)
    assert _router_tokens(wide) == 64
    with forbid_ixformer_calls():
        output, bias = gate.forward(wide)
    assert seen["n"] == 1
    assert bias is None
    assert output.shape == (8, 8, gate.weight.shape[0])
    assert output.dtype == torch.float32

    seen["n"] = 0
    decode = torch.randn(1, 8, 2048, device="cuda", dtype=torch.bfloat16)
    assert _router_tokens(decode) == 8
    with forbid_ixformer_calls():
        output, bias = gate.forward(decode)
    assert seen["n"] == 0
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, _bf16_reference(decode, gate.weight), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_fp32_gate_reload_refreshes_bf16_cache():
    gate = _bare_fp32_gate(256, 2048)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    with forbid_ixformer_calls():
        first, _ = gate.forward(x)
        torch.testing.assert_close(
            first, _bf16_reference(x, gate.weight), rtol=2e-2, atol=2e-2
        )
        reloaded = torch.randn_like(gate.weight)
        # vLLM reload enters through weight_loader, which data.copy_'s the
        # same storage (data_ptr and _version stay unchanged).
        gate.weight_loader(gate.weight, reloaded)
        second, _ = gate.forward(x)
    torch.testing.assert_close(
        second, _bf16_reference(x, gate.weight), rtol=2e-2, atol=2e-2
    )
    assert not torch.allclose(first, second, rtol=1e-3, atol=1e-3)


# The version-2 TN byte groups store sixteen low nibbles followed by sixteen high ones.
def _int4_weights(weight):
    byte = weight.cpu().to(torch.uint8).reshape(*weight.shape[:2], -1, 16)
    nibbles = [((byte >> shift) & 15).to(torch.int32) for shift in (0, 4)]
    return torch.cat(
        [torch.where(part >= 8, part - 16, part) for part in nibbles], -1
    ).flatten(2)


# Match round-to-even INT8 quantization without calling either kernel implementation.
def _quantize(row):
    scale = row.abs().max() / 127
    return (row / scale).round().clamp(-128, 127).to(torch.int32), scale


# Independent integer products check the packing multiplier and BF16 storage at both GEMMs.
def _w4_reference(layer, x, weights, ids, shared, start_expert=0):
    w13, w2 = (_int4_weights(value) for value in (layer.w13_weight, layer.w2_weight))
    s13, s2 = (
        value.cpu().flatten(1)
        for value in (layer.w13_weight_scale, layer.w2_weight_scale)
    )
    x, ids, weights = (value.cpu() for value in (x, ids, weights))
    output = torch.zeros_like(x, dtype=torch.float32)
    # 1. Quantize the input and unpack each selected expert's version-2 weights.
    # 2. Round the first projection to BF16, apply SwiGLU, and quantize again.
    # 3. Round the down projection to BF16 before weighting and summing the routes.
    for token in range(x.shape[0]):
        quant, scale = _quantize(x[token].float())
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot]) - start_expert
            if not 0 <= expert < w13.shape[0]:
                continue
            projected = (
                ((w13[expert] @ quant).float() * scale * s13[expert] * 18)
                .bfloat16()
                .float()
            )
            gate, up = projected.chunk(2)
            activated, scale2 = _quantize(torch.nn.functional.silu(gate) * up)
            down = (
                ((w2[expert] @ activated).float() * scale2 * s2[expert] * 18)
                .bfloat16()
                .float()
            )
            output[token] += down * weights[token, slot]
    return output + shared.float().cpu()


# Check empty experts, BS1 GEMV and GPU-planned prefill through create/load and replay.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize(
    "tokens,path",
    [
        (1, "decode"),
        (8, "decode"),
        (32, "general"),
        (4, "ep_decode"),
        (4, "ep_general"),
        (4, "ep_empty"),
        (4, "standard"),
    ],
)
def test_w4a8_lifecycle_and_graph_match_integer_reference(
    tokens, path, default_vllm_config, monkeypatch
):
    torch.manual_seed(3829)
    with forbid_ixformer_calls(), torch.device("cuda"):
        method = w4a8.IluCompressedTensorsW4A8Int8MoEMethod(
            QuantizationArgs(num_bits=4, strategy="channel", symmetric=True),
            QuantizationArgs(
                num_bits=8, strategy="token", symmetric=True, dynamic=True
            ),
            _config(),
            ilu_params={"gemm_format": "TN", "pack_version": 2},
        )
        layer = torch.nn.Module()
        layer.activation = MoEActivation.SILU
        layer.swiglu_limit = None
        use_ep = path.startswith("ep_")
        layer.expert_map = torch.arange(8, dtype=torch.int32) if use_ep else None
        layer.global_num_experts = 8
        layer.local_num_experts = 4 if use_ep else 8
        layer.moe_config = SimpleNamespace(ep_rank=1 if use_ep else 0)
        layer.top_k = 2
        method.create_weights(layer, layer.local_num_experts, 256, 256, torch.bfloat16)
        with torch.no_grad():
            layer.w13_weight.random_(-128, 128)
            layer.w2_weight.random_(-128, 128)
            layer.w13_weight_scale.fill_(1e-3)
            layer.w2_weight_scale.fill_(1e-3)
        method.process_weights_after_loading(layer)
        x = torch.randn(tokens, 256, dtype=torch.bfloat16).clamp(-1, 1)
        selected_ids = [0, 1] if path == "ep_empty" else [1, 5] if use_ep else [0, 3]
        ids = torch.tensor([selected_ids] * tokens, dtype=torch.int32)
        weights = torch.tensor([[0.3, 0.7]] * tokens, dtype=torch.float32)
        shared = torch.randn_like(x) * 0.1
        if path == "standard":
            from vllm_iluvatar.backends.ixaite.moe import standard_w4a8

            experts = standard_w4a8.IluMoEStandardExpertsW4A8(
                _config(),
                method.get_fused_moe_quant_config(layer),
                "w4a8",
                method.gemm1_info,
                method.gemm2_info,
                group_size=-1,
            )
            experts.local_num_experts = 8
            monkeypatch.setattr(
                standard_w4a8, "get_ep_group", lambda: SimpleNamespace(rank=0)
            )
            standard_output = torch.empty_like(x)
            shared.zero_()

        def run():
            if path == "standard":
                return experts.apply(
                    standard_output,
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    weights,
                    ids,
                    layer.activation,
                    8,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            if use_ep:
                return method._apply_ep(
                    layer,
                    x,
                    weights,
                    ids,
                    shared,
                    run_decode=path in ("ep_decode", "ep_empty"),
                )
            return getattr(method, "_apply_" + path)(layer, x, weights, ids, shared)

        start_expert = 4 if use_ep else 0
        expected = _w4_reference(layer, x, weights, ids, shared, start_expert)
        output = run()
        torch.testing.assert_close(output.float().cpu(), expected, rtol=1e-2, atol=1e-2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(2):
            x.mul_(0.7)
            expected = _w4_reference(layer, x, weights, ids, shared, start_expert)
            graph.replay()
            torch.testing.assert_close(
                captured.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )
