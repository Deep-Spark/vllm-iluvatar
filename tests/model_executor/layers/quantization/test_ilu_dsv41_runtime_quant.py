# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 runtime method selection and real W8A8 loading/compute contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ixformer.inference.functions as ops
import pytest
import torch
from conftest import requires_corex30, requires_cuda
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase

from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v41_fp8 import (
    IluDeepseekV41FP8Config,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.linear.dsv41_fp8_to_w8a8 import (
    IluDsv41Fp8ToW8A8GroupedLinearMethod,
    IluDsv41Fp8ToW8A8LinearMethod,
)

EXAMPLE = (
    Path(__file__).resolve().parents[4]
    / "examples/iluvatar-runtime-quant"
    / "deepseek-ai/DeepSeek-V4.1-Flash/w4a8-runtime.json"
)


class Linear(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def select(prefix, monkeypatch, enabled=True, mutation=None):
    payload = json.loads(EXAMPLE.read_text())
    payload["source"] = "inline"
    payload["enabled"] = enabled
    if mutation is not None:
        mutation(payload)
    config = VllmConfig()
    hf = SimpleNamespace(model_type="deepseek_v41_text", expert_dtype="fp4")
    config.model_config = SimpleNamespace(
        hf_config=hf, hf_text_config=hf, dtype=torch.bfloat16
    )
    config.additional_config = {"ilu_quant": payload}
    quant = IluDeepseekV41FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = Linear()
    with set_current_vllm_config(config):
        method = quant.get_quant_method(layer, prefix)
    return layer, method


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("attn.fused_wqa_wkv", IluDsv41Fp8ToW8A8LinearMethod),
        ("attn.wq_b", IluDsv41Fp8ToW8A8LinearMethod),
        ("attn.wo_b", IluDsv41Fp8ToW8A8LinearMethod),
        ("attn.indexer.wq_b", IluDsv41Fp8ToW8A8LinearMethod),
        ("ffn.shared_experts.gate_up_proj", IluDsv41Fp8ToW8A8LinearMethod),
        ("ffn.shared_experts.down_proj", IluDsv41Fp8ToW8A8LinearMethod),
        ("attn.wo_a", IluDsv41Fp8ToW8A8GroupedLinearMethod),
    ],
)
def test_dispatch_and_disabled(suffix, expected, monkeypatch):
    prefix = "model.layers.0." + suffix
    _, method = select(prefix, monkeypatch)
    assert isinstance(method, expected)
    _, original = select(prefix, monkeypatch, enabled=False)
    assert not isinstance(original, IluDsv41Fp8ToW8A8LinearMethod)


@pytest.mark.parametrize(
    "field,value", [("block_shape", [128, 128]), ("scale_format", "fp32")]
)
def test_reject_wrong_source_contract(field, value, monkeypatch):
    def mutate(payload):
        payload["config_groups"]["dsv41_dense_fp8_to_w8a8"]["weights"]["source"][
            field
        ] = value

    with pytest.raises(ValueError, match="block_shape"):
        select("model.layers.0.attn.wq_b", monkeypatch, mutation=mutate)


@requires_cuda
@pytest.mark.parametrize(
    "grouped,rows", [(False, 1), (False, 80), (True, 1), (True, 80)]
)
def test_load_and_compute(grouped, rows, monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    torch.manual_seed(43)
    groups, n, k = (2 if grouped else 1), 128, 256
    prefix = "model.layers.0.attn." + ("wo_a" if grouped else "wq_b")

    def only_tested_group(payload):
        key = "dsv41_grouped_wo_a_to_w8a8" if grouped else "dsv41_dense_fp8_to_w8a8"
        payload["config_groups"] = {key: payload["config_groups"][key]}

    layer, method = select(prefix, monkeypatch, mutation=only_tested_group)
    loader = Mock()
    with set_current_vllm_config(VllmConfig()):
        method.create_weights(
            layer, k, [groups * n], k, groups * n, torch.bfloat16, weight_loader=loader
        )
    assert layer.weight.weight_loader is loader
    values = torch.randn(groups * n, k).to(torch.float8_e4m3fn)
    scales = torch.full((groups * n // 32, k // 32), 0.5).to(torch.float8_e8m0fnu)
    layer.weight.data = values.cuda()
    layer.weight_scale_inv.data = scales.cuda()
    dequantized = values.cuda().float() * scales.cuda().float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    channel = dequantized.abs().amax(-1, keepdim=True) / 127
    expected_w = (dequantized / channel).round().clamp(-127, 127).to(torch.int8)
    if grouped:
        layer.bmm_geom = SimpleNamespace(
            n_local_groups=groups, o_lora_rank=n, heads_per_group=1, head_dim=k
        )
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "weight_scale_inv")
    if grouped:
        assert not hasattr(layer, "weight")
        torch.testing.assert_close(
            layer.W_int8_grouped,
            expected_w.reshape(groups, n, k).transpose(1, 2),
            atol=0,
            rtol=0,
        )
        q = torch.randint(-16, 17, (groups, rows, k), device="cuda", dtype=torch.int8)
        sa = torch.rand(groups, rows, device="cuda") / 128
        call = lambda: method.apply_quantized(layer, q, sa, output_dtype=torch.bfloat16)
        reference = lambda: ops.w8a8(
            q,
            layer.W_int8_grouped,
            sa,
            layer.W_int8_scales,
            format="NN",
            batched_scales=True,
            out_dtype=torch.bfloat16,
            backend="torch",
        ).transpose(0, 1)
        from ixformer.inference.backend_registry import registry

        if "ixmm" in registry.available_backends("w8a8"):
            monkeypatch.setattr(
                ops, "w8a8_bmm", Mock(side_effect=AssertionError("legacy BMM called"))
            )
        input_tensor = q
    else:
        input_tensor = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        call = lambda: method.apply(layer, input_tensor)

        def reference():
            q, sa = ops.scaled_int8_quant(input_tensor)
            return (
                (q.float() @ expected_w.cuda().float().T)
                * sa.reshape(-1, 1)
                * channel.cuda().T
            ).bfloat16()

    actual = call()
    torch.testing.assert_close(actual, reference(), atol=0.02, rtol=0.008)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    input_tensor.neg_()
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0.02, rtol=0.008)


@requires_corex30
def test_mxfp4_direct_nn_matches_native_packing():
    from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.moe.mxfp4_to_w4a8_nn import (
        quantize_mxfp4_to_w4a8_nn,
    )

    torch.manual_seed(43)
    weight = torch.randint(0, 256, (128, 128), dtype=torch.uint8, device="cuda")
    scale = torch.randint(120, 128, (128, 8), dtype=torch.uint8, device="cuda")
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    code = torch.stack((weight & 15, weight >> 4), -1).reshape(128, 256)
    value = table[code.long()] * torch.exp2(scale.float() - 127).repeat_interleave(
        32, 1
    )
    channel = value.abs().amax(-1, keepdim=True) / 127
    q8 = (value / channel).round().clamp(-127, 127).to(torch.int8)
    packed, second, _ = ops.quant_repack_int4(q8.unsqueeze(0), -1, 1, "NN", False)
    actual, actual_scale = quantize_mxfp4_to_w4a8_nn(weight, scale)
    torch.testing.assert_close(actual, packed[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_scale, channel.T * second[0], atol=0, rtol=0)
    zeros, zero_scale = quantize_mxfp4_to_w4a8_nn(torch.zeros_like(weight), scale)
    assert not zeros.count_nonzero()
    assert torch.equal(zero_scale, torch.ones_like(zero_scale))
