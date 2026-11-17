# SPDX-License-Identifier: Apache-2.0
"""Model-specific consumers must stay outside ordinary FP8 Linear selection."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_iluvatar import envs
from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.configs.deepseek.v41_fp8 import (
    IluDeepseekV41Bf16DequantLinearMethod,
    IluDeepseekV41FP8Config,
)


@pytest.mark.parametrize(
    "architecture,enabled", [("ivcore11", False), ("ivcore30", True)]
)
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("preference", ["auto", "fp8", "w8a8_int8"])
def test_grouped_wo_a_keeps_bf16(
    monkeypatch, tp_size, preference, architecture, enabled
):
    monkeypatch.setattr(arch, "cuda_arch", lambda: architecture)
    monkeypatch.setattr(envs, "VLLM_ILU_DS41_GROUPED_INT8_O_PROJ", enabled)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": preference}}}
        ),
    )
    config = SimpleNamespace(
        ignored_layers=[], packed_modules_mapping={}, weight_block_size=[32, 32]
    )
    layer = object.__new__(LinearBase)
    method = IluDeepseekV41FP8Config.get_quant_method(
        config, layer, "model.layers.0.attn.wo_a"
    )
    assert type(method) is IluDeepseekV41Bf16DequantLinearMethod


def test_ignored_bf16_stays_unquantized():
    config = SimpleNamespace(
        ignored_layers=["model.layers.0.e_proj"], packed_modules_mapping={}
    )
    layer = object.__new__(LinearBase)
    method = IluDeepseekV41FP8Config.get_quant_method(
        config, layer, "model.layers.0.e_proj"
    )
    assert type(method) is UnquantizedLinearMethod


@pytest.mark.parametrize("preference", ["auto", "fp8", "w8a8_int8"])
def test_ivcore11_grouped_loader_reuses_existing_bmm(monkeypatch, preference):
    from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
        IluDeepseekV41Fp8GroupedInt8Method,
        IluV4W8A8BMMLinearMethod,
    )

    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(envs, "VLLM_ILU_DS41_GROUPED_INT8_O_PROJ", True)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": preference}}}
        ),
    )
    config = SimpleNamespace(
        ignored_layers=[], packed_modules_mapping={}, weight_block_size=[32, 32]
    )
    method = IluDeepseekV41FP8Config.get_quant_method(
        config, object.__new__(LinearBase), "model.layers.0.attn.wo_a"
    )
    assert type(method) is IluDeepseekV41Fp8GroupedInt8Method
    assert method.apply_quantized.__func__ is IluV4W8A8BMMLinearMethod.apply_quantized
    assert method.apply_with_rope.__func__ is IluV4W8A8BMMLinearMethod.apply_with_rope


def test_grouped_source_finalization_preserves_group_channel_scales():
    from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
        IluDeepseekV41Fp8GroupedInt8Method,
    )

    # Small two-group loading case: differing block scales catch group/channel
    # transposes. This tests producer/layout only, not BMM kernel accuracy.
    layer = torch.nn.Module()
    layer.bmm_geom = SimpleNamespace(
        n_local_groups=2, o_lora_rank=64, heads_per_group=1, head_dim=128
    )
    layer.weight_block_size = [32, 32]
    source = (
        ((torch.arange(128 * 128).reshape(128, 128) % 113) - 56)
        .float()
        .to(torch.float8_e4m3fn)
    )
    scales = torch.tensor(
        [
            [0.125, 0.25, 0.5, 1.0],
            [2.0, 1.0, 0.5, 0.25],
            [0.5, 2.0, 1.0, 0.125],
            [1.0, 0.25, 2.0, 0.5],
        ]
    )
    layer.register_parameter("weight", torch.nn.Parameter(source, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False)
    )
    dense = source.float() * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected_scales = dense.abs().amax(1, keepdim=True) / 127
    expected = (dense / expected_scales).round().clamp(-127, 127).to(torch.int8)
    method = IluDeepseekV41Fp8GroupedInt8Method(SimpleNamespace())
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.W_int8_grouped, expected.view(2, 64, 128).transpose(1, 2))
    assert torch.equal(layer.W_int8_scales, expected_scales.view(2, 64))
    assert not hasattr(layer, "weight") and not hasattr(layer, "weight_scale_inv")
    assert not hasattr(layer, "weight_scale")
