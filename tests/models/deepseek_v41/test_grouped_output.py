# SPDX-License-Identifier: Apache-2.0
"""Grouped output loading, capability fallback and single inverse ownership."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_iluvatar.model_executor.layers.quantization.ilu_v4_bmm import (
    IluV4W8A8BMMLinearMethod as Int8Method,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_v4_bmm import (
    IluV4W8A16BMMLinearMethod as OriginalMethod,
)
from vllm_iluvatar.models.deepseek_v41 import attention as module


def geometry(groups=2, channels=64, heads=4, dim=64):
    return NS(
        n_local_groups=groups, o_lora_rank=channels, heads_per_group=heads, head_dim=dim
    )


@pytest.mark.parametrize("format", ["TN", "NN"])
@pytest.mark.parametrize("groups", [1, 2, 8])
def test_grouped_loader_keeps_one_int8_layout(format, groups, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = torch.nn.Module()
    layer.bmm_geom = geometry(groups=groups)
    method = OriginalMethod(None, {"bmm_w8a16_gemm_format": "k16n16"})
    method.create_weights(
        layer,
        256,
        [groups * 64],
        256,
        groups * 64,
        torch.bfloat16,
        weight_loader=lambda *args: None,
    )
    original = torch.arange(groups * 64 * 256).reshape(groups * 64, 256).to(torch.int8)
    scales = torch.arange(groups * 64).float().reshape(-1, 1) + 1
    layer.weight.data.copy_(original)
    layer.weight_scale.data.copy_(scales)
    replacement = Int8Method(None, {"bmm_w8a8_gemm_format": format})
    replacement.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight") and not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "W_repacked")
    stored = layer.W_int8_grouped
    restored = stored if format == "TN" else stored.transpose(1, 2)
    torch.testing.assert_close(restored, original.view(groups, 64, 256))
    torch.testing.assert_close(layer.W_int8_scales, scales.view(groups, 64))
    assert stored.is_contiguous() and stored.dtype == torch.int8


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("channels", [63, 64, 1024])
def test_layer_selection_preserves_unsupported_geometry(enabled, channels, monkeypatch):
    layer = module.IluDeepseekV41Attention.__new__(module.IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    original = OriginalMethod(None, {"bmm_w8a16_gemm_format": "k16n16"})
    layer.wo_a = NS(quant_method=original, bmm_geom=geometry(channels=channels))
    layer.rope_head_dim = 64
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS41_GROUPED_INT8_O_PROJ", enabled)
    layer._configure_grouped_int8_output(torch.bfloat16)
    selected = enabled and channels % 64 == 0
    assert layer._grouped_int8_o_proj is selected
    assert isinstance(layer.wo_a.quant_method, Int8Method) is selected
    if not selected:
        assert layer.wo_a.quant_method is original


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads,dim", [(4, 64), (8, 512), (40, 512), (4, 66)])
def test_native_capability_is_geometry_based(dtype, heads, dim):
    expected = dtype != torch.float32 and dim % 8 == 0 and heads * dim <= 16384
    assert (
        Int8Method.supports_geometry(geometry(heads=heads, dim=dim), dtype, 64)
        is expected
    )


@pytest.mark.parametrize("format", [None, "i4g32", "i8g64"])
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("rows", [0, 6, 129])
def test_grouped_projection_owns_the_only_inverse(format, ratio, rows, monkeypatch):
    layer = module.IluDeepseekV41Attention.__new__(module.IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.compressed_kv_format, layer.compress_ratio = format, ratio
    layer.n_local_heads, layer.head_dim = 2, 512
    layer._grouped_int8_o_proj = True
    layer.rotary_emb = NS(cos_sin_cache=torch.empty(256, 64))
    states = torch.zeros(rows, 2, 512, dtype=torch.bfloat16)
    positions = torch.arange(rows)
    events = []
    layer._run_parallel_input_projections = lambda x: (x, None, None)
    layer._split_qkv_and_norm = lambda x: (x, x)

    def prepare(*args):
        assert not layer._uses_fused_inverse_rope()
        args[-1].zero_()

    def projection(owner, output, pos, table):
        assert pos is positions and table is layer.cos_sin
        events.append("projection")
        return output + 3

    def forbidden(*args, **kwargs):
        pytest.fail("Grouped output rotated outside its projection")

    layer._prepare_and_attn = prepare
    layer._o_proj = forbidden
    layer.wo_a = NS(quant_method=NS(apply_with_rope=projection))
    layer.wo_b = lambda x: x
    monkeypatch.setattr(module, "apply_rope", forbidden)
    result = layer(positions, states)
    assert events == ["projection"]
    assert bool((result == 3).all())
    assert not torch.cuda.is_initialized()


def test_empty_grouped_output_does_not_launch_native(monkeypatch):
    import ixformer.inference.functions as ops

    def forbidden(*args):
        pytest.fail("An empty projection launched a native kernel")

    monkeypatch.setattr(ops, "ds4_inv_rope_int8_quant", forbidden)
    layer = NS(bmm_geom=geometry())
    result = Int8Method(None, {}).apply_with_rope(
        layer,
        torch.empty(0, 8, 64),
        torch.empty(0, dtype=torch.int64),
        torch.empty(1, 64),
    )
    assert result.shape == (0, 2, 64)
