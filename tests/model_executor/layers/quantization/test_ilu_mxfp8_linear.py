# SPDX-License-Identifier: Apache-2.0
"""MXFP8 dense Linear dispatch and the MXFP8 -> W8A8 INT8 requant.

Both MXFP8 configs used to leave Linear entirely to community dispatch, which
selects a kernel from ``_POSSIBLE_MXFP8_KERNELS[CUDA]``. ``IluMxfp8LinearKernel``
is only prepended there on ivcore30, so on ivcore11 the layer had no runnable
kernel at all. These cover the ivcore11 takeover, that ivcore30 keeps falling
through, and that the requant decodes UE8M0 on the ``(1, 32)`` grid.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.linear import (
    mxfp8_int8_linear_method,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_int8 import (
    quantize_fp8_block_to_w8a8_int8,
)

_BLOCK = 32


def _config(**overrides):
    """An MXFP8 config stand-in; only ``is_layer_excluded`` is consulted."""
    excluded = overrides.pop("excluded", ())
    return SimpleNamespace(
        is_layer_excluded=lambda prefix: prefix in excluded, **overrides
    )


def _select(monkeypatch, architecture, prefix, preference="auto", layer=None, **kw):
    monkeypatch.setattr(arch, "cuda_arch", lambda: architecture)
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": preference}}}
        ),
    )
    layer = object.__new__(LinearBase) if layer is None else layer
    return mxfp8_int8_linear_method(_config(**kw), layer, prefix)


# --- dispatch -------------------------------------------------------------


@pytest.mark.parametrize("preference", ["auto", "w8a8_int8"])
def test_ivcore11_dense_linear_is_claimed(monkeypatch, preference):
    from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.linear import (
        IluMxFp8ToW8A8Int8LinearMethod,
    )

    method = _select(
        monkeypatch, "ivcore11", "model.layers.0.self_attn.o_proj", preference
    )
    assert type(method) is IluMxFp8ToW8A8Int8LinearMethod
    assert method.block_shape == (1, _BLOCK)
    assert method.scheme.format == "TN"
    assert not method.scheme.is_static_input_scheme
    assert not method.supports_pre_processed_weights


@pytest.mark.parametrize("preference", ["auto", "fp8", "w8a8_int8"])
def test_only_ivcore11_is_claimed(monkeypatch, preference):
    """ivcore30 keeps FP8 execution on the patched MXFP8 kernel.

    The takeover is architecture-gated, not backend-gated: ivcore30 accepts an
    explicit ``w8a8_int8`` preference for other checkpoints, and that must not
    drag an MXFP8 model off its working FP8 path.
    """
    prefix = "model.layers.0.self_attn.o_proj"
    assert _select(monkeypatch, "ivcore30", prefix, preference) is None


def test_excluded_and_vision_layers_fall_through(monkeypatch):
    """``super()`` owns every unquantized branch, so return None rather than
    guessing which unquantized method it would have built."""
    excluded = ("lm_head",)
    assert _select(monkeypatch, "ivcore11", "lm_head", excluded=excluded) is None
    assert _select(monkeypatch, "ivcore11", "vision_tower.layers.0.qkv") is None


def test_lm_head_is_not_a_linear_base(monkeypatch):
    """Community routes ``ParallelLMHead`` through ``build_linear_method`` too;
    converting the head to INT8 is out of scope, so it must not be claimed."""
    head = object.__new__(ParallelLMHead)
    assert _select(monkeypatch, "ivcore11", "lm_head", layer=head) is None


def test_both_configs_route_linear_through_the_shared_helper(monkeypatch):
    """``mxfp8`` and ``modelopt_mxfp8`` differ only in the registry name."""
    from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.linear import (
        IluMxFp8ToW8A8Int8LinearMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.configs.modelopt.mxfp8 import (
        IluModelOptMxFp8Config,
    )
    from vllm_iluvatar.model_executor.layers.quantization.configs.mxfp8 import (
        IluMxFp8Config,
    )

    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(additional_config={}),
    )
    prefix = "model.layers.0.self_attn.o_proj"
    for cls in (IluMxFp8Config, IluModelOptMxFp8Config):
        config = cls(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        method = config.get_quant_method(object.__new__(LinearBase), prefix)
        assert type(method) is IluMxFp8ToW8A8Int8LinearMethod, cls.__name__


# --- weight conversion ----------------------------------------------------


def test_mxfp8_grid_requant_reference():
    """UE8M0 bytes must be decoded, not cast: 0x7F is 1.0, not 127.0."""
    torch.manual_seed(7)
    n, k = 65, 288  # a ragged N against the 512-row staging band
    weight = torch.randn(n, k).to(torch.float8_e4m3fn)
    # Exponents around the 127 bias, i.e. scales around 1.0.
    raw = torch.randint(120, 134, (n, k // _BLOCK), dtype=torch.uint8)

    q, scale = quantize_fp8_block_to_w8a8_int8(
        weight, raw.view(torch.float8_e8m0fnu), (1, _BLOCK)
    )

    decoded = raw.view(torch.float8_e8m0fnu).float().repeat_interleave(_BLOCK, dim=1)
    value = weight.float() * decoded
    expected_s = value.abs().amax(dim=1, keepdim=True) / 127.0
    expected_q = torch.round(value / expected_s).clamp(-127, 127).to(torch.int8)

    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_s, rtol=0, atol=0)
    assert scale.shape == (n, 1)


def test_staging_band_does_not_change_the_128_grid_result():
    """Stepping whole row-blocks in bands must be bit-identical to one block
    per step, for a shape that spans several bands."""
    torch.manual_seed(11)
    weight = torch.randn(1100, 256).to(torch.float8_e4m3fn)
    blocks = torch.rand(9, 2, dtype=torch.float32) * 0.03

    q, scale = quantize_fp8_block_to_w8a8_int8(weight, blocks, (128, 128))

    expanded = blocks.repeat_interleave(128, 0).repeat_interleave(128, 1)
    value = weight.float() * expanded[:1100, :256]
    expected_s = value.abs().amax(dim=1, keepdim=True) / 127.0
    expected_q = torch.round(value / expected_s).clamp(-127, 127).to(torch.int8)

    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_s, rtol=0, atol=0)


def test_create_weights_registers_the_checkpoint_contract(monkeypatch):
    """The loader contract is the community MXFP8 one; nothing is restated."""
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    method = _select(monkeypatch, "ivcore11", "model.layers.0.mlp.down_proj")
    layer = nn.Module()
    loader = object()

    method.create_weights(
        layer,
        input_size_per_partition=256,
        output_partition_sizes=[128, 128],
        input_size=256,
        output_size=256,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )

    assert tuple(layer.weight.shape) == (256, 256)
    assert layer.weight.dtype is torch.float8_e4m3fn
    assert tuple(layer.weight_scale.shape) == (256, 256 // _BLOCK)
    # uint8, not e8m0: the checkpoint's storage dtype, decoded at requant.
    assert layer.weight_scale.dtype is torch.uint8
    assert layer.weight.weight_loader is loader
    assert layer.weight_block_size == [1, _BLOCK]
    assert layer.logical_widths == [128, 128]


def test_process_weights_publishes_int8_and_drops_the_block_scale(monkeypatch):
    """The W8A8 scheme reads five names; the MXFP8 block scale is not one."""
    method = _select(monkeypatch, "ivcore11", "model.layers.0.mlp.down_proj")
    monkeypatch.setattr(
        type(method.scheme), "process_weights_after_loading", lambda self, layer: None
    )
    layer = nn.Module()
    layer.logical_widths = [64]
    layer.register_parameter(
        "weight",
        nn.Parameter(torch.randn(64, 128).to(torch.float8_e4m3fn), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        nn.Parameter(
            torch.full((64, 128 // _BLOCK), 0x7F, dtype=torch.uint8),
            requires_grad=False,
        ),
    )

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype is torch.int8
    assert tuple(layer.weight_scale.shape) == (64, 1)
    assert layer.weight_scale.dtype is torch.float32
    assert layer.input_scale is None and layer.azp_adj is None
