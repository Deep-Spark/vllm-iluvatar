# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite layer_norm routing and the Qwen3.5 ViT LayerNorm swap.

Covers:
  * The community vision-tower attributes the swap depends on.
  * ``qwen35_vision_layernorm_fusion`` selection rules.
  * ``IluVisionLayerNorm.forward`` routing and its dtype fallback.
  * Provider selection for the ``layer_norm`` op.
  * Numerics against ``torch.nn.LayerNorm`` at the ViT width.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import Mock

import pytest
import torch
from conftest import requires_cuda
from torch import nn

from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5


def _assigned_attrs(cls: type) -> set[str]:
    """Return the ``self.<name>`` targets assigned in ``cls.__init__``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return names


def test_community_vision_tower_still_exposes_the_swapped_norms():
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3_VisionBlock,
        Qwen3_VisionPatchMerger,
    )

    block_attrs = _assigned_attrs(Qwen3_VisionBlock)
    assert {"norm1", "norm2"} <= block_attrs
    # The merger's entry norm is `norm`, not `ln_q`.
    assert "norm" in _assigned_attrs(Qwen3_VisionPatchMerger)


def test_vision_layernorm_fusion_swaps_trailing_dim_layernorms():
    visual = nn.Module()
    visual.norm1 = nn.LayerNorm(1152, eps=1e-6)
    visual.block = nn.Module()
    visual.block.norm2 = nn.LayerNorm(1152, eps=1e-6)
    visual.merger = nn.Module()
    visual.merger.norm = nn.LayerNorm(4608, eps=1e-6)

    ilu_qwen3_5.qwen35_vision_layernorm_fusion(visual)

    for module in (visual.norm1, visual.block.norm2, visual.merger.norm):
        assert module.__class__ is ilu_qwen3_5.IluVisionLayerNorm
    # Weight, bias and eps survive a __class__ swap.
    assert visual.norm1.normalized_shape == (1152,)
    assert visual.norm1.eps == 1e-6
    assert visual.norm1.weight.shape == (1152,)


def test_vision_layernorm_fusion_skips_multi_axis_and_subclasses():
    class OtherLayerNorm(nn.LayerNorm):
        pass

    visual = nn.Module()
    visual.multi_axis = nn.LayerNorm((4, 1152))
    visual.subclassed = OtherLayerNorm(1152)
    visual.rms = nn.RMSNorm(1152)

    ilu_qwen3_5.qwen35_vision_layernorm_fusion(visual)

    assert visual.multi_axis.__class__ is nn.LayerNorm
    assert visual.subclassed.__class__ is OtherLayerNorm
    assert visual.rms.__class__ is nn.RMSNorm


def test_vision_layernorm_fusion_tolerates_a_text_only_model():
    ilu_qwen3_5.qwen35_vision_layernorm_fusion(None)


def test_vision_layernorm_forward_calls_the_routed_op(monkeypatch):
    routed = Mock(return_value=torch.zeros(2, 8))
    monkeypatch.setattr(ops, "layer_norm", routed, raising=False)
    module = nn.LayerNorm(8, eps=1e-6)
    module.__class__ = ilu_qwen3_5.IluVisionLayerNorm
    x = torch.randn(2, 8)

    result = module(x)

    assert result is routed.return_value
    args = routed.call_args.args
    assert args[0] is x
    assert args[1] is module.weight
    assert args[2] is module.bias
    assert args[3] == 1e-6


def test_vision_layernorm_forward_falls_back_on_a_param_dtype_mismatch(monkeypatch):
    routed = Mock()
    monkeypatch.setattr(ops, "layer_norm", routed, raising=False)
    module = nn.LayerNorm(8, eps=1e-6).to(torch.float32)
    module.__class__ = ilu_qwen3_5.IluVisionLayerNorm
    x = torch.randn(2, 8, dtype=torch.bfloat16)

    result = module(x)

    routed.assert_not_called()
    assert result.dtype is torch.bfloat16


# The private export must bind the Python entry without adding a baseline wrapper.
def test_layer_norm_binds_ixaite_python_entry(monkeypatch):
    kernel = Mock(return_value=object())
    loader = Mock(return_value=(kernel, "available"))
    monkeypatch.setattr(ops, "load_ixaite_python_op", loader)
    x = torch.empty(2, 8)
    weight = object()
    bias = object()

    implementation, _ = ops._load_ixaite_routed_op("layer_norm")
    result = implementation(x, weight, bias, 1e-6)

    loader.assert_called_once_with("layer_norm", "layer_norm")
    kernel.assert_called_once_with(x, weight, bias, 1e-6)
    assert result is kernel.return_value


@requires_cuda
@pytest.mark.parametrize("rows", [1, 784, 3136])
@torch.inference_mode()
def test_vision_layernorm_matches_nn_layernorm(monkeypatch, rows: int) -> None:
    """The swapped module must reproduce the community forward on ViT shapes."""
    import ixaite

    from vllm_iluvatar.ilu_util.arch import cuda_arch

    ixaite.initialize(cuda_arch())
    monkeypatch.setattr(ops, "layer_norm", ixaite.layer_norm, raising=False)

    torch.manual_seed(rows)
    module = nn.LayerNorm(1152, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn((rows, 1, 1152), device="cuda", dtype=torch.bfloat16)
    expected = module(x)

    module.__class__ = ilu_qwen3_5.IluVisionLayerNorm
    from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls

    with forbid_ixformer_calls():
        actual = module(x)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
