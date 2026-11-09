# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5 Vision GELU provider routing and model wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from conftest import requires_cuda

from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5


def test_startup_binds_ixaite_when_supported(monkeypatch) -> None:
    ixaite_op = Mock(return_value="ixaite")
    loader = Mock(return_value=(ixaite_op, "supported on ivcore11"))
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(ilu_qwen3_5, "load_ixaite_torch_op", loader)

    ilu_qwen3_5.configure_vision_gelu_provider()

    assert ilu_qwen3_5._VISION_GELU is ixaite_op
    loader.assert_called_once_with("gelu")


def test_forced_ixaite_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        ilu_qwen3_5,
        "load_ixaite_torch_op",
        Mock(return_value=(None, "not validated on this arch")),
    )

    with pytest.raises(RuntimeError, match="not validated on this arch"):
        ilu_qwen3_5.configure_vision_gelu_provider()


def _fake_visual() -> SimpleNamespace:
    blocks = [
        SimpleNamespace(mlp=SimpleNamespace(act_fn=torch.nn.GELU("tanh")))
        for _ in range(27)
    ]
    return SimpleNamespace(
        blocks=blocks,
        merger=SimpleNamespace(act_fn=torch.nn.GELU()),
        deepstack_merger_list=[SimpleNamespace(act_fn=torch.nn.GELU())],
    )


def test_model_wiring_replaces_only_target_gelus() -> None:
    visual = _fake_visual()
    deepstack_activation = visual.deepstack_merger_list[0].act_fn

    ilu_qwen3_5.qwen35_vision_gelu(visual)

    assert len(visual.blocks) == 27
    assert all(
        isinstance(block.mlp.act_fn, ilu_qwen3_5._VisionGELU)
        and block.mlp.act_fn.approximate == "tanh"
        for block in visual.blocks
    )
    assert isinstance(visual.merger.act_fn, ilu_qwen3_5._VisionGELU)
    assert visual.merger.act_fn.approximate == "none"
    assert visual.deepstack_merger_list[0].act_fn is deepstack_activation


def test_dense_constructor_applies_vision_wiring(monkeypatch) -> None:
    visual = object()
    language_model = SimpleNamespace(model=object())

    def community_init(self, *, vllm_config, prefix) -> None:
        del vllm_config, prefix
        torch.nn.Module.__init__(self)
        self.visual = visual
        self.language_model = language_model
        self.multimodal_config = SimpleNamespace(video_pruning_method=None)

    vision_wiring = Mock()
    vision_attention_wiring = Mock()
    attention_wiring = Mock()
    layernorm_wiring = Mock()
    monkeypatch.setattr(
        ilu_qwen3_5._Qwen3_5ForConditionalGeneration,
        "__init__",
        community_init,
    )
    monkeypatch.setattr(ilu_qwen3_5, "qwen35_vision_gelu", vision_wiring)
    monkeypatch.setattr(
        ilu_qwen3_5,
        "qwen35_vision_attention_fusion",
        vision_attention_wiring,
    )
    monkeypatch.setattr(ilu_qwen3_5, "qwen35_attention_fusion", attention_wiring)
    monkeypatch.setattr(
        ilu_qwen3_5,
        "qwen35_vision_layernorm_fusion",
        layernorm_wiring,
    )

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align")
    )
    ilu_qwen3_5.Qwen3_5ForConditionalGeneration(vllm_config=vllm_config)

    vision_wiring.assert_called_once_with(visual)
    vision_attention_wiring.assert_called_once_with(visual)
    attention_wiring.assert_called_once_with(language_model.model)
    layernorm_wiring.assert_called_once_with(visual)


def test_moe_constructor_applies_vision_attention_not_gelu(monkeypatch) -> None:
    visual = object()
    language_model = SimpleNamespace(model=object())

    def community_init(self, *, vllm_config, prefix) -> None:
        del vllm_config, prefix
        torch.nn.Module.__init__(self)
        self.visual = visual
        self.language_model = language_model
        self.multimodal_config = SimpleNamespace(video_pruning_method=None)

    vision_wiring = Mock()
    vision_attention_wiring = Mock()
    attention_wiring = Mock()
    layernorm_wiring = Mock()
    monkeypatch.setattr(
        ilu_qwen3_5._Qwen3_5MoeForConditionalGeneration,
        "__init__",
        community_init,
    )
    monkeypatch.setattr(ilu_qwen3_5, "qwen35_vision_gelu", vision_wiring)
    monkeypatch.setattr(
        ilu_qwen3_5,
        "qwen35_vision_attention_fusion",
        vision_attention_wiring,
    )
    monkeypatch.setattr(ilu_qwen3_5, "qwen35_attention_fusion", attention_wiring)
    monkeypatch.setattr(
        ilu_qwen3_5,
        "qwen35_vision_layernorm_fusion",
        layernorm_wiring,
    )

    ilu_qwen3_5.Qwen3_5MoeForConditionalGeneration(vllm_config=object())

    vision_wiring.assert_not_called()
    vision_attention_wiring.assert_called_once_with(visual)
    attention_wiring.assert_called_once_with(language_model.model)
    layernorm_wiring.assert_called_once_with(visual)


@pytest.mark.parametrize(
    ("target", "approximate"),
    (("block", "none"), ("merger", "tanh")),
)
def test_model_wiring_rejects_semantic_drift(target, approximate) -> None:
    visual = _fake_visual()
    if target == "block":
        visual.blocks[0].mlp.act_fn = torch.nn.GELU(approximate)
    else:
        visual.merger.act_fn = torch.nn.GELU(approximate)

    with pytest.raises(RuntimeError, match="must be nn.GELU"):
        ilu_qwen3_5.qwen35_vision_gelu(visual)


@pytest.mark.parametrize("approximate", ("none", "tanh"))
def test_vision_module_passes_fixed_mode(monkeypatch, approximate) -> None:
    input = torch.randn(2, 4)
    routed = Mock(return_value=input)
    monkeypatch.setattr(ilu_qwen3_5, "_VISION_GELU", routed)

    actual = ilu_qwen3_5._VisionGELU(approximate)(input)

    assert actual is input
    routed.assert_called_once_with(input, approximate)


@requires_cuda
@pytest.mark.parametrize(
    ("approximate", "shape"),
    (("tanh", (7, 4304)), ("none", (7, 4608))),
)
def test_ixaite_routed_op_matches_torch(monkeypatch, approximate, shape) -> None:
    pytest.importorskip("ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")

    import vllm_iluvatar.backends.ixaite.runtime as kernel_provider

    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_torch_op.cache_clear()
    try:
        ilu_qwen3_5.configure_vision_gelu_provider()
        input = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        expected = F.gelu(input.float(), approximate=approximate).to(input.dtype)

        actual = ilu_qwen3_5._VISION_GELU(input, approximate)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    finally:
        kernel_provider.load_ixaite_torch_op.cache_clear()
        kernel_provider._load_ixaite_runtime.cache_clear()
