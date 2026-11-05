# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar OOT Qwen3.5 dense and MoE models.

Covers:
  * OOT registration for both Qwen3.5 architectures.
  * The community attributes required by the attention-class replacement.
  * Numerical and operator checks for the two fused kernels used by
    ``Qwen3NextAttention._project_qkv_gate``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401
from conftest import requires_cuda
from vllm.model_executor.layers.fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate as triton_fused_qk_rmsnorm_rope_gate,
)


def has_attr(cls: type, attr_name: str) -> bool:
    """Return whether ``cls.__init__`` directly assigns ``self.<attr_name>``."""
    source = textwrap.dedent(inspect.getsource(cls.__init__))
    tree = ast.parse(source)

    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attr_name
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("tp", "intermediate", "padded"),
    ((1, 4304, 4352), (2, 2152, 2176), (4, 1076, 1088)),
)
def test_vision_mlp_padding_replaces_loaded_weights_without_duplication(
    monkeypatch, tp: int, intermediate: int, padded: int
):
    from vllm_iluvatar.backends.ixaite.models import qwen3_5

    fc1 = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(intermediate, 1152)),
        bias=torch.nn.Parameter(torch.ones(intermediate)),
        tp_size=tp,
        output_size=4304,
        output_size_per_partition=intermediate,
        output_partition_sizes=[intermediate],
    )
    fc2 = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(1152, intermediate)),
        bias=torch.nn.Parameter(torch.ones(1152)),
        tp_size=tp,
        input_size=4304,
        input_size_per_partition=intermediate,
    )
    model = SimpleNamespace(
        visual=SimpleNamespace(
            blocks=[
                SimpleNamespace(mlp=SimpleNamespace(linear_fc1=fc1, linear_fc2=fc2))
            ]
        )
    )

    qwen3_5.qwen35_vision_mlp_padding(model)

    assert fc1.weight.shape == (padded, 1152)
    assert fc1.bias.shape == (padded,)
    assert fc2.weight.shape == (1152, padded)
    assert fc1.output_size_per_partition == padded
    assert fc1.output_partition_sizes == [padded]
    assert fc1.output_size == 4352
    assert fc2.input_size_per_partition == padded
    assert fc2.input_size == 4352
    assert torch.count_nonzero(fc1.weight[intermediate:]) == 0
    assert torch.count_nonzero(fc1.bias[intermediate:]) == 0
    assert torch.count_nonzero(fc2.weight[:, intermediate:]) == 0


def test_qwen3_5_oot_registration_points_at_plugin_subclasses(monkeypatch):
    from vllm import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_iluvatar.backends.ixaite.models.qwen3_5 import (
        register_model as apply_oot_registrations,
    )

    architectures = (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    )

    # The plugin keeps the official architecture names, so both must already
    # exist in the community registry.
    for architecture in architectures:
        assert architecture in _VLLM_MODELS

    # Check the minimum community import dependencies used by the OOT module.
    try:
        qwen3_5 = importlib.import_module("vllm.model_executor.models.qwen3_5")
        qwen3_next = importlib.import_module("vllm.model_executor.models.qwen3_next")
    except ImportError as exc:
        raise AssertionError(
            "Failed to import Qwen3.5 community model dependencies"
        ) from exc

    assert hasattr(qwen3_5, "Qwen3_5ForConditionalGeneration")
    assert hasattr(qwen3_5, "Qwen3_5MoeForConditionalGeneration")
    assert hasattr(qwen3_next, "Qwen3NextAttention")

    # Statically check every community attribute consumed by the optimized
    # _project_qkv_gate implementation.
    attention = qwen3_next.Qwen3NextAttention
    for attr_name in (
        "attn_output_gate",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "q_size",
        "kv_size",
        "rotary_emb",
        "q_norm",
        "k_norm",
        "use_fused_qk_norm_rope_gate",
    ):
        assert has_attr(attention, attr_name), (
            f"Qwen3NextAttention.__init__ no longer defines self.{attr_name}"
        )

    # Apply OOT registration and verify both official entries resolve to the
    # Iluvatar subclasses.
    for architecture in architectures:
        monkeypatch.setitem(
            ModelRegistry.models, architecture, ModelRegistry.models[architecture]
        )
    apply_oot_registrations()
    for architecture in architectures:
        entry = ModelRegistry.models[architecture]
        assert entry.module_name == "vllm_iluvatar.backends.ixaite.models.qwen3_5"
        assert entry.class_name == architecture


def test_qwen3_5_plugin_classes_subclass_community():
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration as _CommunityQwen3_5,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration as _CommunityQwen3_5Moe,
    )
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextAttention as _CommunityQwen3NextAttention,
    )

    from vllm_iluvatar.backends.ixaite.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3NextAttention,
    )

    assert issubclass(Qwen3_5ForConditionalGeneration, _CommunityQwen3_5)
    assert issubclass(Qwen3_5MoeForConditionalGeneration, _CommunityQwen3_5Moe)
    assert issubclass(Qwen3NextAttention, _CommunityQwen3NextAttention)


def test_qwen3_5_attention_fusion_swaps_community_attention():
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextAttention as _CommunityQwen3NextAttention,
    )

    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    attention = object.__new__(_CommunityQwen3NextAttention)
    model = type("Model", (), {})()
    model.layers = [type("Layer", (), {"self_attn": attention})()]

    ilu_qwen3_5.qwen35_attention_fusion(model)

    assert attention.__class__ is ilu_qwen3_5.Qwen3NextAttention


def test_qwen3_5_vision_attention_fusion_swaps_community_attention():
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VisionAttention as _CommunityVisionAttention,
    )

    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    attention = object.__new__(_CommunityVisionAttention)
    visual = SimpleNamespace(blocks=[SimpleNamespace(attn=attention)])

    ilu_qwen3_5.qwen35_vision_attention_fusion(visual)

    assert attention.__class__ is ilu_qwen3_5.Qwen3_5VisionAttention


def test_qwen3_5_vision_attention_uses_layout_helpers(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    seq_len = 2
    batch_size = 1
    num_heads = 2
    head_dim = 4
    padded_head_dim = 8
    qkv = torch.arange(
        seq_len * batch_size * 3 * num_heads * head_dim,
        dtype=torch.float32,
    ).reshape(seq_len, batch_size, 3 * num_heads * head_dim)
    calls = []

    def fake_qkv(x):
        calls.append(("qkv", x))
        return qkv, None

    def fake_preproc(**kwargs):
        calls.append(("preproc", kwargs))
        shape = (batch_size, seq_len, num_heads, padded_head_dim)
        return torch.ones(shape), torch.ones(shape), torch.ones(shape)

    def fake_attn(**kwargs):
        calls.append(("attn", kwargs))
        return kwargs["query"][..., :head_dim]

    def fake_output_layout(context_layer):
        calls.append(("output_layout", context_layer))
        return context_layer.permute(1, 0, 2, 3).reshape(
            seq_len, batch_size, num_heads * head_dim
        )

    def fake_proj(x):
        calls.append(("proj", x))
        return x, None

    class FakeAttentionOp:
        def __init__(self, padded_head_dim: int) -> None:
            self.padded_head_dim = padded_head_dim

        def __call__(self, **kwargs):
            return fake_attn(**kwargs)

    monkeypatch.setattr(ilu_qwen3_5, "vit_fused_qkv_preproc", fake_preproc)
    monkeypatch.setattr(ilu_qwen3_5, "vit_attn_output_layout", fake_output_layout)

    attention = object.__new__(ilu_qwen3_5.Qwen3_5VisionAttention)
    attention.qkv = fake_qkv
    attention.attn = FakeAttentionOp(padded_head_dim)
    attention.proj = fake_proj
    attention.num_attention_heads_per_partition = num_heads
    attention.hidden_size_per_attention_head = head_dim

    output = attention.forward(
        x="hidden",
        cu_seqlens="cu",
        rotary_pos_emb_cos="cos",
        rotary_pos_emb_sin="sin",
        max_seqlen="max",
        sequence_lengths="lengths",
    )

    assert calls[1][1]["qkv_packed"] is qkv
    assert calls[1][1]["padded_head_dim"] == padded_head_dim
    assert calls[2][1]["cu_seqlens"] == "cu"
    assert calls[2][1]["max_seqlen"] == "max"
    assert calls[3][0] == "output_layout"
    assert output.shape == (seq_len, batch_size, num_heads * head_dim)


def test_qwen3_5_attention_forward_calls_routed_output_gate(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    hidden_states = torch.randn(2, 16)
    positions = torch.arange(2)
    qkv = torch.randn(2, 32)
    q = torch.randn(2, 8)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    gate = torch.randn(2, 8)
    attn_output = torch.randn_like(gate)
    routed_gate = Mock(return_value=attn_output)
    monkeypatch.setattr(ilu_qwen3_5, "_ATTN_OUTPUT_SIGMOID_GATE", routed_gate)

    attention = SimpleNamespace(
        qkv_proj=Mock(return_value=(qkv, None)),
        _project_qkv_gate=Mock(return_value=(q, k, v, gate)),
        attn=Mock(return_value=attn_output),
        o_proj=Mock(return_value=(attn_output, None)),
    )

    actual = ilu_qwen3_5.Qwen3NextAttention.forward(
        attention,
        positions,
        hidden_states,
    )

    assert actual is attn_output
    args = routed_gate.call_args.args
    assert args[0] is attn_output
    assert args[1] is gate
    assert args[2] is attn_output


def test_qwen3_5_video_evs_calls_routed_retention_mask(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    class VideoInput(dict):
        timestamps = [[0.0, 0.1]]

    video = torch.arange(12 * 8, dtype=torch.bfloat16).reshape(12, 8)
    mask = torch.tensor(
        [True, True, True, True, True, False, True, False, True, False, False, True]
    )
    routed_evs = Mock(return_value=mask)
    monkeypatch.setattr(ilu_qwen3_5, "_VIDEO_EVS", routed_evs)
    create_final = Mock(side_effect=lambda **kwargs: kwargs["video_embeddings"])
    model = SimpleNamespace(
        is_multimodal_pruning_enabled=True,
        video_pruning_method="evs",
        video_pruning_rate=0.5,
        visual=SimpleNamespace(spatial_merge_size=2),
        _create_final_video_embeddings=create_final,
    )
    video_input = VideoInput(video_grid_thw=torch.tensor([[3, 4, 4]], dtype=torch.long))

    actual = ilu_qwen3_5._IluVideoEvsMixin._postprocess_video_embeds_evs(
        model, (video,), video_input
    )

    torch.testing.assert_close(actual[0], video[mask])
    routed_evs.assert_called_once_with(
        video,
        [3, 4, 4],
        spatial_merge_size=2,
        q=0.5,
    )
    assert create_final.call_args.kwargs["num_tokens_per_frame"] == [4, 2, 2]


def test_configure_video_evs_provider_binds_selected_callable(monkeypatch):
    from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5

    selected = Mock()
    loader = Mock(return_value=(selected, "test"))
    monkeypatch.setattr(ilu_qwen3_5, "load_ixaite_python_op", loader)
    monkeypatch.setattr(ilu_qwen3_5, "_VIDEO_EVS", None)

    ilu_qwen3_5.configure_video_evs_provider()

    assert ilu_qwen3_5._VIDEO_EVS is selected
    loader.assert_called_once_with("video_evs", "compute_retention_mask")


def _strided_2d(
    rows: int, columns: int, padding: int, dtype: torch.dtype
) -> torch.Tensor:
    storage = torch.randn(rows, columns + padding, dtype=dtype, device="cuda")
    result = storage[:, :columns]
    assert result.stride(0) > result.size(1)
    assert result.stride(1) == 1
    return result


def _qwen3_5_gate_reference(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    epsilon: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = q_gate.shape[0]
    q_gate_heads = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate_heads, 2, dim=-1)
    k_heads = k.view(num_tokens, num_kv_heads, head_dim)

    def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (
            x.float() * torch.rsqrt(variance + epsilon) * (weight.float() + 1.0)
        ).to(x.dtype)

    return (
        norm(q, q_weight).reshape(num_tokens, -1).contiguous(),
        norm(k_heads, k_weight).reshape(num_tokens, -1).contiguous(),
        gate.reshape(num_tokens, -1).contiguous(),
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 256])
@torch.inference_mode()
def test_qwen3_5_fused_qk_rmsnorm_gate_matches_reference(
    dtype: torch.dtype, head_dim: int
) -> None:
    torch.manual_seed(0)
    num_tokens, num_q_heads, num_kv_heads = 37, 8, 2
    epsilon = 1e-6
    q_gate = _strided_2d(
        num_tokens, num_q_heads * 2 * head_dim, padding=74, dtype=dtype
    )
    k = _strided_2d(num_tokens, num_kv_heads * head_dim, padding=42, dtype=dtype)
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")

    expected = _qwen3_5_gate_reference(
        q_gate,
        k,
        q_weight,
        k_weight,
        epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    actual = torch.ops._C.fused_qk_rmsnorm_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    for output, reference in zip(actual[:2], expected[:2]):
        assert output.is_contiguous()
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=6e-3)
    assert actual[2].is_contiguous()
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_qwen3_5_fused_qk_rmsnorm_gate_opcheck(dtype: torch.dtype) -> None:
    num_tokens, num_q_heads, num_kv_heads, head_dim = 4, 2, 1, 128
    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device="cuda")
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")

    torch.library.opcheck(
        torch.ops._C.fused_qk_rmsnorm_gate,
        (
            q_gate,
            k,
            q_weight,
            k_weight,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
        ),
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("position_dtype", [torch.int64])
@pytest.mark.parametrize("head_dim,rotary_dim", [(256, 64)])
@torch.inference_mode()
def test_qwen3_5_fused_qk_rmsnorm_rope_gate_matches_reference(
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
    position_dtype: torch.dtype,
    head_dim: int,
    rotary_dim: int,
) -> None:
    torch.manual_seed(0)
    num_tokens, num_q_heads, num_kv_heads = 8, 8, 1
    max_position = 8192
    epsilon = 1e-6
    q_gate = _strided_2d(
        num_tokens, num_q_heads * 2 * head_dim, padding=74, dtype=dtype
    )
    k = _strided_2d(num_tokens, num_kv_heads * head_dim, padding=42, dtype=dtype)
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    positions = torch.randint(
        0, max_position, (num_tokens,), dtype=position_dtype, device="cuda"
    )
    angles = torch.randn(
        max_position, rotary_dim // 2, dtype=torch.float32, device="cuda"
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(cache_dtype)

    expected = triton_fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight.float() + 1.0,
        k_weight.float() + 1.0,
        cos_sin_cache,
        positions,
        epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    actual = torch.ops._C.fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    for output, reference in zip(actual[:2], expected[:2]):
        assert output.is_contiguous()
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=5e-3)
    assert actual[2].is_contiguous()
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_qwen3_5_fused_qk_rmsnorm_rope_gate_opcheck(dtype: torch.dtype) -> None:
    num_tokens, num_q_heads, num_kv_heads, head_dim = 4, 2, 1, 128
    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device="cuda")
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    cache = torch.randn(32, head_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 32, (num_tokens,), device="cuda")

    torch.library.opcheck(
        torch.ops._C.fused_qk_rmsnorm_rope_gate,
        (
            q_gate,
            k,
            q_weight,
            k_weight,
            cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
        ),
    )


# Only the private route repairs the missing video method in the frozen baseline.
@pytest.mark.parametrize("backend", ["baseline", "ixaite"])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("method", ["evs", "vidcom2"])
def test_video_pruning_method_reaches_inherited_postprocessing(
    monkeypatch, backend, moe, method
):
    module = importlib.import_module(
        "vllm_iluvatar.backends.ixaite.models.qwen3_5"
        if backend == "ixaite"
        else "vllm_iluvatar.model_executor.models.qwen3_5"
    )
    name = (
        "Qwen3_5MoeForConditionalGeneration"
        if moe
        else "Qwen3_5ForConditionalGeneration"
    )
    parent = getattr(module, "_" + name)

    # Keep weight loading out of this check; exercise the inherited consumer directly.
    def initialize(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.multimodal_config = SimpleNamespace(video_pruning_method=method)
        self.is_multimodal_pruning_enabled = False
        self.visual = None
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=[]))

    monkeypatch.setattr(parent, "__init__", initialize)
    monkeypatch.setattr(
        parent,
        "_postprocess_video_embeds_evs",
        lambda self, *args: self.video_pruning_method,
    )
    if backend == "ixaite":
        for fusion in (
            "qwen35_vision_gelu",
            "qwen35_vision_attention_fusion",
            "qwen35_vision_layernorm_fusion",
        ):
            monkeypatch.setattr(module, fusion, lambda *args: None)
    model = getattr(module, name)(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="align")
        )
    )
    if backend == "baseline":
        with pytest.raises(AttributeError, match="video_pruning_method"):
            model._postprocess_video_embeds_evs(None, None)
    else:
        assert model._postprocess_video_embeds_evs(None, None) == method
