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


def test_qwen3_5_oot_registration_points_at_plugin_subclasses():
    from vllm import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_iluvatar.ilu_register import apply_oot_registrations

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
    apply_oot_registrations()
    for architecture in architectures:
        entry = ModelRegistry.models[architecture]
        assert entry.module_name == "vllm_iluvatar.model_executor.models.qwen3_5"
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

    from vllm_iluvatar.model_executor.models.qwen3_5 import (
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

    from vllm_iluvatar.model_executor.models import qwen3_5 as ilu_qwen3_5

    attention = object.__new__(_CommunityQwen3NextAttention)
    model = type("Model", (), {})()
    model.layers = [type("Layer", (), {"self_attn": attention})()]

    ilu_qwen3_5.qwen35_attention_fusion(model)

    assert attention.__class__ is ilu_qwen3_5.Qwen3NextAttention


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
