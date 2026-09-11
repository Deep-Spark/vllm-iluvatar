# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``ilu_mxfp4_fused_experts`` (Phase 1 torch path).

This is the compute layer the DeepSeek-V4 FP8 routed experts run on: MXFP4 ->
bf16 dequant per expert plus a grouped ``w13 -> SwiGLU(clamp) -> w2`` MLP,
weighted by ``topk_weights``.

The suite is organised around the three things that could go wrong silently:

1. **Migration equivalence** -- the function was extracted from
   ``IluDeepseekV4Fp8MoE._routed_experts`` / ``_dequant_mxfp4_to_bf16``. A
   reference implementation written the old way (contiguous
   ``[start, end)`` expert range, no ``expert_map``) is kept here and compared
   element-wise, so the refactor is pinned rather than assumed.
2. **Expert ownership** -- the new code reads ``expert_map`` instead of a
   contiguous range. Tested against a non-contiguous (round-robin style) map,
   which is exactly the case the old range-based code got wrong.
3. **Output composition** -- ``output_scale`` scales only the routed sum and
   ``extra_residual`` is added afterwards, matching
   ``ixfops.moe_output_reduce_sum(scaling_factor=..., extra_residual=...)``.
   Getting this backwards would scale the shared-expert output too.

Pure CPU / pure torch: no GPU, no ixformer.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

from vllm_iluvatar.model_executor.layers.activation import swiglu_clamp
from vllm_iluvatar.model_executor.layers.fused_moe import (
    ilu_mxfp4_fused_moe,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_mxfp4_fused_moe import (
    MXFP4_BLOCK,
    _ilu_mxfp4_fused_experts_ixformer,
    _resolve_backend,
    _swigluoai_uninterleave,
    dequant_mxfp4_to_bf16,
    ilu_mxfp4_fused_experts,
    ilu_mxfp4_fused_experts_torch,
)

# Small but structurally faithful: H and I are multiples of MXFP4_BLOCK (32) so
# the block-scale axis is exact, as it is on the real model (4096 / 2048).
_H = 64
_I = 32
_TOPK = 2
_TOKENS = 5
_SWIGLU_LIMIT = 7.0


_M3_ALPHA = 1.702
_M3_BETA = 1.0


def test_swigluoai_uninterleave_uses_packed_halves_alpha_and_beta():
    # First half is gate, second half is up. An interleaved split would pair
    # [1, 2] and [3, 4] instead and produce a visibly different result.
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    actual = _swigluoai_uninterleave(x, _SWIGLU_LIMIT, alpha=_M3_ALPHA, beta=_M3_BETA)

    gate = x[:, :2].clamp(max=_SWIGLU_LIMIT)
    up = x[:, 2:].clamp(min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
    expected = gate * torch.sigmoid(_M3_ALPHA * gate) * (up + _M3_BETA)
    interleaved = (
        x[:, 0::2] * torch.sigmoid(_M3_ALPHA * x[:, 0::2]) * (x[:, 1::2] + _M3_BETA)
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, interleaved)


def test_swigluoai_defaults_match_community_silu_and_mul_with_clamp():
    """Unset alpha/beta must mean "community fallback", not "MiniMax-M3".

    ``MoEActivation.SWIGLUOAI_UNINTERLEAVE`` is backed by
    ``SiluAndMulWithClamp`` (alpha 1.0, beta 0.0), i.e. plain
    ``silu(gate) * up`` -- not the interleaved ``SwigluOAIAndMul`` whose
    defaults are 1.702 / 1.0. Baking M3's values in as the default would apply
    M3's activation to any other checkpoint that forgot to pass them.
    """
    x = torch.tensor([[0.5, -1.5, 2.0, -0.25]], dtype=torch.float32)

    actual = _swigluoai_uninterleave(x, _SWIGLU_LIMIT)

    gate = x[:, :2].clamp(max=_SWIGLU_LIMIT)
    up = x[:, 2:].clamp(min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
    torch.testing.assert_close(actual, torch.nn.functional.silu(gate) * up)
    # And the default is genuinely not M3's, or the guard above is pointless.
    m3 = _swigluoai_uninterleave(x, _SWIGLU_LIMIT, alpha=_M3_ALPHA, beta=_M3_BETA)
    assert not torch.allclose(actual, m3)


def _rand_mxfp4(
    num_experts: int, out_dim: int, in_dim: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random MXFP4 weight + e8m0 block scale at the community parameter shapes.

    Scales are drawn from a narrow exponent window around 2**0 so the bf16 dequant
    stays well inside range and the comparison below is not dominated by rounding.
    """
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(
        0, 256, (num_experts, out_dim, in_dim // 2), dtype=torch.uint8, generator=g
    )
    scale = torch.randint(
        125,
        130,
        (num_experts, out_dim, in_dim // MXFP4_BLOCK),
        dtype=torch.uint8,
        generator=g,
    )
    return weight, scale


def _rand_routing(
    num_tokens: int, global_num_experts: int, topk: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # Distinct experts per token, like a real top-k.
    topk_ids = torch.stack(
        [
            torch.randperm(global_num_experts, generator=g)[:topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    topk_weights = torch.rand(num_tokens, topk, generator=g, dtype=torch.float32)
    return topk_weights, topk_ids


def _reference_routed_experts(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts_start_idx: int,
    swiglu_limit: float | None,
) -> torch.Tensor:
    """``IluDeepseekV4Fp8MoE._routed_experts``, verbatim in shape.

    Kept as the migration oracle: a contiguous ``[experts_start_idx, ...)`` range
    of local experts, global-id comparison, per-expert dequant, ``index_add_``
    accumulation.
    """
    routed = torch.zeros_like(x)
    for local_eid in range(w13_weight.shape[0]):
        global_eid = experts_start_idx + local_eid
        sel = topk_ids == global_eid
        if not bool(sel.any()):
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)
        xin = x[tok_idx]

        w13 = dequant_mxfp4_to_bf16(w13_weight[local_eid], w13_weight_scale[local_eid])
        gate_up = torch.matmul(xin, w13.t())
        act = swiglu_clamp(gate_up, swiglu_limit)

        w2 = dequant_mxfp4_to_bf16(w2_weight[local_eid], w2_weight_scale[local_eid])
        y = torch.matmul(act.to(w2.dtype), w2.t())

        wgt = topk_weights[tok_idx, slot_idx].unsqueeze(-1).to(y.dtype)
        routed.index_add_(0, tok_idx, (y * wgt).to(routed.dtype))
    return routed


@pytest.fixture()
def weights():
    w13, w13_scale = _rand_mxfp4(4, 2 * _I, _H, seed=1)
    w2, w2_scale = _rand_mxfp4(4, _H, _I, seed=2)
    return w13, w13_scale, w2, w2_scale


@pytest.fixture()
def hidden_states():
    g = torch.Generator().manual_seed(3)
    return torch.randn(_TOKENS, _H, generator=g, dtype=torch.float32).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# dequant
# ---------------------------------------------------------------------------


def test_dequant_decodes_nibble_order_sign_and_block_scale():
    """One byte, hand-computed.

    MXFP4 packs the low nibble first (community ``break_fp4_bytes`` order) and
    the e8m0 scale is ``2 ** (v - 127)``. Byte ``0x9A`` therefore decodes to
    ``[+2.0, -1.0] * 2 ** (128 - 127)``:
    low  nibble ``0xA`` = sign 1, magnitude 2 -> -1.0
    high nibble ``0x9`` = sign 1, magnitude 1 -> -0.5
    """
    weight = torch.full((1, MXFP4_BLOCK // 2), 0x9A, dtype=torch.uint8)
    scale = torch.full((1, 1), 128, dtype=torch.uint8)  # 2 ** 1

    out = dequant_mxfp4_to_bf16(weight, scale)

    assert out.shape == (1, MXFP4_BLOCK)
    assert out.dtype is torch.bfloat16
    expected = torch.tensor([-1.0 * 2, -0.5 * 2], dtype=torch.bfloat16).repeat(
        MXFP4_BLOCK // 2
    )
    torch.testing.assert_close(out[0], expected)


def test_dequant_zero_code_is_zero():
    weight = torch.zeros((2, MXFP4_BLOCK // 2), dtype=torch.uint8)
    scale = torch.full((2, 1), 200, dtype=torch.uint8)  # any exponent
    out = dequant_mxfp4_to_bf16(weight, scale)
    assert torch.count_nonzero(out) == 0


# ---------------------------------------------------------------------------
# migration equivalence
# ---------------------------------------------------------------------------


def test_matches_module_4_reference_without_ep(weights, hidden_states):
    """No ``expert_map``: every expert is local, ids pass through unchanged."""
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=4)

    got = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    want = _reference_routed_experts(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        experts_start_idx=0,
        swiglu_limit=_SWIGLU_LIMIT,
    )

    assert got.shape == hidden_states.shape
    assert got.dtype is hidden_states.dtype
    torch.testing.assert_close(got, want)


def test_matches_module_4_reference_with_contiguous_ep_shard(weights, hidden_states):
    """Contiguous EP shard: local experts are a contiguous global range.

    Rank 1 of 2 owns globals ``[2, 4)``. ``expert_map`` expresses the same thing,
    so the two implementations must agree, and tokens routed elsewhere must
    contribute nothing (the EP partial is completed by the runner's all-reduce).
    """
    w13, w13_scale, w2, w2_scale = weights
    global_num_experts = 8
    topk_weights, topk_ids = _rand_routing(_TOKENS, global_num_experts, _TOPK, seed=5)

    start = 2
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    expert_map[start : start + 4] = torch.arange(4, dtype=torch.int32)

    got = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=expert_map,
    )
    want = _reference_routed_experts(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        experts_start_idx=start,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    torch.testing.assert_close(got, want)


def test_non_contiguous_expert_map_is_honoured(weights, hidden_states):
    """Non-contiguous map -- strictly better than the old range arithmetic.

    A round-robin placement (rank 0 of 2 owning globals 0, 2, 4, 6) has no
    contiguous ``[start, end)`` description at all. Building the equivalent
    contiguous problem by permuting the *weights* must give the same answer.
    """
    w13, w13_scale, w2, w2_scale = weights
    global_num_experts = 8
    topk_weights, topk_ids = _rand_routing(_TOKENS, global_num_experts, _TOPK, seed=6)

    owned = [0, 2, 4, 6]
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    for local, global_eid in enumerate(owned):
        expert_map[global_eid] = local

    got = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=expert_map,
    )

    # Same computation expressed without a map: relabel the routing so that
    # global id ``owned[i]`` becomes ``i`` and mark the rest as remote.
    relabelled = torch.full_like(topk_ids, -1)
    for local, global_eid in enumerate(owned):
        relabelled[topk_ids == global_eid] = local
    want = _reference_routed_experts(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        relabelled,
        topk_weights,
        experts_start_idx=0,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    torch.testing.assert_close(got, want)


def test_no_local_expert_selected_returns_zeros(weights, hidden_states):
    w13, w13_scale, w2, w2_scale = weights
    global_num_experts = 8
    topk_weights, topk_ids = _rand_routing(_TOKENS, global_num_experts, _TOPK, seed=7)
    # Own nothing at all.
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)

    got = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=expert_map,
    )
    assert torch.count_nonzero(got) == 0
    assert got.shape == hidden_states.shape


# ---------------------------------------------------------------------------
# output composition
# ---------------------------------------------------------------------------


def test_output_scale_applies_to_the_routed_sum_only(weights, hidden_states):
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=8)
    kwargs = dict(
        swiglu_limit=_SWIGLU_LIMIT,
    )

    base = ilu_mxfp4_fused_experts_torch(
        hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, **kwargs
    )
    scaled = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output_scale=2.0,
        **kwargs,
    )
    torch.testing.assert_close(scaled, base * 2.0)


def test_extra_residual_is_added_after_scaling_and_not_mutated(weights, hidden_states):
    """``out = output_scale * routed + extra_residual``.

    Matches ``moe_output_reduce_sum(scaling_factor=..., extra_residual=...)``: the
    shared-expert output must **not** pick up the routed scale. Also asserts the
    caller's tensor survives, because ``SharedExperts`` owns it.
    """
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=9)

    g = torch.Generator().manual_seed(10)
    shared = torch.randn(_TOKENS, _H, generator=g, dtype=torch.float32).to(
        torch.bfloat16
    )
    shared_before = shared.clone()

    routed_only = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    combined = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        output_scale=2.0,
        extra_residual=shared,
    )

    torch.testing.assert_close(combined, routed_only * 2.0 + shared)
    torch.testing.assert_close(shared, shared_before)


def test_swiglu_limit_none_disables_clamping(weights, hidden_states):
    """``swiglu_limit=None`` must mean "no clamp", not "clamp at 0"."""
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=11)

    unclamped = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=None,
    )
    tight = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=1e-4,
    )
    assert torch.count_nonzero(unclamped) > 0
    assert not torch.allclose(unclamped, tight)


def test_w2_bias_is_applied_when_present(hidden_states):
    """DeepSeek-V4 has no MoE bias; GPT-OSS-style MXFP4 does. The parameters are
    accepted so this compute layer can be reused, so make sure they are wired.

    All-zero weights make the assertion exact rather than a bf16-rounding
    comparison: every expert output is then just ``w2_bias``, so each token
    receives ``sum_over_slots(topk_weight * 1)`` on every channel.
    """
    w13 = torch.zeros(4, 2 * _I, _H // 2, dtype=torch.uint8)
    w13_scale = torch.full((4, 2 * _I, _H // MXFP4_BLOCK), 127, dtype=torch.uint8)
    w2 = torch.zeros(4, _H, _I // 2, dtype=torch.uint8)
    w2_scale = torch.full((4, _H, _I // MXFP4_BLOCK), 127, dtype=torch.uint8)
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=12)

    out = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        w2_bias=torch.ones(4, _H, dtype=torch.bfloat16),
    )

    expected = (
        topk_weights.sum(dim=-1, keepdim=True).to(torch.bfloat16).expand(_TOKENS, _H)
    )
    torch.testing.assert_close(out, expected.contiguous())


def test_w13_bias_changes_the_result(weights, hidden_states):
    """``w13_bias`` lands before the SwiGLU, so an exact expectation would have to
    re-derive the whole MLP. Asserting that it is consumed at all is enough here;
    ``test_w2_bias_is_applied_when_present`` pins the arithmetic.
    """
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=13)
    common = (w13, w13_scale, w2, w2_scale, topk_weights, topk_ids)

    without = ilu_mxfp4_fused_experts_torch(
        hidden_states, *common, swiglu_limit=_SWIGLU_LIMIT
    )
    with_bias = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        *common,
        swiglu_limit=_SWIGLU_LIMIT,
        w13_bias=torch.full((4, 2 * _I), 4.0, dtype=torch.bfloat16),
    )
    assert not torch.allclose(without, with_bias)


# ---------------------------------------------------------------------------
# dispatcher + dtype normalisation + ixformer pipeline wiring
# ---------------------------------------------------------------------------


class _FakeHidden:
    """Minimal stand-in for ``hidden_states`` in ``_resolve_backend`` unit tests.

    Only the three attributes the auto path reads (``is_cuda``, ``shape``,
    ``dtype``) are populated, so the fallback matrix can be exercised without a
    real CUDA tensor. ``w13_weight`` is covered separately via the ``w13_weight``
    kwarg (only ``.dtype`` is read).
    """

    def __init__(self, *, is_cuda=True, num_tokens=5, dtype=torch.bfloat16):
        self.is_cuda = is_cuda
        self.shape = (num_tokens, _H)
        self.dtype = dtype


def _resolve_auto(
    hidden_states,
    *,
    swiglu_limit=_SWIGLU_LIMIT,
    w13_bias=None,
    w2_bias=None,
    w13_weight_dtype=torch.int8,
):
    """Call ``_resolve_backend`` on the explicit ``auto`` path with the
    ``is_ivcore30`` gate forced True, so the only variables are the fallback
    conditions.

    ``_resolve_backend`` imports ``is_ivcore30`` lazily from
    ``vllm_iluvatar.ilu_util.arch`` at call time, so patching that module's
    attribute is sufficient.
    """
    import vllm_iluvatar.ilu_util.arch as arch

    saved = arch.is_ivcore30
    arch.is_ivcore30 = lambda: True
    try:
        w13 = type("_W", (), {"dtype": w13_weight_dtype})()
        w2 = type("_W", (), {"dtype": w13_weight_dtype})()
        scale_dtype = (
            torch.float32 if w13_weight_dtype is torch.uint8 else torch.float8_e8m0fnu
        )
        s13 = type("_S", (), {"dtype": scale_dtype})()
        s2 = type("_S", (), {"dtype": scale_dtype})()
        return _resolve_backend(
            "auto",
            hidden_states,
            swiglu_limit=swiglu_limit,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_weight=w13,
            w13_weight_scale=s13,
            w2_weight=w2,
            w2_weight_scale=s2,
        )
    finally:
        arch.is_ivcore30 = saved


def test_resolve_backend_explicit_torch_bypasses_ixformer_conditions():
    hs = _FakeHidden(is_cuda=False, num_tokens=0, dtype=torch.float16)
    assert (
        _resolve_backend(
            "torch",
            hs,
            swiglu_limit=None,
            w13_bias=None,
            w2_bias=None,
            w13_weight=type("_W", (), {"dtype": torch.uint8})(),
            w13_weight_scale=type("_S", (), {"dtype": torch.uint8})(),
            w2_weight=type("_W", (), {"dtype": torch.uint8})(),
            w2_weight_scale=type("_S", (), {"dtype": torch.uint8})(),
        )
        == "torch"
    )


def test_resolve_backend_unknown_name_is_rejected():
    hs = _FakeHidden()
    with pytest.raises(ValueError, match="Unknown MXFP4 MoE backend"):
        _resolve_backend(
            "nonsense",
            hs,
            swiglu_limit=_SWIGLU_LIMIT,
            w13_bias=None,
            w2_bias=None,
            w13_weight=type("_W", (), {"dtype": torch.int8})(),
            w13_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
            w2_weight=type("_W", (), {"dtype": torch.int8})(),
            w2_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
        )


def test_resolve_backend_auto_cpu_falls_back_to_torch(monkeypatch):
    """``is_cuda=False`` is the first hard condition (§3.4 row 2). Even with
    ``is_ivcore30`` forced True the answer must be torch."""
    hs = _FakeHidden(is_cuda=False)
    assert _resolve_auto(hs) == "torch"


def test_resolve_backend_auto_non_ivcore30_falls_back_to_torch(monkeypatch):
    """Condition 2: ``is_ivcore30()`` False -> torch, even on a CUDA tensor."""
    import vllm_iluvatar.ilu_util.arch as arch

    monkeypatch.setattr(arch, "is_ivcore30", lambda: False)
    hs = _FakeHidden(is_cuda=True)
    assert (
        _resolve_backend(
            None,
            hs,
            swiglu_limit=_SWIGLU_LIMIT,
            w13_bias=None,
            w2_bias=None,
            w13_weight=type("_W", (), {"dtype": torch.int8})(),
            w13_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
            w2_weight=type("_W", (), {"dtype": torch.int8})(),
            w2_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
        )
        == "torch"
    )


def test_resolve_backend_auto_zero_tokens_falls_back_to_torch():
    hs = _FakeHidden(is_cuda=True, num_tokens=0)
    assert _resolve_auto(hs) == "torch"


def test_resolve_backend_auto_non_bf16_falls_back_to_torch():
    hs = _FakeHidden(is_cuda=True, dtype=torch.float16)
    assert _resolve_auto(hs) == "torch"


def test_resolve_backend_auto_no_swiglu_limit_falls_back_to_torch():
    hs = _FakeHidden(is_cuda=True)
    assert _resolve_auto(hs, swiglu_limit=None) == "torch"


def test_resolve_backend_auto_with_bias_falls_back_to_torch():
    hs = _FakeHidden(is_cuda=True)
    bias = torch.zeros(4, 2 * _I, dtype=torch.bfloat16)
    assert _resolve_auto(hs, w13_bias=bias) == "torch"
    assert _resolve_auto(hs, w2_bias=bias) == "torch"


def test_resolve_backend_auto_ixmm_format_selects_ixmm(monkeypatch):
    hs = _FakeHidden(is_cuda=True)
    monkeypatch.setattr(ilu_mxfp4_fused_moe, "_ixmm_backend_available", lambda: True)
    assert _resolve_auto(hs, w13_weight_dtype=torch.uint8) == "ixmm"


def test_resolve_backend_explicit_ixmm_rejects_original_layout(monkeypatch):
    import vllm_iluvatar.ilu_util.arch as arch

    monkeypatch.setattr(arch, "is_ivcore30", lambda: True)
    hs = _FakeHidden(is_cuda=True)
    with pytest.raises(RuntimeError, match="preprocess_weight_for_ixmm"):
        _resolve_backend(
            "ixmm",
            hs,
            swiglu_limit=_SWIGLU_LIMIT,
            w13_bias=None,
            w2_bias=None,
            w13_weight=type("_W", (), {"dtype": torch.int8})(),
            w13_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
            w2_weight=type("_W", (), {"dtype": torch.int8})(),
            w2_weight_scale=type("_S", (), {"dtype": torch.float8_e8m0fnu})(),
        )


def test_dispatcher_on_cpu_routes_to_torch_and_does_not_import_ixformer(
    weights, hidden_states, monkeypatch
):
    """End-to-end dispatcher behaviour on CPU: must produce the torch result
    and must not pull ixformer into ``sys.modules`` (§6 CPU suite)."""
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=14)

    before = {k for k in sys.modules if k.startswith("ixformer")}
    got = ilu_mxfp4_fused_experts(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    after = {k for k in sys.modules if k.startswith("ixformer")}
    assert after == before, f"dispatcher imported ixformer: {after - before}"

    want = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    torch.testing.assert_close(got, want)


def test_dequant_is_dtype_agnostic_after_normalisation():
    """Covers §2.1 / §2.2: the int8 + e8m0 views produced by
    ``process_weights_after_loading`` must dequantise identically to the raw
    uint8 checkpoint bytes."""
    w_uint8, s_uint8 = _rand_mxfp4(2, 2 * _I, _H, seed=21)
    w_int8 = w_uint8.data.view(torch.int8)
    s_e8m0 = s_uint8.data.view(torch.float8_e8m0fnu)

    out_uint8 = dequant_mxfp4_to_bf16(w_uint8[0], s_uint8[0])
    out_normalised = dequant_mxfp4_to_bf16(w_int8[0], s_e8m0[0])
    torch.testing.assert_close(out_normalised, out_uint8)


def test_torch_impl_is_invariant_under_dtype_normalisation(weights, hidden_states):
    """The torch oracle must give the same answer whether the weights are raw
    uint8 (pre-loading) or the int8/e8m0 views (post-loading). This is the
    ``process_weights_after_loading`` zero-copy claim end-to-end."""
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=22)

    raw = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    w13_n = w13.data.view(torch.int8)
    s13_n = w13_scale.data.view(torch.float8_e8m0fnu)
    w2_n = w2.data.view(torch.int8)
    s2_n = w2_scale.data.view(torch.float8_e8m0fnu)
    normalised = ilu_mxfp4_fused_experts_torch(
        hidden_states,
        w13_n,
        s13_n,
        w2_n,
        s2_n,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    torch.testing.assert_close(normalised, raw)


def _install_fake_ixfops(monkeypatch):
    """Inject a MagicMock ``ixformer.inference.functions`` into ``sys.modules`` so
    ``_ilu_mxfp4_fused_experts_ixformer``'s lazy ``import`` binds to the mock
    without requiring a real ixformer install or a CUDA device.

    A MagicMock responds to any attribute access, so the production code's
    ``ixfops.<op>(...)`` calls land on auto-created child mocks whose
    ``call_args`` we assert on. The parent ``ixformer`` / ``ixformer.inference``
    modules are also mocked so the dotted import resolves without touching the
    real package.
    """
    fake_ixfops = MagicMock()
    _routing_out = {}

    def _routing(topk_ids, start, end, expert_map=None):
        M = topk_ids.numel()
        src_to_dst = torch.arange(M, dtype=torch.int32)
        dst_to_src = torch.arange(M, dtype=torch.int32)
        group_layout = (
            torch.tensor([[0, M]], dtype=torch.int32)
            .expand(end - start, 2)
            .contiguous()
        )
        result = (src_to_dst, dst_to_src, group_layout)
        _routing_out["v"] = result
        return result

    fake_ixfops.moe_compute_token_index_ep_group_layout.side_effect = _routing
    fake_ixfops.moe_expand_input_dynamic_scaled_fp8.return_value = (
        torch.empty(_TOKENS * _TOPK, _H, dtype=torch.float8_e4m3fn),
        torch.ones(_TOKENS * _TOPK, dtype=torch.float32),
    )
    fake_ixfops.swiglu_limit_dynamic_scaled_fp8.return_value = (
        torch.empty(_TOKENS * _TOPK, _I, dtype=torch.float8_e4m3fn),
        torch.ones(_TOKENS * _TOPK, dtype=torch.float32),
    )

    fake_ixformer = MagicMock()
    fake_inference = MagicMock()
    fake_inference.functions = fake_ixfops
    fake_ixformer.inference = fake_inference
    monkeypatch.setitem(sys.modules, "ixformer", fake_ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", fake_ixfops)
    return fake_ixfops, _routing_out


def test_ixformer_pipeline_wires_args_correctly(weights, hidden_states, monkeypatch):
    """Mock-based argument assertion for the six-step pipeline (§6 CPU suite:
    'CPU 侧改为对传给每个 op 的实参做 MagicMock 断言'). Pins
    ``fuse_swiglu=False`` / ``row_mapping=None`` / ``scaling_factor=output_scale``
    / ``extra_residual`` pass-through and the ``start=0, end=E_local`` routing
    interval."""
    w13, w13_scale, w2, w2_scale = weights
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=23)
    shared = torch.randn(
        _TOKENS, _H, generator=torch.Generator().manual_seed(24), dtype=torch.float32
    ).to(torch.bfloat16)

    # Normalise weights to the post-loading dtypes the pipeline expects.
    w13_n = w13.data.view(torch.int8)
    s13_n = w13_scale.data.view(torch.float8_e8m0fnu)
    w2_n = w2.data.view(torch.int8)
    s2_n = w2_scale.data.view(torch.float8_e8m0fnu)

    fake_ixfops, routing_out = _install_fake_ixfops(monkeypatch)
    # Make group_gemm_fp4 and gather return plausible-shaped tensors so the
    # function body runs to completion.
    fake_ixfops.group_gemm_fp4.side_effect = [
        torch.empty(_TOKENS * _TOPK, 2 * _I, dtype=torch.bfloat16),
        torch.empty(_TOKENS * _TOPK, _H, dtype=torch.bfloat16),
    ]
    fake_ixfops.moe_gather_output_reduce_sum.return_value = torch.zeros(
        _TOKENS, _H, dtype=torch.bfloat16
    )

    _ilu_mxfp4_fused_experts_ixformer(
        hidden_states,
        w13_n,
        s13_n,
        w2_n,
        s2_n,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=None,
        output_scale=2.0,
        extra_residual=shared,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixformer_native",
        gemm_mode=0,
        swiglu_variant="dsv4_clamp",
    )

    src_to_dst, dst_to_src, group_layout = routing_out["v"]

    # 1) routing: topk_ids forwarded untouched, start=0, end=E_local, and the
    # expert_map remap delegated to the op.
    routing_call = fake_ixfops.moe_compute_token_index_ep_group_layout
    routing_call.assert_called_once()
    r_args = routing_call.call_args.args
    assert r_args[0] is topk_ids
    assert r_args[1] == 0 and r_args[2] == w13_n.shape[0]
    assert routing_call.call_args.kwargs["expert_map"] is None

    # 2) expand: dst_to_src + top_k forwarded.
    expand_call = fake_ixfops.moe_expand_input_dynamic_scaled_fp8
    expand_call.assert_called_once()
    assert expand_call.call_args.args[1] is dst_to_src
    assert expand_call.call_args.args[2] == _TOPK

    # 3) + 5) group_gemm_fp4: two calls, both fuse_swiglu=False, row_mapping=None.
    assert fake_ixfops.group_gemm_fp4.call_count == 2
    for call in fake_ixfops.group_gemm_fp4.call_args_list:
        assert call.kwargs["fuse_swiglu"] is False
        assert call.kwargs["row_mapping"] is None
        assert call.kwargs["mode"] == 0
        assert call.kwargs["backend"] == "ixformer_native"
    # GEMM1 gets w13, GEMM2 gets w2.
    assert fake_ixfops.group_gemm_fp4.call_args_list[0].args[2] is w13_n
    assert fake_ixfops.group_gemm_fp4.call_args_list[1].args[2] is w2_n

    # 4) swiglu: limit forwarded as float.
    swiglu_call = fake_ixfops.swiglu_limit_dynamic_scaled_fp8
    swiglu_call.assert_called_once()
    assert swiglu_call.call_args.args[2] == float(_SWIGLU_LIMIT)

    # 6) gather: scaling_factor=output_scale, extra_residual=shared.
    gather_call = fake_ixfops.moe_gather_output_reduce_sum
    gather_call.assert_called_once()
    assert gather_call.call_args.kwargs["scaling_factor"] == 2.0
    assert gather_call.call_args.kwargs["extra_residual"] is shared
    assert gather_call.call_args.args[1].dtype is torch.float32


@pytest.mark.parametrize("alpha,beta", [(1.702, 1.0), (0.75, -0.25)])
def test_ixmm_uses_parameterized_fused_packed_swiglu_quant(
    hidden_states, monkeypatch, alpha, beta
):
    expert_count = 4
    w13 = torch.zeros(
        expert_count, _H // 2, 2 * _I, dtype=torch.uint8
    )
    w13_scale = torch.ones(
        expert_count, 1, 2 * _I, dtype=torch.float32
    )
    w2 = torch.zeros(
        expert_count, 256 // 2, _H, dtype=torch.uint8
    )
    w2_scale = torch.ones(expert_count, 1, _H, dtype=torch.float32)
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4, _TOPK, seed=29)
    fake_ixfops, routing_out = _install_fake_ixfops(monkeypatch)
    fake_ixfops.swigluoai_uninterleave_dynamic_scaled_fp8.return_value = (
        torch.empty(_TOKENS * _TOPK, _I, dtype=torch.float8_e4m3fn),
        torch.ones(_TOKENS * _TOPK, dtype=torch.float32),
    )
    fake_ixfops.moe_gather_output_reduce_sum.return_value = torch.zeros(
        _TOKENS, _H, dtype=torch.bfloat16
    )

    _ilu_mxfp4_fused_experts_ixformer(
        hidden_states,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=None,
        output_scale=1.0,
        extra_residual=None,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixmm",
        gemm_mode=0,
        swiglu_variant="swigluoai",
        swiglu_alpha=alpha,
        swiglu_beta=beta,
    )

    fused = fake_ixfops.swigluoai_uninterleave_dynamic_scaled_fp8
    fused.assert_called_once()
    assert fused.call_args.args[1] is routing_out["v"][1]
    assert fused.call_args.args[2] == float(_SWIGLU_LIMIT)
    assert fused.call_args.args[3] == alpha
    assert fused.call_args.args[4] == beta
    assert fused.call_args.kwargs["backend"] == "ixformer_native"
    fake_ixfops.fp8_per_token_dynamic_scaled.assert_not_called()


def test_ixmm_pipeline_pads_only_w2_k(monkeypatch):
    """w13 keeps its K while w2 and the SwiGLU activation pad 640 to 768."""
    expert_count = 4
    hidden_size = 256
    logical_inter_size = 640
    padded_inter_size = 768
    hidden_states = torch.zeros(_TOKENS, hidden_size, dtype=torch.bfloat16)
    w13 = torch.zeros(
        expert_count, hidden_size // 2, 2 * logical_inter_size, dtype=torch.uint8
    )
    s13 = torch.ones(
        expert_count, hidden_size // 256, 2 * logical_inter_size, dtype=torch.float32
    )
    w2 = torch.zeros(
        expert_count, padded_inter_size // 2, hidden_size, dtype=torch.uint8
    )
    s2 = torch.ones(
        expert_count,
        padded_inter_size // 256,
        hidden_size,
        dtype=torch.float32,
    )
    topk_weights, topk_ids = _rand_routing(_TOKENS, expert_count, _TOPK, seed=25)

    fake_ixfops, _ = _install_fake_ixfops(monkeypatch)
    fake_ixfops.moe_expand_input_dynamic_scaled_fp8.return_value = (
        torch.empty(_TOKENS * _TOPK, hidden_size, dtype=torch.float8_e4m3fn),
        torch.ones(_TOKENS * _TOPK, dtype=torch.float32),
    )
    fake_ixfops.swiglu_limit_dynamic_scaled_fp8.return_value = (
        torch.empty(_TOKENS * _TOPK, logical_inter_size, dtype=torch.float8_e4m3fn),
        torch.ones(_TOKENS * _TOPK, dtype=torch.float32),
    )
    fake_ixfops.moe_gather_output_reduce_sum.return_value = torch.zeros(
        _TOKENS, hidden_size, dtype=torch.bfloat16
    )

    _ilu_mxfp4_fused_experts_ixformer(
        hidden_states,
        w13,
        s13,
        w2,
        s2,
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=None,
        output_scale=1.0,
        extra_residual=None,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixmm",
        gemm_mode=0,
        swiglu_variant="dsv4_clamp",
    )

    expand_input = fake_ixfops.moe_expand_input_dynamic_scaled_fp8.call_args.args[0]
    assert expand_input is hidden_states

    gemm_calls = fake_ixfops.group_gemm_fp4.call_args_list
    assert w13.shape[1] * 2 == hidden_size
    assert gemm_calls[0].args[0].shape[-1] == hidden_size
    assert gemm_calls[0].args[5].shape == (
        _TOKENS * _TOPK,
        2 * logical_inter_size,
    )
    assert gemm_calls[1].args[0].shape[-1] == padded_inter_size
    assert (
        torch.count_nonzero(gemm_calls[1].args[0][:, logical_inter_size:].float()) == 0
    )
    assert gemm_calls[1].args[5].shape == (_TOKENS * _TOPK, hidden_size)


def test_ixformer_pipeline_forwards_expert_map_without_remapping(
    weights, hidden_states, monkeypatch
):
    """``expert_map`` goes to the routing op verbatim.

    The global -> local remap used to happen here as
    ``expert_map[topk_ids.long()].to(torch.int32)``; the op owns it now. Pinning
    ``r_args[0] is topk_ids`` is what keeps that Python gather from creeping
    back: an equality assertion would still pass on a remapped copy.
    """
    w13, w13_scale, w2, w2_scale = weights
    expert_count = w13.shape[0]
    topk_weights, topk_ids = _rand_routing(_TOKENS, 4 * expert_count, _TOPK, seed=31)
    # Community layout: local index for owned experts, -1 for remote ones. Here
    # this rank owns every 4th global expert (round-robin placement), which no
    # contiguous [start, end) interval could describe.
    expert_map = torch.full((4 * expert_count,), -1, dtype=torch.int32)
    expert_map[::4] = torch.arange(expert_count, dtype=torch.int32)

    fake_ixfops, _ = _install_fake_ixfops(monkeypatch)
    fake_ixfops.moe_gather_output_reduce_sum.return_value = torch.zeros(
        _TOKENS, _H, dtype=torch.bfloat16
    )

    _ilu_mxfp4_fused_experts_ixformer(
        hidden_states,
        w13.data.view(torch.int8),
        w13_scale.data.view(torch.float8_e8m0fnu),
        w2.data.view(torch.int8),
        w2_scale.data.view(torch.float8_e8m0fnu),
        topk_weights,
        topk_ids,
        swiglu_limit=_SWIGLU_LIMIT,
        expert_map=expert_map,
        output_scale=1.0,
        extra_residual=None,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixformer_native",
        gemm_mode=0,
        swiglu_variant="dsv4_clamp",
    )

    routing_call = fake_ixfops.moe_compute_token_index_ep_group_layout
    routing_call.assert_called_once()
    assert routing_call.call_args.args[0] is topk_ids
    assert routing_call.call_args.args[1] == 0
    assert routing_call.call_args.args[2] == expert_count
    assert routing_call.call_args.kwargs["expert_map"] is expert_map
