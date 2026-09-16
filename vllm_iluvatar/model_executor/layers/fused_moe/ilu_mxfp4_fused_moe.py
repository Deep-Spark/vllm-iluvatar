# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP4 fused-MoE routed-expert execution.

Corresponds to:
    vllm/model_executor/layers/quantization/mxfp4.py (the routed-expert compute
    the community ``Mxfp4MoeBackend`` kernels own)

Target   : routed MXFP4 expert execution for Iluvatar
Baseline : main_9c12a81ac
Why      : none of the community MXFP4 MoE backends (TRTLLM / DeepGEMM / Marlin /
           FlashInfer / AITER) has an Iluvatar implementation, so the plugin owns
           the compute. Keeping it in a standalone function -- rather than inside
           the quant method -- mirrors ``ilu_fused_moe.py::ilu_w16a16_fused_experts``
           and lets several MXFP4 methods share one implementation.
How      : callers (currently ``IluMxfp4MoEMethod.apply``) call
           ``ilu_mxfp4_fused_experts`` with explicit tensors. The signature takes
           no ``layer`` and no ``RoutedExperts`` attribute names, so a future
           GPT-OSS-MXFP4 or compressed-tensors-W4A4-MXFP4 method only has to
           write its own wiring.
Scope    : routed MXFP4 experts on Iluvatar.
Remove   : never as a whole -- it is the only implementation of this compute.

Two implementations, one dispatcher
-----------------------------------
``ilu_mxfp4_fused_experts`` is the public entry point. It resolves a backend
(``_resolve_backend``) and delegates to one of:

* ``_ilu_mxfp4_fused_experts_ixformer`` -- the production path. Six device
  kernels, no bf16 weight materialisation::

      moe_compute_token_index_ep_group_layout
        -> moe_expand_input_dynamic_scaled_fp8
        -> group_gemm_fp4(w13) -> swiglu_limit_dynamic_scaled_fp8
        -> group_gemm_fp4(w2)  -> moe_gather_output_reduce_sum(extra_residual=...)

  The ``group_gemm_fp4`` backend is ``ixmm`` by default
  (``VLLM_ILU_MXFP4_MOE_BACKEND``), which consumes the ix30 layout that
  ``preprocess_weight_for_ixmm`` produces once at load time. ``ixformer_native``
  consumes the original MXFP4 layout instead and must be requested explicitly:
  its kernel is GEMV-shaped and unusable at prefill scale.

* ``ilu_mxfp4_fused_experts_torch`` -- pure torch, per-expert MXFP4 -> bf16
  dequant plus a grouped MLP. Not a dead migration artefact; see its docstring
  for the three roles it still serves.

Three invariants both implementations share:

* expert ownership comes from ``expert_map`` (the community global -> local map),
  not from a model's own ``experts_start_idx`` / ``n_local_experts``, which are
  not authoritative once ``FusedMoEParallelConfig`` decides the split. The
  ixformer path forwards ``expert_map`` straight into the routing op, which does
  the remap on device;
* the shared-expert output arrives as ``extra_residual`` and must be folded in
  here, because on the fused-shared path the runner returns ``shared_output ==
  None`` and nobody adds it later;
* the output scale arrives as ``output_scale`` (``layer._ilu_output_scale``,
  which is ``1.0``) -- never ``layer.routed_scaling_factor``, which is the
  router-aligned ``1.5`` and would double-scale the result.

ixformer is deliberately **not** imported at module scope: the torch path must
stay importable (and testable) on a plain CPU host. The ixformer import lives
inside ``_ilu_mxfp4_fused_experts_ixformer``, behind the dispatcher's CUDA/arch
gate.
"""

from __future__ import annotations

import functools

import torch
from vllm.logger import init_logger

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    _e8m0_to_fp32,
)
from vllm_iluvatar.model_executor.layers.activation import swiglu_clamp

logger = init_logger(__name__)

# MXFP4 microscaling block size: 32 fp4 (e2m1) values share one e8m0 exponent.
MXFP4_BLOCK = 32
_IXMM_GROUP_SIZE = 256

# FP4 E2M1 magnitude lookup (3-bit magnitude -> fp32); the sign bit is applied
# separately. Same table community ``nvfp4_emulation_utils.break_fp4_bytes`` uses.
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _zero_pad_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    """Right-pad the last dimension with zeros, preserving dtype and device."""
    if x.shape[-1] == size:
        return x
    assert x.shape[-1] < size, f"cannot pad dimension {x.shape[-1]} to {size}"
    padded = x.new_zeros((*x.shape[:-1], size))
    padded[..., : x.shape[-1]] = x
    return padded


def dequant_mxfp4_to_bf16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = MXFP4_BLOCK,
) -> torch.Tensor:
    """Pure-torch MXFP4 -> bf16 dequant for one expert's weight matrix.

    ``weight`` ``[out, in // 2]`` uint8 packs two fp4 (e2m1) nibbles per byte,
    low nibble first (community ``break_fp4_bytes`` order). ``scale``
    ``[out, in // block]`` is an e8m0 (exponent-only) block scale. Returns
    ``[out, in]`` bf16.

    Reuses ``_e8m0_to_fp32`` for the exponent decode. Unlike NVFP4
    there is no global scale.

    Accepts ``weight`` as either ``uint8`` (raw checkpoint) or ``int8``
    (post-``process_weights_after_loading`` view), and ``scale`` as either
    ``uint8`` or ``float8_e8m0fnu``.  Both are
    normalised to ``uint8`` here so the nibble arithmetic below is independent
    of the dtype label the caller carries.
    """
    if weight.dtype is not torch.uint8:
        weight = weight.view(torch.uint8)
    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    # element 2i = low nibble of byte i, element 2i+1 = high nibble
    codes = torch.stack((low, high), dim=-1).flatten(start_dim=-2)  # [out, in]
    sign = torch.where((codes & 0x08) != 0, -1.0, 1.0)
    mag = (codes & 0x07).long()
    lut = _FP4_E2M1_LUT.to(weight.device)
    vals = lut[mag] * sign  # fp32 [out, in]

    in_dim = codes.shape[-1]
    scale_f32 = _e8m0_to_fp32(scale)  # [out, in // block] fp32
    vals = vals.reshape(*vals.shape[:-1], in_dim // block, block)
    vals = vals * scale_f32.unsqueeze(-1)
    return vals.reshape(*vals.shape[:-2], in_dim).to(torch.bfloat16)


def _to_local_expert_ids(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Map ``topk_ids`` (physical global ids) to rank-local ids, ``-1`` if remote.

    ``expert_map`` is the community ``RoutedExperts.expert_map`` buffer, i.e. a
    ``[global_num_experts]`` int32 tensor holding the local index of each global
    expert and ``-1`` for the ones this rank does not own. Pure TP (no EP) has
    ``expert_map is None``, in which case every expert is local and the ids pass
    through unchanged.

    Deliberately driven by ``expert_map`` rather than an ``[start, end)`` range:
    it is correct for every ``expert_placement_strategy`` (round-robin included)
    and for EPLB's physical-id remapping, both of which break the
    contiguous-range assumption.

    Only the torch path uses this. The ixformer path forwards ``expert_map`` to
    ``moe_compute_token_index_ep_group_layout``, which applies the same formula
    inside the routing kernel.
    """
    ids = topk_ids.long()
    if expert_map is None:
        return ids
    return expert_map[ids].long()


def ilu_mxfp4_fused_experts_torch(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    swiglu_limit: float | None = None,
    expert_map: torch.Tensor | None = None,
    output_scale: float = 1.0,
    extra_residual: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the routed MXFP4 experts owned by this rank (pure-torch oracle).

    Not on the production hot path, but not removable either -- it serves three
    roles simultaneously:

    1. the numerical oracle for ``_ilu_mxfp4_fused_experts_ixformer``;
    2. the only runnable path on non-ivcore30 hosts or when the ixmm wheel is
       unavailable (cuBLAS bf16 matmul on the dequantised weights beats the
       naive ``ixformer_native`` kernel at prefill scale);
    3. the absolute accuracy baseline for the ixmm path, whose per-256
       re-quantisation loses precision against the checkpoint's per-32 block
       scales and so can only be compared against this, never assumed equal.

    Args:
        hidden_states: ``[T, H]``, contiguous, activation dtype (bf16).
        w13_weight: ``[E_local, 2 * I, H // 2]`` uint8 (or int8 view after
            ``process_weights_after_loading``), fused gate+up.
        w13_weight_scale: ``[E_local, 2 * I, H // 32]`` uint8 e8m0 (or
            ``float8_e8m0fnu`` view after loading).
        w2_weight: ``[E_local, H, I // 2]`` uint8, down projection.
        w2_weight_scale: ``[E_local, H, I // 32]`` uint8 e8m0.
        topk_weights: ``[T, k]`` routing weights, already carrying
            ``routed_scaling_factor`` (the router applied it).
        topk_ids: ``[T, k]`` physical global expert ids.
        swiglu_limit: clamp bound for the SwiGLU; ``None`` disables clamping.
        expert_map: community global -> local expert map, ``None`` for pure TP.
        output_scale: scale applied to the routed sum, i.e.
            ``layer._ilu_output_scale``. **Not** ``routed_scaling_factor``.
        extra_residual: shared-expert output to add after scaling the routed
            sum, matching ``ixfops.moe_output_reduce_sum(extra_residual=...)``
            semantics: ``out = output_scale * routed + extra_residual``.
        w13_bias / w2_bias: optional ``[E_local, ...]`` bf16 biases. DeepSeek-V4
            has none; accepted so GPT-OSS-style MXFP4 checkpoints can reuse this.

    Returns:
        ``[T, H]`` rank-local partial in ``hidden_states.dtype``. Expert-parallel
        and tensor-parallel partials are summed by the runner's single
        all-reduce (``MoERunner._maybe_reduce_final_output``), so this function
        performs no collective.
    """
    assert hidden_states.dim() == 2, f"expected [T, H], got {hidden_states.shape}"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w13_weight.shape[0] == w2_weight.shape[0], "expert count mismatch"

    local_ids = _to_local_expert_ids(topk_ids, expert_map)
    routed = torch.zeros_like(hidden_states)

    # Dequant is per expert rather than once per layer so peak memory stays at
    # one expert's bf16 weights instead of the whole layer's.
    for local_eid in range(w13_weight.shape[0]):
        sel = local_ids == local_eid
        if not bool(sel.any()):
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)
        xin = hidden_states[tok_idx]

        w13 = dequant_mxfp4_to_bf16(w13_weight[local_eid], w13_weight_scale[local_eid])
        gate_up = torch.matmul(xin.to(w13.dtype), w13.t())
        if w13_bias is not None:
            gate_up = gate_up + w13_bias[local_eid].to(gate_up.dtype)
        act = swiglu_clamp(gate_up, swiglu_limit)

        w2 = dequant_mxfp4_to_bf16(w2_weight[local_eid], w2_weight_scale[local_eid])
        y = torch.matmul(act.to(w2.dtype), w2.t())
        if w2_bias is not None:
            y = y + w2_bias[local_eid].to(y.dtype)

        wgt = topk_weights[tok_idx, slot_idx].unsqueeze(-1).to(y.dtype)
        routed.index_add_(0, tok_idx, (y * wgt).to(routed.dtype))

    if output_scale != 1.0:
        routed = routed * output_scale
    if extra_residual is not None:
        # Not in-place: ``extra_residual`` is the SharedExperts-owned output
        # tensor and the caller may still hold a reference to it.
        routed = routed + extra_residual.to(routed.dtype)
    return routed


# ---------------------------------------------------------------------------
# ixformer ep_moe + group_gemm_fp4 pipeline
# ---------------------------------------------------------------------------

# Backend names accepted by the dispatcher. Kept as plain strings (not an
# enum) so they can be passed straight through to ``group_gemm_fp4``'s
# ``backend=`` kwarg, which is itself a string.
_TORCH_BACKEND = "torch"
_IXFORMER_NATIVE_BACKEND = "ixformer_native"
_IXMM_BACKEND = "ixmm"
_VALID_BACKENDS = (_TORCH_BACKEND, _IXFORMER_NATIVE_BACKEND, _IXMM_BACKEND)
_AUTO_BACKEND = "auto"

# SwiGLU activation variants for the ixformer step-4 fusion. The default keeps
# the DeepSeek-V4 / MXFP4 behaviour (clamp before silu, no no-clamp mode) so
# existing callers are unaffected; Step-3.7 NVFP4 selects the community
# ``swiglustep`` semantics (clamp after silu, ``limit <= 0`` = plain SwiGLU).
_DSV4_CLAMP_VARIANT = "dsv4_clamp"
_STEP3_SWIGLUSTEP_VARIANT = "step3_swiglustep"
# MiniMax-M3 routes its experts through community
# ``swigluoai_uninterleave``, whose math matches neither of the above: gate/up
# are *packed* halves (``x[..., :d]`` / ``x[..., d:]``, as emitted by a
# MergedColumnParallelLinear gate_up_proj), the gate carries an ``alpha`` inside
# the sigmoid, and the up branch is offset by ``beta`` --
# ``gate.clamp(max=limit) * sigmoid(alpha * gate) * (up.clamp(±limit) + beta)``.
# Note ixformer's ``swigluoai_and_mul`` is the *interleaved* gpt-oss split and
# must not be used here. ``swigluoai_uninterleave_dynamic_scaled_fp8`` exposes
# the packed alpha/beta semantics directly.
_SWIGLUOAI_VARIANT = "swigluoai"
_VALID_SWIGLU_VARIANTS = (
    _DSV4_CLAMP_VARIANT,
    _STEP3_SWIGLUSTEP_VARIANT,
    _SWIGLUOAI_VARIANT,
)
# Defaults of community ``SiluAndMulWithClamp``
# (vllm/model_executor/layers/activation.py), which is the class that backs
# ``MoEActivation.SWIGLUOAI_UNINTERLEAVE`` -- NOT the interleaved
# ``SwigluOAIAndMul`` (1.702 / 1.0). These mean "the caller passed nothing",
# i.e. plain ``silu(gate) * up``, and are deliberately NOT MiniMax-M3's values:
# otherwise a caller that forgets to thread alpha/beta silently gets M3's
# activation applied to some other checkpoint. Callers that require the real
# values guard for ``None`` themselves (see ``IluMxFp8FusedMoE.apply``).
_SWIGLUOAI_ALPHA = 1.0
_SWIGLUOAI_BETA = 0.0


def _swigluoai_uninterleave(
    input: torch.Tensor,
    limit: float,
    *,
    alpha: float = _SWIGLUOAI_ALPHA,
    beta: float = _SWIGLUOAI_BETA,
) -> torch.Tensor:
    """MiniMax-M3 packed-halves SwiGLU-OAI activation."""
    d = input.shape[-1] // 2
    gate = input[..., :d].clamp(max=limit)
    up = input[..., d:].clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


@functools.lru_cache(maxsize=1)
def _ixmm_backend_available() -> bool:
    """True iff the ixmm ``group_gemm_fp4`` backend is usable on this build.

    Mirrors ixformer's own ``_ixmm_available`` (``gemm.py:620``) but goes
    through the public registry so a missing ixformer import degrades to
    ``False`` rather than raising. Cached because it is consulted per layer per
    step in ``auto`` mode and the answer is fixed for the process.
    """
    try:
        import ixformer.inference.backend_registry as _registry  # type: ignore
    except ImportError:
        return False
    try:
        return _IXMM_BACKEND in _registry.registry.available_backends("group_gemm_fp4")
    except Exception:
        # Not "ixmm is absent" but "the probe itself broke". Silently degrading
        # here costs an order of magnitude of throughput with nothing in the
        # log, so say it once.
        logger.warning(
            "Failed to probe the ixmm group_gemm_fp4 backend; falling back to "
            "the torch MXFP4 path.",
            exc_info=True,
        )
        return False


def _resolve_backend(
    backend: str | None,
    hidden_states: torch.Tensor,
    *,
    swiglu_limit: float | None,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> str:
    """Pick the compute backend for ``ilu_mxfp4_fused_experts``.

    The return value is one of the ``group_gemm_fp4`` backend strings
    (``"torch"`` / ``"ixformer_native"`` / ``"ixmm"``). ``"torch"`` means the
    pure-torch oracle path is taken (no ixformer call at all).

    A failed ixformer precondition falls back to ``"torch"`` only while weights
    still use the original layout. ixmm preprocessing is not reversible here,
    so an already-preprocessed layer fails fast instead of being misread by the
    torch oracle. ``ixformer_native`` is never selected by ``auto``.
    """
    # 1) Resolve the requested backend (env var or call-site kwarg).
    if backend is None:
        import vllm.envs as envs

        backend = envs.VLLM_ILU_MXFP4_MOE_BACKEND
    if backend not in (_AUTO_BACKEND, *_VALID_BACKENDS):
        raise ValueError(
            f"Unknown MXFP4 MoE backend {backend!r}; expected one of "
            f"{(_AUTO_BACKEND, *_VALID_BACKENDS)}"
        )

    raw_format = (
        w13_weight.dtype is torch.uint8
        and w2_weight.dtype is torch.uint8
        and w13_weight_scale.dtype is torch.uint8
        and w2_weight_scale.dtype is torch.uint8
    )
    original_format = (
        w13_weight.dtype is torch.int8
        and w2_weight.dtype is torch.int8
        and w13_weight_scale.dtype in (torch.float8_e8m0fnu, torch.float32)
        and w2_weight_scale.dtype in (torch.float8_e8m0fnu, torch.float32)
    )
    ixmm_format = (
        w13_weight.dtype is torch.uint8
        and w2_weight.dtype is torch.uint8
        and w13_weight_scale.dtype is torch.float32
        and w2_weight_scale.dtype is torch.float32
    )

    def torch_fallback(reason: str) -> str:
        if ixmm_format:
            raise RuntimeError(
                "MXFP4 weights are already in ixmm's preprocessed layout and "
                f"cannot fall back to the original-layout torch path: {reason}"
            )
        if not (raw_format or original_format):
            raise RuntimeError(
                "MXFP4 weights have an unsupported dtype/layout combination: "
                f"w13=({w13_weight.dtype}, {w13_weight_scale.dtype}), "
                f"w2=({w2_weight.dtype}, {w2_weight_scale.dtype})."
            )
        return _TORCH_BACKEND

    if backend == _TORCH_BACKEND:
        return torch_fallback("backend='torch' was requested")

    # 2) Every ixformer backend, including an explicit one, must satisfy these
    # hard preconditions. Explicit selection only controls the implementation;
    # it must not bypass input validity checks.
    if not hidden_states.is_cuda:
        return torch_fallback("hidden_states is not a CUDA tensor")
    try:
        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        ivcore30 = is_ivcore30()
    except Exception:
        return torch_fallback("device architecture could not be determined")
    if not ivcore30:
        return torch_fallback("device architecture is not ivcore30")
    if hidden_states.dtype is not torch.bfloat16:
        return torch_fallback(f"activation dtype is {hidden_states.dtype}, not bf16")
    if swiglu_limit is None:
        return torch_fallback("swiglu_limit is None")
    if w13_bias is not None or w2_bias is not None:
        return torch_fallback("MoE bias is present")

    # 3) Backend and weight layout are inseparable. ixmm consumes only the
    # preprocessed ix30 layout; native consumes only the original MXFP4 layout.
    if backend == _IXMM_BACKEND:
        if not ixmm_format:
            raise RuntimeError(
                "MXFP4 MoE backend 'ixmm' requires weights preprocessed by "
                "preprocess_weight_for_ixmm(): uint8 weights and float32 scales. "
                f"Got w13=({w13_weight.dtype}, {w13_weight_scale.dtype}), "
                f"w2=({w2_weight.dtype}, {w2_weight_scale.dtype})."
            )
        if not _ixmm_backend_available():
            raise RuntimeError(
                "MXFP4 MoE backend 'ixmm' is not available. Install the ixmm "
                "wheel and rebuild ixformer with HAVE_IXMM on ivcore30."
            )
        return _IXMM_BACKEND

    if backend == _IXFORMER_NATIVE_BACKEND:
        if not original_format:
            raise RuntimeError(
                "MXFP4 MoE backend 'ixformer_native' requires original-layout "
                "int8 weights with e8m0/float32 scales."
            )
        return _IXFORMER_NATIVE_BACKEND

    # ``auto`` selects a backend compatible with the format produced at load
    # time. Native is intentionally never automatic because its GEMV-shaped
    # kernel is unusable for prefill-sized workloads.
    if ixmm_format and _ixmm_backend_available():
        return _IXMM_BACKEND
    return torch_fallback("no compatible ixmm backend is available")


def _ilu_mxfp4_fused_experts_ixformer(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    swiglu_limit: float | None,
    expert_map: torch.Tensor | None,
    output_scale: float,
    extra_residual: torch.Tensor | None,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    gemm_backend: str,
    gemm_mode: int | None,
    swiglu_variant: str,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> torch.Tensor:
    """The ixformer pipeline.

    Six steps: routing -> expand+E4M3 -> GEMM1 -> SwiGLU+E4M3 -> GEMM2
    -> gather+reduce. Stays in compact expert order between GEMM1 and the
    final gather; ``fuse_swiglu=False`` because the SwiGLU needs the clamp and
    the community w13 layout is gate-then-up, not 16-column interleaved.

    ``gemm_backend`` is one of ``"torch"`` / ``"ixformer_native"`` / ``"ixmm"``
    and is forwarded verbatim to ``group_gemm_fp4``. It also selects how the
    weight shapes are read: ``ixmm`` consumes the preprocessed ``[E, K/2, N]``
    layout, the other two the original ``[E, N, K/2]``. For ixmm, K may include
    right-padding to a 256-element group boundary; the matching activation is
    zero-padded here before each GEMM.

    ``gemm_mode`` is forwarded to ``group_gemm_fp4``: ``None`` (default)
    auto-selects gemv when ``M <= 128``, else gemm; explicit ``0`` / ``1``
    override that choice.

    ``swiglu_variant`` selects the step-4 activation op:
    ``"dsv4_clamp"`` (default) -> ``swiglu_limit_dynamic_scaled_fp8`` (DSv4
    clamp-before-silu); ``"step3_swiglustep"`` ->
    ``swiglustep_dynamic_scaled_fp8`` (clamp-after-silu, ``limit <= 0`` =
    plain SwiGLU).
    """
    # Lazy import: keeps this module importable on a CPU host / when ixformer
    # is absent. The dispatcher only reaches here on the CUDA + ivcore30 path.
    import ixformer.inference.functions as ixfops  # type: ignore

    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k
    E_local = w13_weight.shape[0]
    if gemm_backend == _IXMM_BACKEND:
        # ixmm layout: [E, K/2, N].
        assert w13_weight.dtype is torch.uint8
        assert w2_weight.dtype is torch.uint8
        assert w13_weight.shape[1] * 2 == H
        w13_out = w13_weight.shape[2]
        inter_size = w13_out // 2
        assert w13_out % 2 == 0
        w2_k = w2_weight.shape[1] * 2
        expected_w2_k = (
            (inter_size + _IXMM_GROUP_SIZE - 1) // _IXMM_GROUP_SIZE * _IXMM_GROUP_SIZE
        )
        assert w2_k == expected_w2_k
        assert w2_weight.shape[2] == H
        assert w2_weight_scale.shape == (
            E_local,
            w2_k // _IXMM_GROUP_SIZE,
            H,
        )
    else:
        # Original layout: [E, N, K/2].
        assert w13_weight.dtype is torch.int8
        assert w2_weight.dtype is torch.int8
        assert w13_weight.shape[2] * 2 == H
        w13_out = w13_weight.shape[1]
        inter_size = w13_out // 2
        assert w13_out % 2 == 0
        assert w2_weight.shape[2] * 2 == inter_size
        assert w2_weight.shape[1] == H
        w2_k = inter_size

    # --- 0) Preconditions (each guards a "wrong but no error" case) -------
    assert hidden_states.dtype is torch.bfloat16, (
        "ixformer MoE pipeline requires bf16 activation "
        f"(got {hidden_states.dtype}); dispatcher should have fallen back."
    )
    assert swiglu_variant in _VALID_SWIGLU_VARIANTS, (
        f"unknown swiglu_variant {swiglu_variant!r}; expected one of "
        f"{_VALID_SWIGLU_VARIANTS}"
    )
    # Only the DSv4 op lacks a no-clamp mode. ``step3_swiglustep`` treats
    # ``None`` / ``0`` as plain SwiGLU natively (layers 3-42).
    assert swiglu_variant != _DSV4_CLAMP_VARIANT or swiglu_limit is not None, (
        "swiglu_limit must be set for swiglu_variant='dsv4_clamp': "
        "swiglu_limit_dynamic_scaled_fp8 has no 'no clamp' mode"
    )
    # ``swigluoai`` clamps unconditionally (community default 7.0), so a missing
    # or non-positive limit would silently change the activation rather than
    # disable a clamp.
    assert swiglu_variant != _SWIGLUOAI_VARIANT or (
        swiglu_limit is not None and swiglu_limit > 0
    ), (
        "swiglu_limit must be a positive clamp bound for "
        f"swiglu_variant='{_SWIGLUOAI_VARIANT}', got {swiglu_limit!r}"
    )
    assert w13_bias is None and w2_bias is None, (
        "bias is not wired into group_gemm_fp4; the dispatcher should have "
        "fallen back to the torch path"
    )

    # --- 1) Routing: the op owns the global -> local remap ----------------
    #     ``expert_map`` is forwarded as-is: ``None`` for pure TP / single card
    #     (ids are already local), otherwise the community
    #     ``[global_num_experts]`` int32 map with ``-1`` for remote experts.
    #     Remote and out-of-range routes come back as ``-1`` in ``src_to_dst``
    #     and are skipped by the final gather, so no host-side early-out is
    #     needed even when this rank owns none of the routed experts.
    src_to_dst, dst_to_src, group_layout = (
        ixfops.moe_compute_token_index_ep_group_layout(
            topk_ids, 0, E_local, expert_map=expert_map
        )
    )

    # --- 2) Expand + per-token E4M3 -------------------------------------
    a1_fp8, a1_scale = ixfops.moe_expand_input_dynamic_scaled_fp8(
        hidden_states, dst_to_src, top_k
    )

    # --- 3) GEMM1 (compact expert order, no fused swiglu: DSv4 needs clamp)
    gemm1_out = torch.empty(
        (M, 2 * inter_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    ixfops.group_gemm_fp4(
        a1_fp8,
        a1_scale,
        w13_weight,
        w13_weight_scale,
        group_layout,
        gemm1_out,
        row_mapping=None,
        mode=gemm_mode,
        fuse_swiglu=False,
        backend=gemm_backend,
    )

    # --- 4) SwiGLU + E4M3 -----------------------------------------------
    if swiglu_variant == _SWIGLUOAI_VARIANT:
        # The activation variant defines the layout and formula. Its parameters
        # are data, not a signal for selecting an implementation.
        alpha = _SWIGLUOAI_ALPHA if swiglu_alpha is None else float(swiglu_alpha)
        beta = _SWIGLUOAI_BETA if swiglu_beta is None else float(swiglu_beta)
        a2_fp8, a2_scale = ixfops.swigluoai_uninterleave_dynamic_scaled_fp8(
            gemm1_out,
            dst_to_src,
            float(swiglu_limit),
            alpha,
            beta,
            backend=_IXFORMER_NATIVE_BACKEND,
        )
    elif swiglu_variant == _STEP3_SWIGLUSTEP_VARIANT:
        # Step-3.5 swiglustep: clamp after silu; ``limit <= 0`` = plain SwiGLU.
        a2_fp8, a2_scale = ixfops.swiglustep_dynamic_scaled_fp8(
            gemm1_out, dst_to_src, float(swiglu_limit or 0.0)
        )
    else:
        # DSv4 / MXFP4: clamp before silu (unchanged legacy behaviour).
        a2_fp8, a2_scale = ixfops.swiglu_limit_dynamic_scaled_fp8(
            gemm1_out, dst_to_src, float(swiglu_limit)
        )
    a2_fp8 = _zero_pad_last_dim(a2_fp8, w2_k)

    # --- 5) GEMM2 -------------------------------------------------------
    gemm2_out = torch.empty((M, H), dtype=torch.bfloat16, device=hidden_states.device)
    ixfops.group_gemm_fp4(
        a2_fp8,
        a2_scale,
        w2_weight,
        w2_weight_scale,
        group_layout,
        gemm2_out,
        row_mapping=None,
        mode=gemm_mode,
        fuse_swiglu=False,
        backend=gemm_backend,
    )

    # --- 6) Gather + top-k weighted reduce (+ output_scale + residual) ---
    return ixfops.moe_gather_output_reduce_sum(
        gemm2_out,
        topk_weights.float().contiguous(),
        src_to_dst,
        extra_residual=extra_residual,
        scaling_factor=float(output_scale),
    )


def ilu_mxfp4_fused_experts(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    swiglu_limit: float | None = None,
    expert_map: torch.Tensor | None = None,
    output_scale: float = 1.0,
    extra_residual: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    backend: str | None = None,
    swiglu_variant: str = _DSV4_CLAMP_VARIANT,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> torch.Tensor:
    """Dispatcher for the routed MXFP4 expert compute.

    Takes the same arguments as ``ilu_mxfp4_fused_experts_torch`` plus two
    kw-only ones:

    * ``backend``: force a ``group_gemm_fp4`` backend. ``None`` defers to
      ``VLLM_ILU_MXFP4_MOE_BACKEND``, whose default is ``ixmm``.
    * ``swiglu_variant``: step-4 activation semantics. ``"dsv4_clamp"``
      (default, DSv4/MXFP4, unchanged), ``"step3_swiglustep"`` (Step-3.7
      NVFP4, clamp-after-silu + native no-clamp), or ``"swigluoai"``
      (MiniMax-M3 MXFP8, community ``swigluoai_uninterleave`` semantics:
      packed-halves gate/up, alpha inside the sigmoid, ``up + 1``). Only affects
      the ixformer path; the torch fallback ignores it (see below).

    There is no ``run_decode``: ``group_gemm_fp4`` picks gemv vs gemm itself
    from ``M`` (``mode=None``, ``M <= 128`` -> gemv).

    Returns the ``[T, H]`` rank-local partial; the runner's all-reduce
    completes the EP/TP partials.
    """
    if hidden_states.shape[0] == 0:
        # An empty batch has no backend-specific behaviour, and routing it into
        # ``_resolve_backend`` would turn it into a hard error once the weights
        # are in ixmm's layout (no torch fallback from there).
        empty = torch.zeros_like(hidden_states)
        if extra_residual is not None:
            empty = empty + extra_residual
        return empty
    be = _resolve_backend(
        backend,
        hidden_states,
        swiglu_limit=swiglu_limit,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
    )
    if be == _TORCH_BACKEND:
        return ilu_mxfp4_fused_experts_torch(
            hidden_states,
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            topk_weights,
            topk_ids,
            swiglu_limit=swiglu_limit,
            expert_map=expert_map,
            output_scale=output_scale,
            extra_residual=extra_residual,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
    # ``gemm_mode=None`` lets ``group_gemm_fp4`` auto-select: gemv when
    # ``M = T * top_k <= 128``, else gemm.
    return _ilu_mxfp4_fused_experts_ixformer(
        hidden_states,
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
        topk_weights,
        topk_ids,
        swiglu_limit=swiglu_limit,
        expert_map=expert_map,
        output_scale=output_scale,
        extra_residual=extra_residual,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        gemm_backend=be,
        gemm_mode=None,
        swiglu_variant=swiglu_variant,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
