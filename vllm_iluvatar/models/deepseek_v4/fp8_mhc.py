# SPDX-License-Identifier: Apache-2.0
"""Torch-equivalent MHC (multi-head channel) ops for the raw-FP8 DS-V4 model.

Stage 2, module 1 (design: ``整个框架适配计划/模块1-编排骨架.md``). These are pure
PyTorch ports of the community NVIDIA MHC kernels, kept behind the *same*
interface so a later pass can swap them for the Iluvatar ``ixfops.ds4_mhc_*``
kernels without touching the orchestration in ``fp8_model.py``.

Sources (byte-for-byte semantic reference):

* ``mhc_pre`` / ``mhc_post`` -- ``vllm/model_executor/kernels/mhc/torch.py``
  (``mhc_pre_torch:6-91`` / ``mhc_post_torch:94-106``). The only addition over
  the upstream torch reference is the *fused RMSNorm*: the production TileLang
  kernel folds ``attn_norm`` / ``ffn_norm`` into ``layer_input``
  (``mhc/tilelang_kernels.py:305-350``; NVIDIA ``nvidia/model.py:932`` notes the
  norm is fused into ``mhc_pre``), so we replicate it when ``norm_weight`` is
  passed rather than calling the norm module separately.
* ``mhc_fused_post_pre`` -- the NVIDIA fused-boundary kernel
  ``mhc_fused_post_pre_tilelang`` (``nvidia/model.py:913-930``). It is exactly
  the previous sub-layer's ``mhc_post`` composed with the current sub-layer's
  ``mhc_pre``, so we compose the two here and keep NVIDIA's return order
  ``(residual, post_mix, res_mix, layer_input)``.
* ``hc_head`` -- ``mhc/tilelang_kernels.py::hc_head_fuse_tilelang:882-976`` and
  the matching torch/Triton reference ``mhc/triton.py::
  hc_head_reduce_triton_kernel:108-140``.

dtype contract (asserted, matching upstream): ``residual`` bf16; ``fn`` /
``hc_scale`` / ``hc_base`` fp32; ``norm_weight`` bf16. Outputs: ``post_mix``
``(..., C, 1)`` fp32, ``res_mix`` ``(..., C, C)`` fp32, ``layer_input``
``(..., H)`` bf16.

Functional (not bitwise) equivalence: intermediate math runs in fp32 and only
the final ``layer_input`` / ``hc_head`` output is rounded to bf16, so the GPU
kernels' per-pass bf16 rounding is not reproduced. That is acceptable for the
"run + single-sentence correct" bar; bitwise parity is a kernel-swap concern.
"""

from __future__ import annotations

import torch


def _sinkhorn_normalize(
    comb_logits: torch.Tensor, sinkhorn_repeat: int, hc_sinkhorn_eps: float
) -> torch.Tensor:
    """Row/column alternating Sinkhorn normalisation on the ``(..., C, C)`` mix.

    Mirrors ``mhc_pre_torch`` (``mhc/torch.py:78-82``): first a ``softmax`` over
    the last dim, then a column normalisation, then ``sinkhorn_repeat - 1``
    rounds of alternating row/column normalisation.
    """
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return comb_mix


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre block. Port of ``mhc_pre_torch`` + optional fused RMSNorm.

    Args:
        residual: ``(..., C, H)`` bf16 residual stream.
        fn: ``(mix_hc, C*H)`` fp32, ``mix_hc = (2 + C) * C``.
        hc_scale: ``(3,)`` fp32 (alpha / beta / gamma).
        hc_base: ``(mix_hc,)`` fp32.
        norm_weight: ``(H,)`` bf16; when given, ``attn_norm`` / ``ffn_norm`` is
            fused into ``layer_input`` exactly as the production kernel does.

    Returns:
        ``post_mix`` ``(..., C, 1)`` fp32, ``res_mix`` ``(..., C, C)`` fp32,
        ``layer_input`` ``(..., H)`` bf16.
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = _sinkhorn_normalize(comb_logits, sinkhorn_repeat, hc_sinkhorn_eps)

    # raw = sum_c pre_mix[c] * residual[c]  (fp32 accumulate), rounded to bf16 --
    # this is the unnormed ``layer_input`` the upstream torch reference returns.
    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)  # (num_tokens, H)
    if norm_weight is not None:
        # Fold attn_norm / ffn_norm into layer_input, exactly as a separate
        # ``RMSNorm(layer_input)`` (vLLM RMSNorm: upcast bf16 -> fp32, normalise
        # over H, multiply weight). Keeps the invariant
        # ``mhc_pre(norm=w) == attn_norm(mhc_pre(norm=None))``.
        r = layer_input.to(torch.float32)
        inv_rms = torch.rsqrt(r.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = (r * inv_rms * norm_weight.to(torch.float32)).to(
            torch.bfloat16
        )

    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post block. Byte-for-byte port of ``mhc_post_torch`` (``torch.py:94``).

    Args:
        x: ``(..., H)`` sub-layer (attn / ffn) output.
        residual: ``(..., C, H)`` residual stream carried into the sub-layer.
        post_layer_mix: ``(..., C, 1)`` fp32 (the ``post_mix`` from ``mhc_pre``).
        comb_res_mix: ``(..., C, C)`` fp32 (the ``res_mix`` from ``mhc_pre``).

    Returns:
        ``(..., C, H)`` residual stream in ``residual.dtype`` (bf16).
    """
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused boundary op: previous sub-layer's ``mhc_post`` then this one's ``mhc_pre``.

    Equivalent to NVIDIA ``mhc_fused_post_pre_tilelang`` (``nvidia/model.py:913``);
    returns NVIDIA's order ``(new_residual, post_mix, res_mix, layer_input)``.

    Args:
        x: ``(..., H)`` output of the *previous* sub-layer (attn or ffn).
        residual / post_mix / res_mix: the state carried out of the previous
            ``mhc_pre`` / ``mhc_fused_post_pre``.
        fn / hc_scale / hc_base / eps...: this sub-layer's MHC parameters.
        norm_weight: this sub-layer's norm weight (fused into ``layer_input``).
    """
    new_residual = mhc_post(x, residual, post_mix, res_mix)
    post_mix2, res_mix2, layer_input = mhc_pre(
        new_residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return new_residual, post_mix2, res_mix2, layer_input


def hc_head(
    hidden_states: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse the ``C`` residual streams to a single ``(..., H)`` tensor.

    Torch equivalent of ``hc_head_fuse_tilelang`` (``mhc/tilelang_kernels.py:882``)
    / ``hc_head_reduce_triton_kernel`` (``mhc/triton.py:108``):

        x_flat  = hidden_states.flatten(-2)                    # (..., C*H)
        rsqrt   = rsqrt(sum(x_flat^2) / (C*H) + rms_norm_eps)
        mixes   = (x_flat @ fn.T) * rsqrt                      # (..., C)
        pre     = sigmoid(mixes * hc_scale + hc_base) + hc_eps # (..., C)
        out     = sum_c pre[c] * hidden_states[c]              # (..., H)

    The weighted sum uses the *raw* (un-normed) streams, matching both kernels.

    Args:
        hidden_states: ``(..., C, H)`` bf16.
        fn: ``(C, C*H)`` fp32.
        hc_scale: ``(1,)`` fp32.
        hc_base: ``(C,)`` fp32.

    Returns:
        ``(..., H)`` bf16.
    """
    assert hidden_states.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)

    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.reshape(-1, hc_mult, hidden_size)
    num_tokens = hs_flat.shape[0]

    x_flat = hs_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    sqrsum = x_flat.square().sum(dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_norm_eps)
    mixes = torch.matmul(x_flat, fn.t()) * rsqrt_val  # (num_tokens, C)
    pre_mix = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # (num_tokens, C)

    out = torch.sum(pre_mix.unsqueeze(-1) * x_flat.reshape(
        num_tokens, hc_mult, hidden_size
    ), dim=1)  # (num_tokens, H)
    return out.to(torch.bfloat16).reshape(*outer_shape, hidden_size)
