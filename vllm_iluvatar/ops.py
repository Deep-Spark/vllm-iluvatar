"""Iluvatar-specific implementations of ops that vLLM's ``_custom_ops``
either lacks (``silu_and_mul_with_clamp``, ``router_gemm_bf16_fp32``) or
ships a CUDA-only kernel for (``topk_hash_softplus_sqrt``).

These bodies are factored out of the patch module so they can be unit
tested in isolation and reused from our own model code without going
through ``vllm._custom_ops``.

Adapted from support_dsv4 branch of vllm fork.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (kept for backward-compat import)
import vllm.envs as envs


def _ixf_ops():  # pragma: no cover - thin import shim
    """Return ``ixformer.inference.functions``; imported lazily so that
    importing this module doesn't crash on machines without ixformer."""
    import ixformer.inference.functions as ixf_ops  # type: ignore[import]

    return ixf_ops


def silu_and_mul_with_clamp(out: torch.Tensor, x: torch.Tensor, limit: float) -> None:
    """In-place ``out = silu_and_mul(clamp(x, limit))`` via ixformer fused op."""
    _ixf_ops().silu_and_mul_with_clamp(input=x, output=out, limit=limit)


def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``bf16 x bf16 -> fp32`` GEMM with fp32 accumulate.

    ``weight`` has shape ``(N, K)`` so we want ``input @ weight.T``. The
    ixformer op ``gemm_bf16_bf16_fp32`` expects format ``"TN"`` for this
    layout (weight rows are ``N``, weight cols are ``K``); numerically
    equivalent to the upstream cuBLAS kernel but runs natively on
    iluvatar instead of falling back to a fp32 cast + matmul.
    """
    return _ixf_ops().gemm_bf16_bf16_fp32(input, weight, format="TN")


def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> None:
    """Pure PyTorch equivalent of ``torch.ops._moe_C.topk_softplus_sqrt``.

    Writes the three pre-allocated output buffers in place:

    * ``topk_weights``         ``(num_tokens, top_k)`` fp32
    * ``topk_indices``         ``(num_tokens, top_k)`` int32/int64
    * ``token_expert_indices`` ``(num_tokens, top_k)`` int32/int64

    Pipeline:
      1. ``score = sqrt(softplus(gating_output))``
      2. expert selection - either ``hash_indices_table[input_tokens]`` for
         hash routing, or ``topk(score + e_score_correction_bias)``.
      3. gather scores at chosen indices, optional renormalize, then
         multiply by ``routed_scaling_factor`` (matches the kernel which
         applies the scale internally).
      4. ``token_expert_indices[t, j] = j`` (positional helper).
    """
    n_routed_experts = gating_output.shape[-1]
    num_tokens, top_k = topk_weights.shape

    scores = torch.nn.functional.softplus(gating_output).sqrt()

    if hash_indices_table is not None:
        assert (
            input_tokens is not None
        ), "hash routing requires input_tokens to look up the hash table"
        ids = hash_indices_table[input_tokens]
    else:
        if e_score_correction_bias is not None:
            scores_for_choice = scores.view(-1, n_routed_experts) + (
                e_score_correction_bias.unsqueeze(0)
            )
        else:
            scores_for_choice = scores.view(-1, n_routed_experts)
        use_sorted = envs.VLLM_BATCH_INVARIANT
        ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=use_sorted)[1]

    weights = scores.gather(1, ids.long()).to(torch.float32)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor

    topk_weights.copy_(weights)
    topk_indices.copy_(ids.to(topk_indices.dtype))
    token_expert_indices.copy_(
        torch.arange(
            top_k,
            device=token_expert_indices.device,
            dtype=token_expert_indices.dtype,
        )
        .unsqueeze(0)
        .expand(num_tokens, top_k)
    )


def dsa_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Route to the ixformer kernel (the upstream ``ops.``-prefixed alias
    does not exist on iluvatar)."""
    return _ixf_ops().dsa_convert_req_index_to_global_index(
        req_id, block_table, token_indices, block_size, output
    )
