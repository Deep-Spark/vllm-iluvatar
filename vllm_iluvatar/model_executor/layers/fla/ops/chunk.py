# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
"""Iluvatar GDN FLA chunk entry points.

This module mirrors the community FLA chunk wrapper but routes the
``chunk_scaled_dot_kkt_fwd`` step to the Iluvatar kernel implementation. The
remaining FLA stages stay upstream-owned.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.utils import (
    FLA_CHUNK_SIZE,
    SUPPRESS_LEVEL,
)
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

from vllm_iluvatar.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(
        g,
        chunk_size=FLA_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    if SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new
    raise AssertionError("unreachable")


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                "when using `cu_seqlens`. Please flatten variable-length inputs "
                "before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} rather "
                f"than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    g, o, A, final_state, _w, _h, _v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        core_attn_out=core_attn_out,
    )
    if core_attn_out is not None:
        assert not torch.is_grad_enabled(), (
            "core_attn_out buffer reuse is only supported for inference"
        )
        assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
    return o.to(q.dtype), final_state
