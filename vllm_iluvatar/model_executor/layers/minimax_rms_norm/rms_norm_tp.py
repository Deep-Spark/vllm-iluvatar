# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for the MiniMax QK-Norm forward path.

Corresponds to: vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py

Target   : vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp
           .MiniMaxText01RMSNormTP.forward_qkv
Baseline : main_4cd213ead42e
Why      : the community ``forward_qkv`` sends the TP>1 case through
           ``torch.ops.vllm.minimax_qk_norm_fusion``, whose three tiers all fail
           or are risky on Iluvatar corex: the Lamport fused op
           (``torch.ops._C.minimax_allreduce_rms_qk``) is not registered in the
           plugin ``_C`` (falls to ``None``), the ROCm AITER path is not ROCm,
           and the Triton double-kernel fallback is unsafe on corex.
How      : ``forward_qkv`` dispatches to ixformer ``rms_norm_qk`` when TP==1,
           and to the two-stage plugin ops ``torch.ops._C.minimax_qk_var`` +
           all-reduce + ``torch.ops._C.minimax_qk_rms_apply`` when TP>1.
           Installed by ``ilu_patch.general_patches.patch_minimax_qk_norm`` via
           ``setattr(MiniMaxText01RMSNormTP, "forward_qkv", staticmethod(...))``.
           ``forward_qkv_torch`` keeps the pure-torch reference for tests /
           debugging and is not used at runtime.
Scope    : all MiniMax-M2 (``minimax_m2``) QK-Norm forwards on Iluvatar.
Remove   : when upstream gains a portable platform hook that covers this path.
"""

from __future__ import annotations

from typing import Any

import torch
from ixformer.inference.functions import rms_norm_qk
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size


def forward_qkv_torch(
    q_norm: Any,
    k_norm: Any,
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch reference QK-Norm over the packed qkv tensor.

    Equivalent to the community ``_minimax_qk_norm_tp_eager`` (variance in fp32,
    optional cross-TP all-reduce, per-channel weight) with ``v`` returned
    unchanged. Kept for numerical tests and debugging; runtime uses
    ``forward_qkv``.
    """
    assert qkv.ndim == 2
    assert q_norm.variance_epsilon == k_norm.variance_epsilon
    eps = q_norm.variance_epsilon

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    orig_dtype = q.dtype
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)

    tp_world = get_tensor_model_parallel_world_size()
    if tp_world > 1:
        # Each shard holds only part of the heads, so the per-shard means must
        # be summed across TP and divided by tp_world to recover the global
        # mean-of-squares. Flattening keeps the fp32 reduction off FlashInfer's
        # 16-bit fast path (matches community ``_all_reduce_variance``).
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = tensor_model_parallel_all_reduce(qk_var.flatten()).view_as(qk_var)
        qk_var = qk_var / tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)

    q_out = (q_fp32 * torch.rsqrt(q_var + eps) * q_norm.weight).to(orig_dtype)
    k_out = (k_fp32 * torch.rsqrt(k_var + eps) * k_norm.weight).to(orig_dtype)
    return q_out, k_out, v


def forward_qkv(
    q_norm: Any,
    k_norm: Any,
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Operator-backed QK-Norm over the packed qkv tensor.

    TP==1 uses ixformer ``rms_norm_qk``; TP>1 uses the two-stage plugin ops
    ``minimax_qk_var`` / ``minimax_qk_rms_apply`` with a Python all-reduce
    between them. Returns ``(q_out, k_out, v)`` with ``v`` unchanged.

    Constraints (hard-fail via TORCH_CHECK / ixformer checks, no fallback):
      dtype   : ``qkv`` / ``q_norm.weight`` / ``k_norm.weight`` are fp16 or
                bf16 and share dtype.
      shape   : ``qkv`` is 2D ``[T, >= q_size + 2*kv_size]``;
                weight numel equals ``q_size`` / ``kv_size``.
      stride  : ``qkv.stride(-1) == 1`` (outer strides may be arbitrary;
                views from ``qkv.split`` are accepted).
      values  : ``q_size > 0``, ``kv_size > 0``,
                ``q_norm.variance_epsilon == k_norm.variance_epsilon > 0``.
      output  : contiguous ``q_out [T, q_size]``, ``k_out [T, kv_size]``,
                dtype == ``qkv``.
    """
    assert qkv.ndim == 2
    assert q_norm.variance_epsilon == k_norm.variance_epsilon
    eps = q_norm.variance_epsilon

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    tp_world = get_tensor_model_parallel_world_size()
    if tp_world == 1:
        q_out, k_out = rms_norm_qk(q, k, q_norm.weight, k_norm.weight, eps)
    else:
        # Stage 1: local mean-of-squares -> fp32 [T, 2].
        qk_var = torch.ops._C.minimax_qk_var(qkv, q_size, kv_size)
        # Flatten keeps the fp32 reduction off FlashInfer's 16-bit fast path
        # (matches community ``_all_reduce_variance``). /tp_world is folded
        # into the apply kernel's rsqrt.
        qk_var = tensor_model_parallel_all_reduce(qk_var.flatten()).view_as(qk_var)
        q_out, k_out = torch.ops._C.minimax_qk_rms_apply(
            qkv,
            qk_var,
            q_norm.weight,
            k_norm.weight,
            q_size,
            kv_size,
            tp_world,
            eps,
        )
    return q_out, k_out, v
