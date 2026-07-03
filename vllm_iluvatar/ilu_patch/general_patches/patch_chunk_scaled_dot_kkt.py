# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar FLA chunk_scaled_dot_kkt override.

Corresponds to: vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel
Baseline : main_4875b4456b5
Why      : the fork adjusts the Triton ``tl.dot`` dtype conversion inside the
           FLA chunk kernel for Iluvatar.
How      : guard the copied community kernel source with a fingerprint, then
           rebind the community module-global ``chunk_scaled_dot_kkt_fwd_kernel``.
           The live community ``chunk_scaled_dot_kkt_fwd`` wrapper resolves that
           global at call time, so the wrapper is not copied.
Scope    : FLA chunk attention paths on Iluvatar.
Remove   : when upstream exposes a backend hook or the community dtype path
           works on Iluvatar.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from vllm_iluvatar.ilu_patch import general_patch as p

# Baseline: main_4875b4456b5
CHUNK_SCALED_DOT_KKT_FWD_KERNEL = '''
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
'''


def register() -> None:
    """Register the Iluvatar FLA chunk_scaled_dot_kkt override."""
    p.register_once(
        __name__,
        _apply,
        detail="FLA chunk_scaled_dot_kkt Triton dtype override",
    )


def _unwrap_triton_kernel(kernel) -> Callable:
    target = kernel
    seen: set[int] = set()
    while not inspect.isfunction(target):
        obj_id = id(target)
        if obj_id in seen:
            break
        seen.add(obj_id)
        for attr in ("fn", "base_fn", "func", "function"):
            inner = getattr(target, attr, None)
            if inner is not None:
                target = inner
                break
        else:
            break
    if not inspect.isfunction(target):
        raise TypeError(
            "Unable to unwrap Triton kernel to a Python function: "
            f"{type(kernel).__name__}"
        )
    return target


def _kernel_source(kernel) -> str:
    target = _unwrap_triton_kernel(kernel)
    return textwrap.dedent(inspect.getsource(target)).strip()


def _check_kernel_signature(community_kernel, ilu_kernel) -> None:
    community_sig = inspect.signature(_unwrap_triton_kernel(community_kernel))
    ilu_sig = inspect.signature(_unwrap_triton_kernel(ilu_kernel))
    if community_sig != ilu_sig:
        raise RuntimeError(
            "ILU patch: chunk_scaled_dot_kkt_fwd_kernel signature mismatch: "
            f"community {community_sig} != Iluvatar {ilu_sig}"
        )


def _apply() -> None:
    import vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt as _m

    import vllm_iluvatar.model_executor.layers.fla.ops.chunk_scaled_dot_kkt as impl

    _check_kernel_signature(
        _m.chunk_scaled_dot_kkt_fwd_kernel,
        impl.chunk_scaled_dot_kkt_fwd_kernel,
    )
    p.check_fingerprint(
        "chunk_scaled_dot_kkt_fwd_kernel",
        _kernel_source(_m.chunk_scaled_dot_kkt_fwd_kernel),
        CHUNK_SCALED_DOT_KKT_FWD_KERNEL,
        resync_hint=(
            "Re-sync vllm_iluvatar/model_executor/layers/fla/ops/"
            "chunk_scaled_dot_kkt.py and the baseline in "
            "patch_chunk_scaled_dot_kkt.py to the new community kernel."
        ),
    )
    # [ILU override] chunk_scaled_dot_kkt_fwd_kernel - delta: use copied Triton
    # kernel with Iluvatar dtype conversion inside tl.dot. The community
    # chunk_scaled_dot_kkt_fwd wrapper resolves this global dynamically.
    p.rebind_module_func(
        _m,
        "chunk_scaled_dot_kkt_fwd_kernel",
        impl.chunk_scaled_dot_kkt_fwd_kernel,
    )
