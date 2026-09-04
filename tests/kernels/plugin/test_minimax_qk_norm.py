# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of MiniMax QK-Norm plugin ops and forward_qkv.

Coverage:
  A. Bare ``torch.ops._C.minimax_qk_var`` vs torch mean-of-squares.
  B. Bare ``torch.ops._C.minimax_qk_rms_apply`` vs torch normalize
     (tp_world in {1, 2, 4}).
  C. Combination: var -> simulate all-reduce (x tp_world) -> apply, vs
     ``forward_qkv_torch`` semantics with the same effective variance.
  D. End-to-end TP=1: ``forward_qkv`` (ixformer) vs ``forward_qkv_torch``.

Shapes include strided qkv (narrowed from a wider buffer), non-power-of-two
sizes, empty tokens, and both bf16 / fp16.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import ixformer  # noqa: F401
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

from vllm_iluvatar.model_executor.layers.minimax_rms_norm import rms_norm_tp

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DTYPES = [torch.bfloat16, torch.float16]
# (num_tokens, q_size, kv_size)
SHAPES = [
    (1, 128, 128),
    (7, 256, 128),
    (17, 1024, 256),
    (3, 6144, 512),  # large MiniMax-like q shard
    (0, 128, 64),  # empty
]
TP_WORLDS = [1, 2, 4]
EPS = 1e-6


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 2e-2, 2e-2  # bfloat16


def _make_qkv(
    num_tokens: int,
    q_size: int,
    kv_size: int,
    dtype: torch.dtype,
    *,
    strided: bool,
) -> torch.Tensor:
    hidden = q_size + 2 * kv_size
    if strided:
        # Wider buffer so row_stride > hidden; view is contiguous in last dim.
        storage = torch.empty(
            (num_tokens, hidden + 32), dtype=dtype, device="cuda"
        ).normal_()
        qkv = storage[:, :hidden]
    else:
        qkv = torch.empty((num_tokens, hidden), dtype=dtype, device="cuda").normal_()
    if num_tokens > 0:
        assert qkv.stride(1) == 1
        if strided:
            assert qkv.stride(0) > hidden
    return qkv


def _make_weights(
    q_size: int, kv_size: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    q_w = torch.empty(q_size, dtype=dtype, device="cuda").normal_()
    k_w = torch.empty(kv_size, dtype=dtype, device="cuda").normal_()
    return q_w, k_w


def _ref_var(qkv: torch.Tensor, q_size: int, kv_size: int) -> torch.Tensor:
    q = qkv[:, :q_size].float()
    k = qkv[:, q_size : q_size + kv_size].float()
    q_var = q.pow(2).mean(dim=-1)
    k_var = k.pow(2).mean(dim=-1)
    return torch.stack([q_var, k_var], dim=-1)


def _ref_apply(
    qkv: torch.Tensor,
    qk_var: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = qkv[:, :q_size].float()
    k = qkv[:, q_size : q_size + kv_size].float()
    q_inv = torch.rsqrt(qk_var[:, 0:1] / tp_world + eps)
    k_inv = torch.rsqrt(qk_var[:, 1:2] / tp_world + eps)
    q_out = (q * q_inv * q_weight.float()).to(qkv.dtype)
    k_out = (k * k_inv * k_weight.float()).to(qkv.dtype)
    return q_out, k_out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens,q_size,kv_size", SHAPES)
@pytest.mark.parametrize("strided", [False, True])
def test_minimax_qk_var(
    dtype: torch.dtype,
    num_tokens: int,
    q_size: int,
    kv_size: int,
    strided: bool,
) -> None:
    qkv = _make_qkv(num_tokens, q_size, kv_size, dtype, strided=strided)
    out = torch.ops._C.minimax_qk_var(qkv, q_size, kv_size)
    assert out.shape == (num_tokens, 2)
    assert out.dtype == torch.float32
    if num_tokens == 0:
        return
    ref = _ref_var(qkv, q_size, kv_size)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens,q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tp_world", TP_WORLDS)
@pytest.mark.parametrize("strided", [False, True])
def test_minimax_qk_rms_apply(
    dtype: torch.dtype,
    num_tokens: int,
    q_size: int,
    kv_size: int,
    tp_world: int,
    strided: bool,
) -> None:
    qkv = _make_qkv(num_tokens, q_size, kv_size, dtype, strided=strided)
    q_w, k_w = _make_weights(q_size, kv_size, dtype)
    # Synthetic all-reduced variance (sum of per-shard means).
    qk_var = torch.rand((num_tokens, 2), dtype=torch.float32, device="cuda").abs() + 0.1
    q_out, k_out = torch.ops._C.minimax_qk_rms_apply(
        qkv, qk_var, q_w, k_w, q_size, kv_size, tp_world, EPS
    )
    assert q_out.shape == (num_tokens, q_size)
    assert k_out.shape == (num_tokens, kv_size)
    assert q_out.dtype == dtype and k_out.dtype == dtype
    assert q_out.is_contiguous() and k_out.is_contiguous()
    if num_tokens == 0:
        return
    ref_q, ref_k = _ref_apply(qkv, qk_var, q_w, k_w, q_size, kv_size, tp_world, EPS)
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(q_out, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens,q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tp_world", TP_WORLDS)
def test_minimax_qk_norm_combo(
    dtype: torch.dtype,
    num_tokens: int,
    q_size: int,
    kv_size: int,
    tp_world: int,
) -> None:
    """var -> *tp_world (identical-shard all-reduce) -> apply."""
    qkv = _make_qkv(num_tokens, q_size, kv_size, dtype, strided=False)
    q_w, k_w = _make_weights(q_size, kv_size, dtype)
    local_var = torch.ops._C.minimax_qk_var(qkv, q_size, kv_size)
    # Simulate all-reduce when every shard holds the same activation shard
    # content: sum = local * tp_world. Apply folds /tp_world back in.
    ar_var = local_var * float(tp_world)
    q_out, k_out = torch.ops._C.minimax_qk_rms_apply(
        qkv, ar_var, q_w, k_w, q_size, kv_size, tp_world, EPS
    )
    if num_tokens == 0:
        return

    # Reference: normalize with the local mean (equivalent after /tp_world).
    q = qkv[:, :q_size].float()
    k = qkv[:, q_size : q_size + kv_size].float()
    q_inv = torch.rsqrt(local_var[:, 0:1] + EPS)
    k_inv = torch.rsqrt(local_var[:, 1:2] + EPS)
    ref_q = (q * q_inv * q_w.float()).to(dtype)
    ref_k = (k * k_inv * k_w.float()).to(dtype)
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(q_out, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "num_tokens,q_size,kv_size",
    [(1, 128, 64), (8, 256, 128), (4, 1024, 256)],
)
def test_forward_qkv_tp1_vs_torch(
    dtype: torch.dtype,
    num_tokens: int,
    q_size: int,
    kv_size: int,
) -> None:
    qkv = _make_qkv(num_tokens, q_size, kv_size, dtype, strided=True)
    q_w, k_w = _make_weights(q_size, kv_size, dtype)
    q_norm = SimpleNamespace(weight=q_w, variance_epsilon=EPS)
    k_norm = SimpleNamespace(weight=k_w, variance_epsilon=EPS)

    with mock.patch.object(
        rms_norm_tp, "get_tensor_model_parallel_world_size", return_value=1
    ):
        ref_q, ref_k, ref_v = rms_norm_tp.forward_qkv_torch(
            q_norm, k_norm, qkv, q_size, kv_size
        )
        out_q, out_k, out_v = rms_norm_tp.forward_qkv(
            q_norm, k_norm, qkv, q_size, kv_size
        )

    assert out_v.data_ptr() == ref_v.data_ptr()
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)
