# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of torch.ops._C.apply_rotary_emb (self-contained kernel).

This plugin-internal op backs the ``ApplyRotaryEmb`` OOT override
(vllm_iluvatar/model_executor/layers/rotary_embedding/common.py), replacing the
community flash-attn rotary path (``vllm.vllm_flash_attn.layers.rotary``) that is
absent on Iluvatar. Two layers of coverage:

A. Bare op level -- calls ``torch.ops._C.apply_rotary_emb`` directly against a
   pure-torch reference (mirrors community forward_static / flash-attn
   apply_rotary_emb_torch): Neox / GPT-J styles, full and partial rotary_dim,
   3D / 4D (batched) inputs, fp16 / bf16 / fp32, and a cos/sin dtype that differs
   from x (fp32 tables + bf16/fp16 x).

B. Module level -- constructs the community ``ApplyRotaryEmb`` CustomOp (swapped
   to ``IluApplyRotaryEmb`` via register_oot) and checks ``forward_cuda`` (OOT ->
   our _C kernel) against ``forward_native``, including enable_fp32_compute.

The native extension is a mandatory dep (importing it registers the op); only the
actual GPU device is gated via skip.
"""

from __future__ import annotations

import pytest
import torch

# Native extension is a MANDATORY dep (importing it registers
# torch.ops._C.apply_rotary_emb): a missing build is an environment error.
import vllm_iluvatar._C  # noqa: F401

# Importing the OOT module registers IluApplyRotaryEmb for the module-level test.
from vllm_iluvatar.model_executor.layers.rotary_embedding.common import (
    IluApplyRotaryEmb,
)

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DTYPES = [torch.bfloat16, torch.float, torch.half]
INTERLEAVED = [False, True]  # False = Neox, True = GPT-J


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float:
        return 1e-4, 1e-4
    if dtype == torch.half:
        return 1e-2, 1e-2
    return 2e-2, 2e-2  # bfloat16


# --- reference (community forward_static / flash-attn apply_rotary_emb_torch) --
def _ref_apply_rotary_emb(
    x: torch.Tensor,  # [..., seqlen, nheads, headdim]
    cos: torch.Tensor,  # [seqlen, rot_dim / 2]
    sin: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    x_rot = x[..., :ro_dim].float()
    x_pass = x[..., ro_dim:]
    # Broadcast cos/sin across the head dim: [seqlen, 1, rot_dim / 2].
    cos_ = cos.unsqueeze(-2).float()
    sin_ = sin.unsqueeze(-2).float()

    if not interleaved:
        x1, x2 = x_rot.chunk(2, dim=-1)
        o1 = x1 * cos_ - x2 * sin_
        o2 = x2 * cos_ + x1 * sin_
        out_rot = torch.cat((o1, o2), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        o1 = x1 * cos_ - x2 * sin_
        o2 = x2 * cos_ + x1 * sin_
        out_rot = torch.stack((o1, o2), dim=-1).flatten(-2)

    return torch.cat((out_rot.to(x.dtype), x_pass), dim=-1)


def _make_cos_sin(
    seqlen: int, rot_dim: int, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    embed = rot_dim // 2
    angles = torch.randn(seqlen, embed, dtype=torch.float32, device=device) * 3.0
    return angles.cos().to(dtype), angles.sin().to(dtype)


# --- layer A: bare op --------------------------------------------------------
@pytest.mark.parametrize("interleaved", INTERLEAVED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128, 256, 120])
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
@pytest.mark.parametrize("num_heads", [1, 8, 17])
@pytest.mark.parametrize("batched", [False, True])
@torch.inference_mode()
def test_apply_rotary_emb_op(
    interleaved: bool,
    dtype: torch.dtype,
    head_dim: int,
    rotary_ratio: float,
    num_heads: int,
    batched: bool,
) -> None:
    torch.manual_seed(0)
    device = "cuda:0"
    seqlen = 137
    batch = 3 if batched else 1

    rot_dim = int(head_dim * rotary_ratio)
    rot_dim -= rot_dim % 2  # rot_dim must be even
    if rot_dim == 0:
        pytest.skip("degenerate rot_dim")

    shape = (batch, seqlen, num_heads, head_dim) if batched else (
        seqlen,
        num_heads,
        head_dim,
    )
    x = torch.randn(shape, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(seqlen, rot_dim, dtype, device)

    ref = _ref_apply_rotary_emb(x, cos, sin, interleaved)
    out = torch.ops._C.apply_rotary_emb(x, cos, sin, interleaved)

    assert out.shape == x.shape
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("interleaved", INTERLEAVED)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_apply_rotary_emb_op_fp32_cache(
    interleaved: bool, x_dtype: torch.dtype
) -> None:
    """fp32 cos/sin tables with bf16/fp16 x (enable_fp32_compute stores fp32)."""
    torch.manual_seed(0)
    device = "cuda:0"
    seqlen, num_heads, head_dim, rot_dim = 128, 8, 128, 128

    x = torch.randn(seqlen, num_heads, head_dim, dtype=x_dtype, device=device)
    cos, sin = _make_cos_sin(seqlen, rot_dim, torch.float32, device)

    ref = _ref_apply_rotary_emb(x, cos, sin, interleaved)
    out = torch.ops._C.apply_rotary_emb(x, cos, sin, interleaved)

    atol, rtol = _atol_rtol(x_dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_apply_rotary_emb_opcheck(dtype: torch.dtype) -> None:
    """torch.library.opcheck: schema / alias / fake consistency."""
    device = "cuda:0"
    seqlen, num_heads, head_dim, rot_dim = 32, 4, 64, 64
    x = torch.randn(seqlen, num_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(seqlen, rot_dim, dtype, device)
    torch.library.opcheck(torch.ops._C.apply_rotary_emb, (x, cos, sin, False))


# --- layer B: module level (OOT forward_cuda vs forward_native) --------------
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("enable_fp32_compute", [False, True])
@pytest.mark.parametrize("input_ndim", [3, 4])
@torch.inference_mode()
def test_apply_rotary_emb_module(
    default_vllm_config,
    is_neox_style: bool,
    dtype: torch.dtype,
    enable_fp32_compute: bool,
    input_ndim: int,
) -> None:
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    torch.manual_seed(0)
    device = "cuda:0"

    op = ApplyRotaryEmb(
        is_neox_style=is_neox_style, enable_fp32_compute=enable_fp32_compute
    )
    # The OOT swap must have taken effect (forward_cuda -> our _C kernel).
    assert isinstance(op, IluApplyRotaryEmb)

    seqlen, num_heads, head_dim = 96, 8, 128
    if input_ndim == 3:
        x = torch.randn(seqlen, num_heads, head_dim, dtype=dtype, device=device)
    else:
        x = torch.randn(2, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(seqlen, head_dim, dtype, "cuda:0")

    ref = op.forward_native(x.clone(), cos, sin)
    out = op.forward_cuda(x.clone(), cos, sin)

    assert out.shape == x.shape
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
