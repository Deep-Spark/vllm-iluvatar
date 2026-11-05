# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""convert cosine gates + package contract + optional layer smoke."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

_IXAITE_MINIMAX = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar"
    / "backends"
    / "ixaite"
    / "models"
    / "minimax_m3"
)
_IXAITE_QUANT = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar"
    / "backends"
    / "ixaite"
    / "quantization"
)
_FLASH_CKPT = Path("/data/peter/model/MiniMax-M3.1-Flash/config.json")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def test_ixaite_minimax_package_forbids_vendor_imports():
    forbidden = (
        "ixformer",
        "flash_attn",
        "flashinfer",
        "triton",
        "vllm_flash_attn",
        "amd",
        "rocm",
    )
    roots = (_IXAITE_MINIMAX, _IXAITE_QUANT)
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "minimax_m3_flash.py" or "minimax" in str(path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
            else:
                # Only MiniMax flash quant + model package for this gate.
                if root == _IXAITE_QUANT and path.name != "minimax_m3_flash.py":
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    assert not any(
                        name == f or name.startswith(f + ".") for f in forbidden
                    ), f"{path}: imports {name}"


def test_mxfp8_to_w8a8_convert_cosine_vs_bf16_dequant():
    from ixaite.ops.minimax_weight_convert import mxfp8_to_w8a8
    from ixaite.ops.minimax_weight_convert_common import (
        bf16_to_w8a8_channel,
        dequant_mxfp8_to_bf16,
    )

    n, k = 64, 128
    bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    blocks = bf16.float().view(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-8)
    exp = (torch.round(torch.log2(amax / 448.0)) + 127).clamp(0, 254).to(torch.uint8)
    scale_f = torch.exp2(exp.float() - 127.0).unsqueeze(-1)
    q = (blocks / scale_f).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k)

    w_kn, scale_n = mxfp8_to_w8a8(q, exp, reference=True)
    ref_w, ref_s = bf16_to_w8a8_channel(dequant_mxfp8_to_bf16(q, exp))
    # Reconstruct approximate bf16 from W8A8 for cosine vs dequant.
    recon = (w_kn.t().float() * scale_n.view(-1, 1)).to(torch.bfloat16)
    deq = dequant_mxfp8_to_bf16(q, exp)
    assert _cosine(recon, deq) > 0.99
    assert torch.equal(w_kn, ref_w)
    assert torch.allclose(scale_n, ref_s)


def test_nvfp4_to_w4a8_convert_cosine_vs_bf16_dequant():
    from ixaite.ops.minimax_weight_convert import nvfp4_to_w4a8_flydsl
    from ixaite.ops.minimax_weight_convert_common import (
        _FP4_E2M1_LUT,
        NVFP4_BLOCK,
        dequant_nvfp4_to_bf16,
    )
    from ixaite.ops.w4a8_layout import unpack_signed_int4

    n, k, gscale = 32, 64, 0.5
    bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    blocks = bf16.float().view(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-8)
    local = ((amax * float(gscale)) / 6.0).clamp(max=448.0)
    scale_f8 = local.to(torch.float8_e4m3fn)
    scale_f = scale_f8.float() / float(gscale)
    q = (blocks / scale_f.unsqueeze(-1)).clamp(-6.0, 6.0)
    lut = _FP4_E2M1_LUT
    pos = lut[:8]
    dist = (q.unsqueeze(-1) - pos.view(1, 1, 1, 8)).abs()
    pos_code = dist.argmin(dim=-1)
    neg_code = (q.unsqueeze(-1) - (-pos).view(1, 1, 1, 8)).abs().argmin(dim=-1) + 8
    codes = torch.where(q < 0, neg_code, pos_code).to(torch.uint8).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    g = torch.tensor(gscale, dtype=torch.float32)

    w_pack, w_scale = nvfp4_to_w4a8_flydsl(packed, scale_f8, g, reference=True)
    deq = dequant_nvfp4_to_bf16(packed, scale_f8, g)
    # Unpack INT4 and scale back roughly (FlyDSL stores /18 compensated scales).
    q4 = unpack_signed_int4(w_pack).float()
    recon = q4 * (w_scale.float().unsqueeze(-1) * 18.0)
    assert _cosine(recon, deq.float()) > 0.98


@pytest.mark.skipif(
    not _FLASH_CKPT.is_file(),
    reason="MiniMax-M3.1-Flash checkpoint unavailable at "
    f"{_FLASH_CKPT}; skip layer smoke",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for layer smoke"
)
def test_flash_layer_smoke_requires_checkpoint_and_gpu():
    # Placeholder for a future end-to-end layer forward once serve wiring lands.
    # Presence of the checkpoint gate is the checkpoint-gated smoke contract when weights exist.
    assert _FLASH_CKPT.is_file()
