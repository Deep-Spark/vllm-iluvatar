# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar FLA chunk wrapper and KKT kernel."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from conftest import requires_cuda

from vllm_iluvatar.third_party.flash_linear_attention.ops import chunk as ilu_chunk

torch = importlib.import_module("torch")


def test_chunk_gated_delta_rule_fwd_uses_ilu_scaled_dot_kkt(monkeypatch):
    calls = []
    g_cumsum = object()
    A = SimpleNamespace(name="A")
    A_solved = SimpleNamespace(name="A_solved")
    w = object()
    u = object()
    h = object()
    v_new = object()
    final_state = object()
    o = object()

    def fake_cumsum(*args, **kwargs):
        calls.append(("cumsum", args, kwargs))
        return g_cumsum

    def fake_kkt(**kwargs):
        calls.append(("kkt", kwargs))
        return A

    def fake_solve_tril(**kwargs):
        calls.append(("solve", kwargs))
        return A_solved

    def fake_recompute(**kwargs):
        calls.append(("recompute", kwargs))
        return w, u

    def fake_fwd_h(**kwargs):
        calls.append(("fwd_h", kwargs))
        return h, v_new, final_state

    def fake_fwd_o(**kwargs):
        calls.append(("fwd_o", kwargs))
        return o

    monkeypatch.setattr(ilu_chunk, "chunk_local_cumsum", fake_cumsum)
    monkeypatch.setattr(ilu_chunk, "chunk_scaled_dot_kkt_fwd", fake_kkt)
    monkeypatch.setattr(ilu_chunk, "solve_tril", fake_solve_tril)
    monkeypatch.setattr(ilu_chunk, "recompute_w_u_fwd", fake_recompute)
    monkeypatch.setattr(ilu_chunk, "chunk_gated_delta_rule_fwd_h", fake_fwd_h)
    monkeypatch.setattr(ilu_chunk, "chunk_fwd_o", fake_fwd_o)
    monkeypatch.setattr(ilu_chunk, "SUPPRESS_LEVEL", 0)

    q = SimpleNamespace(dtype="bf16")
    k = SimpleNamespace(dtype="bf16")
    v = SimpleNamespace(dtype="bf16")
    g = object()
    beta = object()

    assert ilu_chunk.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=0.25,
        initial_state="state",
        output_final_state=True,
        cu_seqlens="cu",
        chunk_indices="indices",
        chunk_offsets="offsets",
        core_attn_out="out",
    ) == (g_cumsum, o, A_solved, final_state, None, None, None)

    assert calls[1] == (
        "kkt",
        {
            "k": k,
            "beta": beta,
            "g": g_cumsum,
            "cu_seqlens": "cu",
            "chunk_indices": "indices",
            "output_dtype": ilu_chunk.torch.float32,
        },
    )


def _chunk_scaled_dot_kkt_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    out = torch.zeros(B, T, H, chunk_size, device=k.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            hg = h // (H // Hg)
            for chunk_start in range(0, T, chunk_size):
                chunk_end = min(chunk_start + chunk_size, T)
                for row in range(chunk_start, chunk_end):
                    local_row = row - chunk_start
                    for col in range(local_row):
                        col_abs = chunk_start + col
                        k_row = (k[b, row, hg].float() * beta[b, row, h].float()).to(
                            k.dtype
                        )
                        k_col = k[b, col_abs, hg].float()
                        gate = torch.exp(
                            g[b, row, h].float() - g[b, col_abs, h].float()
                        )
                        out[b, row, h, col] = (k_row.float() * k_col).sum() * gate
    return out


@requires_cuda
def test_chunk_scaled_dot_kkt_matches_torch_reference(cuda_device):
    chunk_size = 64
    seq_len = 64
    k = torch.full(
        (1, seq_len, 2, 64),
        1.0,
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    beta = torch.full((1, seq_len, 2), 1.003, device=cuda_device, dtype=torch.float32)
    g = torch.zeros(1, seq_len, 2, device=cuda_device, dtype=torch.float32)
    expected = _chunk_scaled_dot_kkt_reference(k, beta, g, chunk_size)

    actual = ilu_chunk.chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
