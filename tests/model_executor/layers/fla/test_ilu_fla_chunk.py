# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar FLA chunk wrapper."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.model_executor.layers.fla.ops import chunk as ilu_chunk


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
