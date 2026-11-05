# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 C4A paged sparse prefill attention (ixformer vs ixaite)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

HEAD_DIM = 512
HEADS = 8
CR = 4
BLOCK = 64
TOPK = 32


def _make_tensors(device, q_len: int, seq_len: int):
    c_seq = seq_len // CR
    n_blocks = max(1, (max(c_seq, 1) + BLOCK - 1) // BLOCK)
    q = torch.randn((q_len, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    kv = torch.randn((n_blocks, BLOCK, HEAD_DIM), device=device, dtype=torch.bfloat16)
    table = torch.arange(n_blocks, device=device, dtype=torch.int32).view(1, -1)
    idx = torch.full((q_len, TOPK), -1, device=device, dtype=torch.int32)
    start = seq_len - q_len
    for t in range(q_len):
        vis = min((start + t + 1) // CR, TOPK)
        if vis > 0 and c_seq > 0:
            idx[t, :vis] = torch.randperm(c_seq, device=device)[:vis]
    cu_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    return q, kv, table, idx, cu_q, seq_lens


@dataclass(frozen=True)
class Ds4PagedSparseCase:
    case_id: str
    q_len: int
    seq_len: int
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "c4a")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"C4A models the BF16 serving path, got {dtype}")
        q, kv, table, idx, cu_q, seq_lens = _make_tensors(device, self.q_len, self.seq_len)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        return Workload(
            args=(q, kv, table, idx, cu_q, seq_lens, CR, BLOCK, sm_scale),
            kwargs={"return_lse": True},
            meta={"T": self.q_len, "seq": self.seq_len},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        out, lse = fn(*wl.args, **wl.kwargs)
        wl.outputs["out"] = out.clone()
        wl.outputs["lse"] = lse.clone() if lse is not None else None

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=3.2e-2,
            atol=3.2e-2,
        )
        if flydsl_wl.outputs["lse"] is not None and baseline_wl.outputs["lse"] is not None:
            got = flydsl_wl.outputs["lse"]
            ref = baseline_wl.outputs["lse"]
            both = torch.isneginf(got) & torch.isneginf(ref)
            torch.testing.assert_close(
                torch.where(both, torch.zeros_like(got), got),
                torch.where(both, torch.zeros_like(ref), ref),
                rtol=3.2e-2,
                atol=3.2e-2,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        flops = 4.0 * q.shape[0] * q.shape[1] * TOPK * q.shape[2]
        return flops / (median_ms * 1e-3) / 1e12, "TFLOPS"


class Ds4PagedSparseAttnFwdSuite:
    name = "ds4_paged_sparse_attn_fwd"

    def resolve_providers(self) -> Providers:
        try:
            initialize_ixaite()
        except Exception:
            import os

            import ixaite.runtime as rt

            os.environ.setdefault("ARCH", "ivcore11")
            os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
            os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
            rt._INITIALIZED_ARCH = "ivcore11"
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_paged_sparse_attn_fwd import ds4_paged_sparse_attn_fwd

        return Providers(
            baseline_name="IXF.ds4_paged_sparse_attn_fwd",
            baseline=ixf_ops.ds4_paged_sparse_attn_fwd,
            flydsl=ds4_paged_sparse_attn_fwd,
            flydsl_name="ixaite.ds4_paged_sparse_attn_fwd",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4PagedSparseCase]:
        del model
        cases = [
            Ds4PagedSparseCase(f"ds4_paged_sparse_attn_fwd;tok={t:04d};seq={s:04d}", t, s)
            for t, s in ((8, 128), (128, 128))
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4PagedSparseAttnFwdSuite())
