# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 C128A dense compressed prefill attention (ixformer vs ixaite)."""

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
CR = 128
C_BLOCK = 16


def _make_tensors(device, q_len: int, seq_len: int):
    c_seq = seq_len // CR
    n_blocks = max(1, (max(c_seq, 1) + C_BLOCK - 1) // C_BLOCK)
    q = torch.randn((q_len, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    kv = torch.randn((n_blocks, C_BLOCK, HEAD_DIM), device=device, dtype=torch.bfloat16)
    table = torch.arange(n_blocks, device=device, dtype=torch.int32).view(1, -1)
    cu_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    return q, kv, table, cu_q, cu_k


@dataclass(frozen=True)
class Ds4DenseCompressedCase:
    case_id: str
    q_len: int
    seq_len: int
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "c128a")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"C128A models the BF16 serving path, got {dtype}")
        q, kv, table, cu_q, cu_k = _make_tensors(device, self.q_len, self.seq_len)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        return Workload(
            args=(q, kv, table, cu_q, cu_k, CR, sm_scale),
            meta={"T": self.q_len, "seq": self.seq_len},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        out, lse = fn(*wl.args)
        wl.outputs["out"] = out.clone()
        wl.outputs["lse"] = lse.clone()

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=3.2e-2,
            atol=3.2e-2,
        )
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
        c_seq = max(self.seq_len // CR, 1)
        flops = 4.0 * q.shape[0] * q.shape[1] * c_seq * q.shape[2]
        return flops / (median_ms * 1e-3) / 1e12, "TFLOPS"


class Ds4DenseCompressedAttnFwdSuite:
    name = "ds4_dense_compressed_attn_fwd"

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
        from ixaite.ops.ds4_dense_compressed_attn_fwd import (
            ds4_dense_compressed_attn_fwd,
        )

        return Providers(
            baseline_name="IXF.ds4_dense_compressed_attn_fwd",
            baseline=ixf_ops.ds4_dense_compressed_attn_fwd,
            flydsl=ds4_dense_compressed_attn_fwd,
            flydsl_name="ixaite.ds4_dense_compressed_attn_fwd",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4DenseCompressedCase]:
        del model
        cases = [
            Ds4DenseCompressedCase(f"ds4_dense_compressed_attn_fwd;tok={t:04d};seq={s:04d}", t, s)
            for t, s in ((8, 128), (128, 128))
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4DenseCompressedAttnFwdSuite())
