# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 fused indexer Q RoPE (ixformer vs ixaite)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import (
    RUNTIME_T,
    filter_cases,
    initialize_ixaite,
)

HEADS = 64
HEAD_DIM = 128
ROPE_DIM = 64


@dataclass(frozen=True)
class Ds4IndexerQRopeCase:
    case_id: str
    t: int
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    rotary_dim: int = ROPE_DIM
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "indexer")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        q = torch.randn((self.t, self.heads, self.head_dim), device=device, dtype=dtype)
        weights = torch.randn((self.t, self.heads), device=device, dtype=dtype)
        weights_out = torch.empty_like(weights)
        positions = torch.arange(self.t, device=device, dtype=torch.int64)
        cos_sin = torch.randn((self.t + 8, self.rotary_dim), device=device, dtype=torch.float32)
        sm = self.head_dim**-0.5
        hs = self.heads**-0.5
        return Workload(
            args=(positions, q, cos_sin, weights, weights_out, sm, hs, self.rotary_dim, False),
            meta={"q0": q.clone(), "w0": weights_out.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[1].copy_(wl.meta["q0"])
        wl.args[4].copy_(wl.meta["w0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[1].float(),
            baseline_wl.args[1].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.args[4].float(),
            baseline_wl.args[4].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[1]
        return 2 * q.numel() * q.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4FusedIndexerQRopeSuite:
    name = "ds4_fused_indexer_q_rope"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        op = importlib.import_module("ixaite.ops.ds4_fused_indexer_q_rope")
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_fused_indexer_q_rope",
            baseline=ixf_ops.ds4_fused_indexer_q_rope,
            flydsl=op.ds4_fused_indexer_q_rope,
            flydsl_name="ixaite.ds4_fused_indexer_q_rope",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4IndexerQRopeCase]:
        cases = [
            Ds4IndexerQRopeCase(f"ds4_fused_indexer_q_rope@{model};tok={t:04d}", t)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4FusedIndexerQRopeSuite())
