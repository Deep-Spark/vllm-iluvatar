# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 paged K-cache gather (ixformer vs ixaite)."""

from __future__ import annotations

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

HEAD_DIM = 512
BLOCK_SIZE = 64


@dataclass(frozen=True)
class Ds4GatherCase:
    case_id: str
    t: int
    tags: tuple[str, ...] = ("attention", "ds4", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        seq = max(self.t, 64)
        max_blocks = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
        k_cache = torch.randn(
            (max_blocks + 1, BLOCK_SIZE, HEAD_DIM), device=device, dtype=torch.bfloat16
        )
        out = torch.empty((1, seq, HEAD_DIM), device=device, dtype=torch.bfloat16)
        sl = torch.tensor([seq], device=device, dtype=torch.int32)
        gl = torch.tensor([min(self.t, seq)], device=device, dtype=torch.int32)
        block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).view(1, -1)
        return Workload(
            args=(out, k_cache, sl, gl, block_table, BLOCK_SIZE, 0),
            meta={"out0": out.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["out0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0].float(), baseline_wl.args[0].float(), rtol=0, atol=0
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        out = wl.args[0]
        return 2 * out.numel() * out.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4GatherKCacheSuite:
    name = "ds4_gather_k_cache"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_gather_k_cache import ds4_gather_k_cache

        return Providers(
            baseline_name="IXF.ds4_gather_k_cache",
            baseline=ixf_ops.ds4_gather_k_cache,
            flydsl=ds4_gather_k_cache,
            flydsl_name="ixaite.ds4_gather_k_cache",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4GatherCase]:
        cases = [Ds4GatherCase(f"ds4_gather_k_cache@{model};tok={t:04d}", t) for t in RUNTIME_T]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4GatherKCacheSuite())
