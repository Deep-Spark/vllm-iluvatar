# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 global topk index map (ixformer vs ixaite)."""

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


@dataclass(frozen=True)
class Ds4GlobalTopkCase:
    case_id: str
    t: int
    topk: int = 32
    num_reqs: int = 4
    block_size: int = 64
    tags: tuple[str, ...] = ("attention", "ds4", "index")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        topk_indices = torch.randint(
            -1, self.block_size * 8, (self.t, self.topk), device=device, dtype=torch.int32
        )
        token_to_req = torch.arange(self.t, device=device, dtype=torch.int32) % self.num_reqs
        block_table = torch.arange(
            self.num_reqs * 8, device=device, dtype=torch.int32
        ).reshape(self.num_reqs, 8)
        is_valid = torch.ones((self.t,), device=device, dtype=torch.bool)
        return Workload(
            args=(topk_indices, token_to_req, block_table, self.block_size, is_valid),
            meta={"T": self.t},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        g, l = fn(*wl.args)
        wl.outputs["g"] = g
        wl.outputs["l"] = l

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(flydsl_wl.outputs["g"], baseline_wl.outputs["g"], rtol=0, atol=0)
        torch.testing.assert_close(flydsl_wl.outputs["l"], baseline_wl.outputs["l"], rtol=0, atol=0)

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        return None


class Ds4ComputeGlobalTopkSuite:
    name = "ds4_compute_global_topk"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_compute_global_topk import (
            ds4_compute_global_topk_indices_and_lens,
        )

        return Providers(
            baseline_name="IXF.ds4_compute_global_topk_indices_and_lens",
            baseline=ixf_ops.ds4_compute_global_topk_indices_and_lens,
            flydsl=ds4_compute_global_topk_indices_and_lens,
            flydsl_name="ixaite.ds4_compute_global_topk_indices_and_lens",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4GlobalTopkCase]:
        cases = [
            Ds4GlobalTopkCase(f"ds4_compute_global_topk@{model};tok={t:04d}", t)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4ComputeGlobalTopkSuite())
