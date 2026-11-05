# SPDX-License-Identifier: Apache-2.0
"""Scaled INT8 GEMM suite (IXF.w8a8 vs ixaite.w8a8)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import (
    RUNTIME_T,
    filter_cases,
    initialize_ixaite,
)


@dataclass(frozen=True)
class W8A8Case:
    case_id: str
    t: int
    k: int
    n: int
    tags: tuple[str, ...] = ("gemm", "int8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("w8a8 output dtype must be float16 or bfloat16")
        x_q = torch.randint(-8, 9, (self.t, self.k), device=device, dtype=torch.int8)
        w_q = torch.randint(-8, 9, (self.k, self.n), device=device, dtype=torch.int8)
        x_s = torch.rand(self.t, device=device, dtype=torch.float32) * 0.01
        w_s = torch.rand(self.n, device=device, dtype=torch.float32) * 0.01
        output = torch.empty(self.t, self.n, device=device, dtype=dtype)
        return Workload(
            args=(x_q, w_q, x_s, w_s, output),
            meta={
                "M": self.t,
                "N": self.n,
                "K": self.k,
                "flops": 2 * self.t * self.n * self.k,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        x_q, w_q, x_s, w_s, output = wl.args
        fn(x_q, w_q, x_s, w_s, None, output=output, format="NN", out_dtype=output.dtype)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[-1], baseline_wl.args[-1], rtol=2e-2, atol=2e-2
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        return float(wl.meta["flops"]) / (median_ms * 1e-3) / 1e12, "TOPS"


class W8A8Suite:
    name = "w8a8"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.w8a8",
            baseline=ixf_ops.w8a8,
            flydsl=ixaite.w8a8,
            flydsl_name="ixaite.w8a8",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[W8A8Case]:
        specs = l2_spec(model, self.name)
        cases = [
            W8A8Case(f"{slug}@{model};T={t}", t, int(spec["K"]), int(spec["N"]))
            for slug, spec in specs.items()
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(W8A8Suite())
