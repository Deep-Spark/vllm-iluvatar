# SPDX-License-Identifier: Apache-2.0
"""ixformer gemm_bf16_bf16_fp32 vs ixaite (TN, bf16 -> fp32)."""

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


def _geom(model: str) -> tuple[int, int]:
    spec = l2_spec(model, "gemm_bf16_bf16_fp32")
    return int(spec["n"]), int(spec["k"])


@dataclass(frozen=True)
class GemmBf16Fp32Case:
    case_id: str
    t: int
    n: int
    k: int
    tags: tuple[str, ...] = ("ds4", "gemm", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        del dtype
        inp = torch.randn((self.t, self.k), device=device, dtype=torch.bfloat16)
        weight = torch.randn((self.n, self.k), device=device, dtype=torch.bfloat16)
        out = torch.empty((self.t, self.n), device=device, dtype=torch.float32)
        return Workload(args=(inp, weight, "TN", out), meta={"T": self.t})

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"],
            baseline_wl.outputs["out"],
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        inp, weight, _, _ = wl.args
        flops = 2 * inp.shape[0] * weight.shape[0] * inp.shape[1]
        return flops / (median_ms * 1e-3) / 1e12, "TFLOPS"


class GemmBf16Fp32Suite:
    name = "gemm_bf16_bf16_fp32"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.gemm_bf16_bf16_fp32",
            baseline=ixf_ops.gemm_bf16_bf16_fp32,
            flydsl=ixaite.gemm_bf16_bf16_fp32,
            flydsl_name="ixaite.gemm_bf16_bf16_fp32",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[GemmBf16Fp32Case]:
        n, k = _geom(model)
        cases = [
            GemmBf16Fp32Case(f"gemm_bf16_fp32@{model};tok={t:04d}", t, n, k) for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(GemmBf16Fp32Suite())
