# SPDX-License-Identifier: Apache-2.0
"""Gemma RMSNorm suite (ixformer vs ixaite FlyDSL)."""

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
class GemmaRMSNormCase:
    case_id: str
    t: int
    hidden: int
    has_residual: bool = True
    tags: tuple[str, ...] = ("norm", "gemma")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = torch.randn((self.t, self.hidden), device=device, dtype=dtype)
        residual = (
            torch.randn((self.t, self.hidden), device=device, dtype=dtype)
            if self.has_residual
            else None
        )
        weight = torch.randn((self.hidden,), device=device, dtype=dtype)
        return Workload(
            args=(x, weight, 1e-6, residual),
            meta={"T": self.t, "H": self.hidden},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        result = fn(*wl.args)
        out, residual_out = result if self.has_residual else (result, None)
        wl.outputs["out"] = out  # type: ignore[assignment]
        wl.outputs["residual_out"] = residual_out  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        if self.has_residual:
            torch.testing.assert_close(
                flydsl_wl.outputs["residual_out"].float(),
                baseline_wl.outputs["residual_out"].float(),
                rtol=2e-2,
                atol=2e-2,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        # x (+ residual) reads, out (+ residual_out) writes, plus weight read.
        elem = wl.args[0].element_size()
        rows = 4 * self.t * self.hidden if self.has_residual else 2 * self.t * self.hidden
        num_bytes = (rows + self.hidden) * elem
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class GemmaRMSNormSuite:
    name = "gemma_rms_norm"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar import _custom_ops as ilu_ops

        return Providers(
            baseline_name="IXF.gemma_rms_norm",
            baseline=ilu_ops.gemma_rms_norm,
            flydsl=ixaite.gemma_rms_norm,
            flydsl_name="ixaite.gemma_rms_norm",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[GemmaRMSNormCase]:
        spec = l2_spec(model, self.name)
        hidden = int(spec["hidden_size"])
        cases = [
            GemmaRMSNormCase(f"decoder_residual_norm@{model};T={t}", t, hidden)
            for t in RUNTIME_T
        ] + [
            GemmaRMSNormCase(
                f"final_norm@{model};T={t}", t, hidden, has_residual=False
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(GemmaRMSNormSuite())
