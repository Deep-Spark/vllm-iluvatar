# SPDX-License-Identifier: Apache-2.0
"""Gated RMSNorm suite (community FLA Triton vs ixaite FlyDSL)."""

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
class RMSNormGatedCase:
    case_id: str
    t: int
    heads: int
    dim: int
    tags: tuple[str, ...] = ("norm", "gdn")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        shape = (self.t, self.heads, self.dim)
        out = torch.empty(shape, device=device, dtype=dtype)
        x = torch.randn(shape, device=device, dtype=dtype)
        z = torch.randn(shape, device=device, dtype=dtype)
        weight = torch.randn(self.dim, device=device, dtype=dtype)
        return Workload(args=(out, x, z, weight))

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args, 1e-6, True, "silu")

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0].float(),
            baseline_wl.args[0].float(),
            rtol=5e-2,
            atol=5e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


class RMSNormGatedSuite:
    name = "rms_norm_gated"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar import _custom_ops as ilu_ops

        return Providers(
            baseline_name="community FLA Triton rmsnorm_fn",
            baseline=ilu_ops.rms_norm_gated,
            flydsl=ixaite.rms_norm_gated,
            flydsl_name="ixaite.rms_norm_gated",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[RMSNormGatedCase]:
        spec = l2_spec(model, self.name)
        cases = [
            RMSNormGatedCase(
                f"gdn_output_norm@{model};T={t}",
                t,
                heads=int(spec["heads"]),
                dim=int(spec["dim"]),
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(RMSNormGatedSuite())
