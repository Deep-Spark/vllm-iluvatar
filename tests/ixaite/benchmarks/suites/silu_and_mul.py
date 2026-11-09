# SPDX-License-Identifier: Apache-2.0
"""silu_and_mul cross-provider suite (first harness validation path)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import RUNTIME_T


@dataclass(frozen=True)
class SiluAndMulCase:
    case_id: str
    rows: int
    d: int
    tags: tuple[str, ...] = ("activation",)

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = torch.randn(self.rows, 2 * self.d, device=device, dtype=dtype)
        out = torch.empty(self.rows, self.d, device=device, dtype=dtype)
        return Workload(
            args=(out, x),
            meta={"rows": self.rows, "D": self.d},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        out, x = wl.args
        fn(out, x)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0],
            baseline_wl.args[0],
            rtol=1e-2,
            atol=1e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        del median_ms, wl
        return None


class SiluAndMulSuite:
    name = "silu_and_mul"

    def resolve_providers(self) -> Providers:
        import ixaite

        from vllm_iluvatar import _custom_ops as ilu_ops
        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_torch_op
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)
        fly, reason = load_ixaite_torch_op("silu_and_mul")
        if fly is None:
            raise RuntimeError(reason)
        return Providers(
            baseline_name="IXF silu",
            baseline=ilu_ops.silu_and_mul,
            flydsl=fly,
            flydsl_name="ixaite.silu_and_mul",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[SiluAndMulCase]:
        d = int(l2_spec(model, self.name)["d"])
        cases = [
            SiluAndMulCase(
                case_id=f"silu_and_mul@{model};rows={rows};D={d}",
                rows=rows,
                d=d,
            )
            for rows in RUNTIME_T
        ]
        if case_filter:
            cases = [case for case in cases if case_filter in case.case_id]
        return cases

    def gate_min_speedup(self) -> float:
        return 1.0


register(SiluAndMulSuite())
