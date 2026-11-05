# SPDX-License-Identifier: Apache-2.0
"""Plain RMSNorm suite (ixformer vs ixaite FlyDSL)."""

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

_HIDDEN = 4096


@dataclass(frozen=True)
class RmsNormCase:
    case_id: str
    t: int
    hidden: int = _HIDDEN
    tags: tuple[str, ...] = ("norm",)

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = torch.randn((self.t, self.hidden), device=device, dtype=dtype)
        weight = torch.randn((self.hidden,), device=device, dtype=dtype)
        out = torch.empty_like(x)
        return Workload(
            args=(out, x, weight, 1e-6),
            meta={"T": self.t, "H": self.hidden},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        _out, x, weight, eps = wl.args
        out_buf = torch.empty_like(x)
        fn(out_buf, x, weight, eps)
        wl.outputs["out"] = out_buf  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        elem = wl.args[1].element_size()
        num_bytes = (2 * self.t * self.hidden + self.hidden) * elem
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class RmsNormSuite:
    name = "rms_norm"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar import _custom_ops as ilu_ops

        return Providers(
            baseline_name="IXF.rms_norm",
            baseline=ilu_ops.rms_norm,
            flydsl=ixaite.rms_norm,
            flydsl_name="ixaite.rms_norm",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[RmsNormCase]:
        spec = l2_spec(model, self.name)
        hidden = int(spec["hidden_size"])
        cases = [
            RmsNormCase(f"input_or_final_norm@{model};T={t}", t, hidden)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(RmsNormSuite())
