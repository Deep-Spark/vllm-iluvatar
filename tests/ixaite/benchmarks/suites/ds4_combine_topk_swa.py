# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 topk+SWA index combiner (ixformer vs ixaite)."""

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
class Ds4CombineCase:
    case_id: str
    t: int
    mode: str
    window_size: int = 128
    tags: tuple[str, ...] = ("attention", "ds4", "index")

    def _mode(self) -> tuple[int, int, int, int]:
        if self.mode == "swa":
            return 1, 0, 0, 1
        if self.mode == "c4a":
            return 4, 32, 256, 32
        return 128, 32, 64, 32

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        cr, topk, n_width, topk_cols = self._mode()
        qsl = torch.tensor([0, self.t], device=device, dtype=torch.int32)
        seq = max(self.t + 16, self.window_size)
        sl = torch.tensor([seq], device=device, dtype=torch.int32)
        gl = torch.tensor([min(self.window_size, seq)], device=device, dtype=torch.int32)
        topk_indices = torch.randint(-1, 64, (self.t, topk_cols), device=device, dtype=torch.int32)
        m_stride = n_width + self.window_size + self.t
        return Workload(
            args=(topk_indices, qsl, sl, gl, self.window_size, cr, topk, m_stride, n_width),
            meta={"T": self.t, "mode": self.mode},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        idx, lens = fn(*wl.args)
        wl.outputs["idx"] = idx
        wl.outputs["lens"] = lens

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(flydsl_wl.outputs["idx"], baseline_wl.outputs["idx"], rtol=0, atol=0)
        torch.testing.assert_close(
            flydsl_wl.outputs["lens"], baseline_wl.outputs["lens"], rtol=0, atol=0
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        return None


class Ds4CombineTopkSwaSuite:
    name = "ds4_combine_topk_swa"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_combine_topk_swa import ds4_combine_topk_swa_indices

        return Providers(
            baseline_name="IXF.ds4_combine_topk_swa_indices",
            baseline=ixf_ops.ds4_combine_topk_swa_indices,
            flydsl=ds4_combine_topk_swa_indices,
            flydsl_name="ixaite.ds4_combine_topk_swa_indices",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4CombineCase]:
        cases = [
            Ds4CombineCase(f"ds4_combine_topk_swa@{model};tok={t:04d};{mode}", t, mode)
            for t in RUNTIME_T
            for mode in ("swa", "c4a", "c128a")
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4CombineTopkSwaSuite())
