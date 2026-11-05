# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 SWA prefill gather-length builder (ixformer vs ixaite)."""

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
class Ds4SwaPrefillGatherCase:
    case_id: str
    t: int
    window_size: int = 128
    tags: tuple[str, ...] = ("attention", "ds4", "index")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        del dtype
        n_decodes = 0
        n_prefills = self.t
        gather = torch.zeros((n_prefills,), device=device, dtype=torch.int32)
        seq_lens = torch.arange(1, n_prefills + 1, device=device, dtype=torch.int32) + 16
        qsl = torch.arange(n_prefills + 1, device=device, dtype=torch.int32)
        return Workload(
            args=(gather, seq_lens, qsl, n_prefills, n_decodes, self.window_size),
            meta={"gather0": gather.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["gather0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(flydsl_wl.args[0], baseline_wl.args[0], rtol=0, atol=0)

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        return None


class Ds4ComputeSwaPrefillGatherSuite:
    name = "ds4_compute_swa_prefill_gather_lens"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_compute_swa_prefill_gather_lens import (
            ds4_compute_swa_prefill_gather_lens,
        )

        return Providers(
            baseline_name="IXF.ds4_compute_swa_prefill_gather_lens",
            baseline=ixf_ops.ds4_compute_swa_prefill_gather_lens,
            flydsl=ds4_compute_swa_prefill_gather_lens,
            flydsl_name="ixaite.ds4_compute_swa_prefill_gather_lens",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4SwaPrefillGatherCase]:
        cases = [
            Ds4SwaPrefillGatherCase(
                f"ds4_compute_swa_prefill_gather_lens@{model};tok={t:04d}", t
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4ComputeSwaPrefillGatherSuite())
