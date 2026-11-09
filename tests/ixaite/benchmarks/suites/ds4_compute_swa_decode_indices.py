# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 SWA decode index builder (ixformer vs ixaite)."""

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
class Ds4SwaDecodeCase:
    case_id: str
    t: int
    window_size: int = 128
    block_size: int = 64
    tags: tuple[str, ...] = ("attention", "ds4", "index")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        max_blocks = 8
        swa_indices = torch.full(
            (self.t, 1, self.window_size), -1, device=device, dtype=torch.int32
        )
        swa_lens = torch.zeros((self.t,), device=device, dtype=torch.int32)
        qsl = torch.tensor([0, self.t], device=device, dtype=torch.int32)
        sl = torch.tensor([self.t + 32], device=device, dtype=torch.int32)
        t2r = torch.zeros((self.t,), device=device, dtype=torch.int32)
        valid = torch.ones((self.t,), device=device, dtype=torch.bool)
        block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).view(1, -1)
        return Workload(
            args=(
                swa_indices,
                swa_lens,
                self.window_size,
                qsl,
                sl,
                t2r,
                valid,
                block_table,
                self.block_size,
                self.t,
            ),
            meta={"idx0": swa_indices.clone(), "lens0": swa_lens.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["idx0"])
        wl.args[1].copy_(wl.meta["lens0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(flydsl_wl.args[0], baseline_wl.args[0], rtol=0, atol=0)
        torch.testing.assert_close(flydsl_wl.args[1], baseline_wl.args[1], rtol=0, atol=0)

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        return None


class Ds4ComputeSwaDecodeSuite:
    name = "ds4_compute_swa_decode_indices"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_compute_swa_decode_indices import (
            ds4_compute_swa_decode_indices_and_lens,
        )

        return Providers(
            baseline_name="IXF.ds4_compute_swa_decode_indices_and_lens",
            baseline=ixf_ops.ds4_compute_swa_decode_indices_and_lens,
            flydsl=ds4_compute_swa_decode_indices_and_lens,
            flydsl_name="ixaite.ds4_compute_swa_decode_indices_and_lens",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4SwaDecodeCase]:
        cases = [
            Ds4SwaDecodeCase(f"ds4_compute_swa_decode_indices@{model};tok={t:04d}", t)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4ComputeSwaDecodeSuite())
