# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 attention output merge (ixformer vs ixaite)."""

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

HEADS = 8
HEAD_DIM = 512


@dataclass(frozen=True)
class Ds4MergeCase:
    case_id: str
    t: int
    use_sink: bool
    tags: tuple[str, ...] = ("attention", "ds4", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        o1 = torch.randn((self.t, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
        o2 = torch.randn((self.t, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
        lse1 = torch.randn((HEADS, self.t), device=device, dtype=torch.float32)
        lse2 = torch.randn((HEADS, self.t), device=device, dtype=torch.float32)
        out = torch.empty_like(o1)
        sink = (
            torch.randn((HEADS,), device=device, dtype=torch.float32) if self.use_sink else None
        )
        return Workload(args=(out, o1, lse1, o2, lse2, sink), meta={"T": self.t})

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0].float(),
            baseline_wl.args[0].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        o = wl.args[0]
        return 3 * o.numel() * o.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4MergeAttnOutputsSuite:
    name = "ds4_merge_attn_outputs"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_merge_attn_outputs import ds4_merge_attn_outputs

        return Providers(
            baseline_name="IXF.ds4_merge_attn_outputs",
            baseline=ixf_ops.ds4_merge_attn_outputs,
            flydsl=ds4_merge_attn_outputs,
            flydsl_name="ixaite.ds4_merge_attn_outputs",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4MergeCase]:
        cases = [
            Ds4MergeCase(
                f"ds4_merge_attn_outputs@{model};tok={t:04d};sink={int(sink)}", t, sink
            )
            for t in RUNTIME_T
            for sink in (False, True)
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4MergeAttnOutputsSuite())
