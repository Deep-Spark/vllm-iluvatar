# SPDX-License-Identifier: Apache-2.0
"""MoE top-k weighted reduce-sum suite (ixformer vs ixaite)."""

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
class MoeOutputReduceSumCase:
    case_id: str
    num_tokens: int
    hidden_size: int
    topk: int
    tags: tuple[str, ...] = ("moe", "w4a8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("moe_output_reduce_sum needs fp16/bf16")
        x = torch.randn(
            self.num_tokens, self.topk, self.hidden_size, device=device, dtype=dtype
        ).clamp(-1, 1)
        weights = torch.rand(
            self.num_tokens, self.topk, device=device, dtype=torch.float32
        )
        return Workload(
            args=(x,),
            kwargs={"topk_weight": weights, "scaling_factor": 1.0},
            meta={"T": self.num_tokens, "H": self.hidden_size, "k": self.topk},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args, **wl.kwargs)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"],
            baseline_wl.outputs["out"],
            rtol=1e-5,
            atol=1e-1,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        num_bytes = wl.args[0].numel() * wl.args[0].element_size()
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class MoeOutputReduceSumSuite:
    name = "moe_output_reduce_sum"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.moe_output_reduce_sum import (
            moe_output_reduce_sum as ixaite_op,
        )

        return Providers(
            baseline_name="IXF.moe_output_reduce_sum",
            baseline=ixf_ops.moe_output_reduce_sum,
            flydsl=ixaite_op,
            flydsl_name="ixaite.moe_output_reduce_sum",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[MoeOutputReduceSumCase]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        topk = int(spec["topk"])
        cases = [
            MoeOutputReduceSumCase(
                case_id=f"reduce@{model};T={num_tokens};H={hidden_size};k={topk}",
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                topk=topk,
            )
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(MoeOutputReduceSumSuite())
