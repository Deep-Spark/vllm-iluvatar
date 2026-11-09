# SPDX-License-Identifier: Apache-2.0
"""MoE W4A8 grouped GEMM suite (ixformer vs ixaite, TN / version-2)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite
from tests.ixaite.benchmarks.suites._w4a8_moe import packed_grouped_operands


@dataclass(frozen=True)
class MoeW4A8GroupGemmCase:
    case_id: str
    tokens_per_experts: tuple[int, ...]
    n: int
    k: int
    tags: tuple[str, ...] = ("gemm", "int8", "moe", "w4a8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("moe_w4a8_group_gemm output dtype must be fp16/bf16")
        x_q, weight, i_scales, w_scales, tokens = packed_grouped_operands(
            tokens_per_experts=list(self.tokens_per_experts),
            n=self.n,
            k=self.k,
            device=device,
        )
        total = sum(self.tokens_per_experts)
        output = torch.empty(total, self.n, device=device, dtype=dtype)
        return Workload(
            args=(x_q, weight, i_scales, w_scales, tokens, output),
            meta={
                "M": total,
                "N": self.n,
                "K": self.k,
                "flops": 2 * total * self.n * self.k,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        x_q, weight, i_scales, w_scales, tokens, output = wl.args
        fn(
            input=x_q,
            weight=weight,
            i_scales=i_scales,
            w_scales=w_scales,
            output_dtype=output.dtype,
            tokens_per_experts=tokens,
            format=2,
            version=2,
            group_size=-1,
            output=output,
        )

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[-1],
            baseline_wl.args[-1],
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        return float(wl.meta["flops"]) / (median_ms * 1e-3) / 1e12, "TOPS"


class MoeW4A8GroupGemmSuite:
    name = "moe_w4a8_group_gemm"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.moe_w4a8_group_gemm import moe_w4a8_group_gemm as ixaite_op
        return Providers(
            baseline_name="IXF.moe_w4a8_group_gemm",
            baseline=ixf_ops.moe_w4a8_group_gemm,
            flydsl=ixaite_op,
            flydsl_name="ixaite.moe_w4a8_group_gemm",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[MoeW4A8GroupGemmCase]:
        spec = l2_spec(model, self.name)
        calls: Mapping[str, Mapping[str, object]] = spec["calls"]
        cases = [
            MoeW4A8GroupGemmCase(
                case_id=(
                    f"{slug}@{model};N={int(call['N'])};K={int(call['K'])};"
                    f"tokens={','.join(str(int(t)) for t in call['tokens'])}"
                ),
                tokens_per_experts=tuple(int(t) for t in call["tokens"]),
                n=int(call["N"]),
                k=int(call["K"]),
            )
            for slug, call in calls.items()
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(MoeW4A8GroupGemmSuite())
