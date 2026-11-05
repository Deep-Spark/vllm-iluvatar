# SPDX-License-Identifier: Apache-2.0
"""MoE fused top-k + bias scoring suite (ixformer vs ixaite)."""

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
class MoeFusedTopkBiasCase:
    case_id: str
    num_tokens: int
    num_experts: int
    topk: int
    scoring_func: str
    renormalize: bool
    has_bias: bool
    tags: tuple[str, ...] = ("moe", "w4a8", "router")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("moe_fused_topk_bias needs fp32/fp16/bf16")
        torch.manual_seed(self.num_tokens * 17 + self.num_experts + self.topk)
        gating = torch.randn(
            self.num_tokens, self.num_experts, device=device, dtype=dtype
        )
        bias = (
            torch.randn(self.num_experts, device=device, dtype=dtype)
            if self.has_bias
            else None
        )
        return Workload(
            args=(gating,),
            kwargs={
                "topk": self.topk,
                "scoring_func": self.scoring_func,
                "routed_scaling_factor": 1.0,
                "renormalize": self.renormalize,
                "e_score_correction_bias": bias,
            },
            meta={"T": self.num_tokens, "E": self.num_experts, "k": self.topk},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        weights, ids = fn(*wl.args, **wl.kwargs)
        wl.outputs["weights"] = weights
        wl.outputs["ids"] = ids

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        dtype = baseline_wl.args[0].dtype
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        elif dtype == torch.float16:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-2, 1e-2
        torch.testing.assert_close(
            flydsl_wl.outputs["weights"],
            baseline_wl.outputs["weights"],
            rtol=rtol,
            atol=atol,
        )
        score = baseline_wl.args[0].float()
        if self.scoring_func == "softmax":
            score = score.softmax(dim=-1)
        else:
            score = score.sigmoid()
        bias = baseline_wl.kwargs["e_score_correction_bias"]
        cmp_score = score if bias is None else score + bias.float().unsqueeze(0)
        torch.testing.assert_close(
            cmp_score.gather(1, flydsl_wl.outputs["ids"].long()),
            cmp_score.gather(1, baseline_wl.outputs["ids"].long()),
            rtol=rtol,
            atol=atol,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        tokens, experts = wl.args[0].shape
        return tokens * experts / (median_ms * 1e-3) / 1e6, "MElem/s"


class MoeFusedTopkBiasSuite:
    name = "moe_fused_topk_bias"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.moe_fused_topk_bias import moe_fused_topk_bias as ixaite_op

        return Providers(
            baseline_name="IXF.moe_fused_topk_bias",
            baseline=ixf_ops.moe_fused_topk_bias,
            flydsl=ixaite_op,
            flydsl_name="ixaite.moe_fused_topk_bias",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[MoeFusedTopkBiasCase]:
        spec = l2_spec(model, self.name)
        num_experts = int(spec["num_experts"])
        topk = int(spec["topk"])
        scoring_func = str(spec["scoring_func"])
        cases = [
            MoeFusedTopkBiasCase(
                case_id=(
                    f"fused_topk@{model};T={num_tokens};E={num_experts};"
                    f"k={topk};bias={int(has_bias)}"
                ),
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                scoring_func=scoring_func,
                renormalize=True,
                has_bias=has_bias,
            )
            for has_bias in (False, True)
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(MoeFusedTopkBiasSuite())
