# SPDX-License-Identifier: Apache-2.0
"""W8A8 activation INT8 quant suite (ixformer vs ixaite)."""

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


def _ixformer_dynamic(x: torch.Tensor):
    import ixformer.inference.functions as ixf_ops

    output = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(
        (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
    )
    ixf_ops.dynamic_scaled_int8_quant(output, x, scales)
    return output, scales


@dataclass(frozen=True)
class ScaledInt8QuantCase:
    case_id: str
    num_tokens: int
    hidden_size: int
    tags: tuple[str, ...] = ("quant", "int8", "w8a8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("scaled_int8_quant models FP16/BF16 activations")
        x = torch.randn(
            self.num_tokens, self.hidden_size, device=device, dtype=dtype
        )
        return Workload(
            args=(x,),
            meta={"T": self.num_tokens, "K": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        result = fn(*wl.args)
        if isinstance(result, tuple):
            wl.outputs["quant"] = result[0]
            wl.outputs["scales"] = result[1]
        else:
            wl.outputs["quant"], wl.outputs["scales"] = result

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["scales"],
            baseline_wl.outputs["scales"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["quant"],
            baseline_wl.outputs["quant"],
            atol=1,
            rtol=0.0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        num_bytes = self.num_tokens * self.hidden_size * (
            wl.args[0].element_size() + 1
        ) + self.num_tokens * 4
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class ScaledInt8QuantSuite:
    name = "scaled_int8_quant"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()

        def ixaite_dynamic(x: torch.Tensor):
            quant, scales, _ = ixaite.scaled_int8_quant(x)
            return quant, scales

        return Providers(
            baseline_name="IXF.dynamic_scaled_int8_quant",
            baseline=_ixformer_dynamic,
            flydsl=ixaite_dynamic,
            flydsl_name="ixaite.scaled_int8_quant",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[ScaledInt8QuantCase]:
        spec = l2_spec(model, self.name)
        hidden_sizes = [int(k) for k in spec["hidden_sizes"]]
        cases = [
            ScaledInt8QuantCase(
                case_id=f"scaled_int8_quant@{model};T={num_tokens};K={hidden}",
                num_tokens=num_tokens,
                hidden_size=hidden,
            )
            for hidden in hidden_sizes
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(ScaledInt8QuantSuite())
