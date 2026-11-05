# SPDX-License-Identifier: Apache-2.0
"""MoE fused swiglu + dynamic INT8 quant suite (ixformer vs ixaite)."""

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
class ActivationDynamicScaledInt8Case:
    case_id: str
    num_tokens: int
    hidden_size: int
    act_type: str
    output_format: int
    tags: tuple[str, ...] = ("quant", "int8", "moe", "w4a8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("activation_dynamic_scaled_int8 models FP16/BF16")
        if self.act_type in ("swiglu", "swiglustep"):
            in_k = 2 * self.hidden_size
        else:
            in_k = self.hidden_size
        x = torch.randn(self.num_tokens, in_k, device=device, dtype=dtype)
        return Workload(
            args=(x,),
            kwargs={
                "act_type": self.act_type,
                "output_format": self.output_format,
            },
            meta={
                "T": self.num_tokens,
                "H": self.hidden_size,
                "fmt": self.output_format,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        quant, scales = fn(*wl.args, **wl.kwargs)
        wl.outputs["quant"] = quant
        wl.outputs["scales"] = scales

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["scales"],
            baseline_wl.outputs["scales"],
            rtol=1e-5,
            atol=1e-6,
        )
        diff = (
            flydsl_wl.outputs["quant"].to(torch.int16)
            - baseline_wl.outputs["quant"].to(torch.int16)
        ).abs()
        assert int(diff.max()) <= 1

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        rows, width = wl.args[0].shape
        num_bytes = rows * width * (wl.args[0].element_size() + 1) + rows * 4
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class ActivationDynamicScaledInt8Suite:
    name = "activation_dynamic_scaled_int8"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.activation_dynamic_scaled_int8 import (
            activation_dynamic_scaled_int8 as ixaite_op,
        )
        return Providers(
            baseline_name="IXF.activation_dynamic_scaled_int8",
            baseline=ixf_ops.activation_dynamic_scaled_int8,
            flydsl=ixaite_op,
            flydsl_name="ixaite.activation_dynamic_scaled_int8",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[ActivationDynamicScaledInt8Case]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        act_type = str(spec["act_type"])
        formats = [int(value) for value in spec["output_formats"]]
        cases = [
            ActivationDynamicScaledInt8Case(
                case_id=(
                    f"activation_dynamic_scaled_int8@{model};"
                    f"T={num_tokens};H={hidden_size};fmt={output_format}"
                ),
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                act_type=act_type,
                output_format=output_format,
            )
            for output_format in formats
            if output_format != 1 or hidden_size % 64 == 0
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(ActivationDynamicScaledInt8Suite())
