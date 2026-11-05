# SPDX-License-Identifier: Apache-2.0
"""MoE expand + dynamic INT8 quant suite (ixformer vs ixaite)."""

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


def _routing(num_tokens: int, topk: int, device: torch.device):
    dst_tokens = num_tokens * topk
    dst_to_src = torch.randperm(dst_tokens, device=device, dtype=torch.int32)
    src_to_dst = torch.empty_like(dst_to_src)
    src_to_dst[dst_to_src] = torch.arange(dst_tokens, device=device, dtype=torch.int32)
    return src_to_dst, dst_to_src, dst_tokens


@dataclass(frozen=True)
class MoeExpandInputDynamicScaledInt8Case:
    case_id: str
    num_tokens: int
    hidden_size: int
    topk: int
    output_format: int
    tags: tuple[str, ...] = ("moe", "w4a8", "int8")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("moe_expand_input_dynamic_scaled_int8 needs fp16/bf16")
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, device=device, dtype=dtype
        ).clamp(-1, 1)
        src_to_dst, dst_to_src, dst_tokens = _routing(
            self.num_tokens, self.topk, device
        )
        return Workload(
            args=(hidden, dst_to_src, dst_tokens, self.topk),
            kwargs={
                "src_to_dst": src_to_dst,
                "output_format": self.output_format,
            },
            meta={"T": self.num_tokens, "H": self.hidden_size, "k": self.topk},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        quant, scales = fn(*wl.args, **wl.kwargs)
        wl.outputs["quant"] = quant
        wl.outputs["scales"] = scales

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        dtype = baseline_wl.args[0].dtype
        atol_i8 = 1 if dtype == torch.float16 else 2
        atol_s = 0.01 if dtype == torch.float16 else 0.03
        diff = (
            flydsl_wl.outputs["quant"].to(torch.int16)
            - baseline_wl.outputs["quant"].to(torch.int16)
        ).abs()
        assert int(diff.max()) <= atol_i8
        torch.testing.assert_close(
            flydsl_wl.outputs["scales"],
            baseline_wl.outputs["scales"],
            rtol=1e-5,
            atol=atol_s,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        rows, width = wl.args[0].shape
        dst = int(wl.args[2])
        num_bytes = rows * width * wl.args[0].element_size() + dst * (width + 4)
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class MoeExpandInputDynamicScaledInt8Suite:
    name = "moe_expand_input_dynamic_scaled_int8"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.moe_expand_input_dynamic_scaled_int8 import (
            moe_expand_input_dynamic_scaled_int8 as ixaite_op,
        )

        return Providers(
            baseline_name="IXF.moe_expand_input_dynamic_scaled_int8",
            baseline=ixf_ops.moe_expand_input_dynamic_scaled_int8,
            flydsl=ixaite_op,
            flydsl_name="ixaite.moe_expand_input_dynamic_scaled_int8",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[MoeExpandInputDynamicScaledInt8Case]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        topk = int(spec["topk"])
        formats = [int(value) for value in spec["output_formats"]]
        cases = [
            MoeExpandInputDynamicScaledInt8Case(
                case_id=(
                    f"expand@{model};T={num_tokens};H={hidden_size};"
                    f"k={topk};fmt={output_format}"
                ),
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                topk=topk,
                output_format=output_format,
            )
            for output_format in formats
            if output_format != 1 or hidden_size % 64 == 0
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(MoeExpandInputDynamicScaledInt8Suite())
