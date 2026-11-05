# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 Vision GELU suite (PyTorch vs ixaite)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


@dataclass(frozen=True)
class GeluCase:
    case_id: str
    sequence_length: int
    rows: int
    width: int
    approximate: str
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"gelu models the BF16 serving path, got {dtype}")
        shape = (
            (self.rows, 1, self.width)
            if self.approximate == "tanh"
            else (self.rows, self.width)
        )
        input = torch.randn(shape, device=device, dtype=dtype)
        return Workload(
            args=(input,),
            kwargs={"approximate": self.approximate},
            meta={
                "S": self.sequence_length,
                "rows": self.rows,
                "N": self.width,
                "approximate": self.approximate,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["result"] = fn(*wl.args, **wl.kwargs)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["result"],
            baseline_wl.outputs["result"],
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        elements = wl.args[0].numel()
        return elements / (median_ms * 1e-3) / 1e9, "GElements/s"


class GeluSuite:
    name = "gelu"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="torch.gelu",
            baseline=F.gelu,
            flydsl=ixaite.gelu,
            flydsl_name="ixaite.gelu",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[GeluCase]:
        spec = l2_spec(model, self.name)
        sequence_lengths = [int(value) for value in spec["sequence_lengths"]]
        cases = []
        for approximate, widths in (
            ("tanh", spec["tanh_widths"]),
            ("none", spec["erf_widths"]),
        ):
            for sequence_length in sequence_lengths:
                rows = (
                    sequence_length
                    if approximate == "tanh"
                    else sequence_length // 4
                )
                for tp_index, width in enumerate(widths):
                    tp = 1 << tp_index
                    cases.append(
                        GeluCase(
                            case_id=(
                                f"gelu@{model};approximate={approximate};"
                                f"S={sequence_length};N={int(width)};tp={tp}"
                            ),
                            sequence_length=sequence_length,
                            rows=rows,
                            width=int(width),
                            approximate=approximate,
                            tags=("vision", "activation", "bf16", f"tp{tp}"),
                        )
                    )
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(GeluSuite())
