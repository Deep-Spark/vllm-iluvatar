# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 Vision packed-QKV layout suite (einops vs ixaite)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


def _einops_split(
    packed: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len, batch, _ = packed.shape
    qkv = packed.view(seq_len, batch, 3, num_heads, head_dim).permute(1, 0, 2, 3, 4)
    return qkv.unbind(dim=2)


@dataclass(frozen=True)
class VitQkvLayoutCase:
    case_id: str
    seq_len: int
    num_heads: int
    head_dim: int
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"vit_qkv_layout models the BF16 serving path, got {dtype}"
            )
        packed = torch.randn(
            (self.seq_len, 1, 3 * self.num_heads * self.head_dim),
            device=device,
            dtype=dtype,
        )
        return Workload(
            args=(packed, self.num_heads, self.head_dim),
            meta={
                "S": self.seq_len,
                "H": self.num_heads,
                "D": self.head_dim,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["q"], wl.outputs["k"], wl.outputs["v"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        for name in ("q", "k", "v"):
            torch.testing.assert_close(
                flydsl_wl.outputs[name],
                baseline_wl.outputs[name],
                rtol=0,
                atol=0,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        elements = wl.args[0].numel()
        return elements / (median_ms * 1e-3) / 1e9, "GElements/s"


class VitQkvLayoutSuite:
    name = "vit_qkv_layout"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="eager permute split",
            baseline=_einops_split,
            flydsl=ixaite.vit_qkv_layout,
            flydsl_name="ixaite.vit_qkv_layout",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[VitQkvLayoutCase]:
        spec = l2_spec(model, self.name)
        head_dim = int(spec["head_dim"])
        cases = []
        for seq_len in spec["sequence_lengths"]:
            for tp_index, num_heads in enumerate(spec["num_heads"]):
                tp = 1 << tp_index
                cases.append(
                    VitQkvLayoutCase(
                        case_id=(
                            f"vit_qkv_layout@{model};S={int(seq_len)};"
                            f"H={int(num_heads)};D={head_dim};tp={tp}"
                        ),
                        seq_len=int(seq_len),
                        num_heads=int(num_heads),
                        head_dim=head_dim,
                        tags=("vision", "layout", "bf16", f"tp{tp}"),
                    )
                )
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 0.0


register(VitQkvLayoutSuite())
