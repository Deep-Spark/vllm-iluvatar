# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 EVS retention-mask suite (Torch vs ixaite)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from vllm.multimodal.video_prune.evs import compute_retention_mask

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


@dataclass(frozen=True)
class VideoEvsCase:
    case_id: str
    temporal: int
    height: int
    width: int
    hidden_size: int
    pruning_rate: float
    tags: tuple[str, ...] = ("vision", "video", "evs", "bf16", "tp1")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"video_evs models the BF16 serving path, got {dtype}")
        tokens = self.temporal * (self.height // 2) * (self.width // 2)
        video = torch.randn((tokens, self.hidden_size), device=device, dtype=dtype)
        return Workload(
            args=(
                video,
                (self.temporal, self.height, self.width),
                2,
                self.pruning_rate,
            ),
            meta={"tokens": tokens, "D": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["mask"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["mask"],
            baseline_wl.outputs["mask"],
            rtol=0,
            atol=0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        del median_ms, wl
        return None


class VideoEvsSuite:
    name = "video_evs"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="torch.evs",
            baseline=compute_retention_mask,
            flydsl=ixaite.video_evs,
            flydsl_name="ixaite.video_evs",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[VideoEvsCase]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        cases = [
            VideoEvsCase(
                case_id=(
                    f"video_evs@{model};T={temporal};H={height};W={width};"
                    f"D={hidden_size};q={pruning_rate:g}"
                ),
                temporal=int(temporal),
                height=int(height),
                width=int(width),
                hidden_size=hidden_size,
                pruning_rate=float(pruning_rate),
            )
            for temporal, height, width in spec["grids"]
            for pruning_rate in spec["pruning_rates"]
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.1


register(VideoEvsSuite())
