# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 wo_a marlin_w8a16 (ixformer vs ixaite)."""

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


def _geom(model: str) -> tuple[int, int, int]:
    spec = l2_spec(model, "marlin_w8a16")
    return int(spec["batch"]), int(spec["n"]), int(spec["k"])


@dataclass(frozen=True)
class MarlinW8A16Case:
    case_id: str
    t: int
    batch: int
    n: int
    k: int
    fmt: str = "k16n16"
    tags: tuple[str, ...] = ("ds4", "w8a16", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        del dtype
        inp = torch.randn((self.t, self.batch, self.k), device=device, dtype=torch.bfloat16)
        src = torch.randint(-8, 8, (self.batch, self.k, self.n), device=device, dtype=torch.int8)
        import ixformer.inference.functions as ixf_ops

        packed = ixf_ops.marlin_w8_weight_repack(
            weights=src, weight_format="int8", reformat=self.fmt
        )
        if self.fmt == "k16n16_grouped_n":
            scale = torch.randn((self.batch, self.k, 1), device=device, dtype=torch.float32) * 0.02
        else:
            scale = torch.randn((self.batch, 1, self.n), device=device, dtype=torch.float32) * 0.02
        return Workload(
            args=(inp, packed, scale),
            meta={"T": self.t, "packed": packed, "scale": scale},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        inp, packed, scale = wl.args
        wl.outputs["out"] = fn(
            inp, packed, scale, group_size=-1, format=self.fmt, batch_first=False
        )

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=8e-2,
            atol=2.5e-1,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        inp = wl.args[0]
        flops = 2 * inp.shape[0] * inp.shape[1] * self.n * self.k
        return flops / (median_ms * 1e-3) / 1e12, "TFLOPS"


class MarlinW8A16Suite:
    name = "marlin_w8a16"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.marlin_w8a16",
            baseline=ixf_ops.marlin_w8a16,
            flydsl=ixaite.marlin_w8a16,
            flydsl_name="ixaite.marlin_w8a16",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[MarlinW8A16Case]:
        batch, n, k = _geom(model)
        cases = []
        for t in RUNTIME_T:
            cases.append(
                MarlinW8A16Case(f"marlin_w8a16@{model};tok={t:04d};B=1", t, 1, n, k)
            )
            if batch != 1:
                cases.append(
                    MarlinW8A16Case(
                        f"marlin_w8a16@{model};tok={t:04d};B={batch}", t, batch, n, k
                    )
                )
        # 0731 W_UK: q_nope (T, H, 448) x (H, 512, 448), scale on K.
        for t in RUNTIME_T:
            for heads in (8, 64):
                cases.append(
                    MarlinW8A16Case(
                        f"marlin_w8a16@{model};uk;tok={t:04d};B={heads}",
                        t,
                        heads,
                        512,
                        448,
                        fmt="k16n16_grouped_n",
                        tags=("ds4", "w8a16", "bf16", "uk"),
                    )
                )
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(MarlinW8A16Suite())
