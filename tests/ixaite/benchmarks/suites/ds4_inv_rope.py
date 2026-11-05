# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 inverse GPT-J RoPE (ixformer vs ixaite)."""

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


def _geom(model: str) -> tuple[int, int]:
    spec = l2_spec(model, "ds4_inv_rope")
    return int(spec["num_query_heads"]), int(spec["rotary_dim"])


@dataclass(frozen=True)
class Ds4InvRopeCase:
    case_id: str
    t: int
    heads: int
    rotary_dim: int
    tags: tuple[str, ...] = ("attention", "ds4", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        o = torch.randn((self.t, self.heads, 512), device=device, dtype=torch.bfloat16)
        positions = torch.arange(self.t, device=device, dtype=torch.int32)
        cos_sin = torch.randn((self.t + 8, self.rotary_dim), device=device, dtype=torch.float32)
        return Workload(
            args=(o, positions, cos_sin, True, None),
            meta={"T": self.t, "o0": o.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["o0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0].float(),
            baseline_wl.args[0].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        o = wl.args[0]
        return 2 * o.numel() * o.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4InvRopeSuite:
    name = "ds4_inv_rope"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_inv_rope",
            baseline=ixf_ops.ds4_inv_rope,
            flydsl=ixaite.ds4_inv_rope,
            flydsl_name="ixaite.ds4_inv_rope",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4InvRopeCase]:
        heads, rotary = _geom(model)
        cases = [
            Ds4InvRopeCase(f"ds4_inv_rope@{model};tok={t:04d}", t, heads, rotary) for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4InvRopeSuite())
