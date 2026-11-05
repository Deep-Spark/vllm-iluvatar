# SPDX-License-Identifier: Apache-2.0
"""Gate-less Q/K RMSNorm suite (ixformer vs ixaite FlyDSL).

Qwen3-8B geometry is D=128 with GQA (Hq, Hk)=(32, 8) at TP1. Do not route
through the GDN fused_qk_rmsnorm_* gate kernels.
"""

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
class RmsNormQkCase:
    case_id: str
    t: int
    hq: int
    hk: int
    d: int
    tags: tuple[str, ...] = ("norm", "qk")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        q = torch.randn((self.t, self.hq, self.d), device=device, dtype=dtype)
        k = torch.randn((self.t, self.hk, self.d), device=device, dtype=dtype)
        wq = torch.randn((self.d,), device=device, dtype=dtype)
        wk = torch.randn((self.d,), device=device, dtype=dtype)
        return Workload(
            args=(q, k, wq, wk, 1e-6),
            meta={"T": self.t, "Hq": self.hq, "Hk": self.hk, "D": self.d},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        q, k, wq, wk, eps = wl.args
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)
        fn(out_q, out_k, q, k, wq, wk, eps)
        wl.outputs["out_q"] = out_q  # type: ignore[assignment]
        wl.outputs["out_k"] = out_k  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out_q"].float(),
            baseline_wl.outputs["out_q"].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["out_k"].float(),
            baseline_wl.outputs["out_k"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        elem = wl.args[0].element_size()
        elems = self.t * (self.hq + self.hk) * self.d
        num_bytes = (2 * elems + 2 * self.d) * elem
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class RmsNormQkSuite:
    name = "rms_norm_qk"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar import _custom_ops as ilu_ops

        return Providers(
            baseline_name="IXF.rms_norm_qk",
            baseline=ilu_ops.rms_norm_qk,
            flydsl=ixaite.rms_norm_qk,
            flydsl_name="ixaite.rms_norm_qk",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[RmsNormQkCase]:
        spec = l2_spec(model, self.name)
        hq = int(spec["num_query_heads"])
        hk = int(spec["num_kv_heads"])
        d = int(spec["head_dim"])
        cases = [
            RmsNormQkCase(f"attn_qk_norm@{model};T={t}", t, hq, hk, d)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(RmsNormQkSuite())
