# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MHC pre / post / hc_head (ixformer v2 ABI vs ixaite)."""

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

_N_HC = 4
_HIDDEN = 4096


def _geom(model: str, suite: str) -> tuple[int, int]:
    spec = l2_spec(model, suite)
    return int(spec["n_hc"]), int(spec["hidden"])


@dataclass(frozen=True)
class Ds4MhcPreCase:
    case_id: str
    t: int
    n_hc: int = _N_HC
    hidden: int = _HIDDEN
    tags: tuple[str, ...] = ("ds4", "mhc", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 MHC models the BF16 serving path, got {dtype}")
        mix = (2 + self.n_hc) * self.n_hc
        k = self.n_hc * self.hidden
        residual = torch.randn(
            (self.t, self.n_hc, self.hidden), device=device, dtype=dtype
        )
        fn = torch.randn((mix, k), device=device, dtype=dtype)
        hc_scale = torch.randn((3,), device=device, dtype=torch.float32)
        hc_base = torch.randn((mix,), device=device, dtype=torch.float32)
        return Workload(
            args=(residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20),
            meta={"T": self.t},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        post, comb, x = fn(*wl.args)
        wl.outputs["post"] = post
        wl.outputs["comb"] = comb
        wl.outputs["x"] = x

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["post"],
            baseline_wl.outputs["post"],
            rtol=1e-2,
            atol=1e-2,
        )
        # IXF Sinkhorn can drift ~0.09 vs the fp32 torch path on some tokens;
        # ixaite tracks torch. Allow that IXF slack on comb only.
        torch.testing.assert_close(
            flydsl_wl.outputs["comb"],
            baseline_wl.outputs["comb"],
            rtol=1.5e-1,
            atol=1.5e-1,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["x"].float(),
            baseline_wl.outputs["x"].float(),
            rtol=3e-2,
            atol=3e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        residual = wl.args[0]
        bytes_moved = residual.numel() * residual.element_size() * 2
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


@dataclass(frozen=True)
class Ds4MhcPostCase:
    case_id: str
    t: int
    n_hc: int = _N_HC
    hidden: int = _HIDDEN
    tags: tuple[str, ...] = ("ds4", "mhc", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 MHC models the BF16 serving path, got {dtype}")
        x = torch.randn((self.t, self.hidden), device=device, dtype=dtype)
        residual = torch.randn(
            (self.t, self.n_hc, self.hidden), device=device, dtype=dtype
        )
        post = torch.randn((self.t, self.n_hc, 1), device=device, dtype=torch.float32)
        comb = torch.randn(
            (self.t, self.n_hc, self.n_hc), device=device, dtype=torch.float32
        )
        out = torch.empty_like(residual)
        return Workload(args=(x, residual, post, comb, out), meta={"T": self.t})

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        residual = wl.args[1]
        bytes_moved = residual.numel() * residual.element_size() * 2
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


@dataclass(frozen=True)
class Ds4HcHeadCase:
    case_id: str
    t: int
    n_hc: int = _N_HC
    hidden: int = _HIDDEN
    tags: tuple[str, ...] = ("ds4", "mhc", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 MHC models the BF16 serving path, got {dtype}")
        k = self.n_hc * self.hidden
        hidden = torch.randn(
            (self.t, self.n_hc, self.hidden), device=device, dtype=dtype
        )
        hc_fn = torch.randn((self.n_hc, k), device=device, dtype=torch.float32)
        hc_scale = torch.randn((1,), device=device, dtype=torch.float32)
        hc_base = torch.randn((self.n_hc,), device=device, dtype=torch.float32)
        out = torch.empty((self.t, self.hidden), device=device, dtype=dtype)
        return Workload(
            args=(hidden, hc_fn, hc_scale, hc_base, 1e-6, 1e-6, out),
            meta={"T": self.t},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-1,
            atol=2e-1,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        hidden = wl.args[0]
        bytes_moved = hidden.numel() * hidden.element_size() * 2
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4MhcPreSuite:
    name = "ds4_mhc_pre"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_mhc_pre_v2",
            baseline=ixf_ops.ds4_mhc_pre_v2,
            flydsl=ixaite.ds4_mhc_pre_v2,
            flydsl_name="ixaite.ds4_mhc_pre_v2",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4MhcPreCase]:
        n_hc, hidden = _geom(model, self.name)
        cases = [
            Ds4MhcPreCase(f"ds4_mhc_pre@{model};tok={t:04d}", t, n_hc, hidden)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


class Ds4MhcPostSuite:
    name = "ds4_mhc_post"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_mhc_post",
            baseline=ixf_ops.ds4_mhc_post,
            flydsl=ixaite.ds4_mhc_post,
            flydsl_name="ixaite.ds4_mhc_post",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4MhcPostCase]:
        n_hc, hidden = _geom(model, self.name)
        cases = [
            Ds4MhcPostCase(f"ds4_mhc_post@{model};tok={t:04d}", t, n_hc, hidden)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


class Ds4HcHeadSuite:
    name = "ds4_hc_head"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_hc_head",
            baseline=ixf_ops.ds4_hc_head,
            flydsl=ixaite.ds4_hc_head,
            flydsl_name="ixaite.ds4_hc_head",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4HcHeadCase]:
        n_hc, hidden = _geom(model, self.name)
        cases = [
            Ds4HcHeadCase(f"ds4_hc_head@{model};tok={t:04d}", t, n_hc, hidden)
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4MhcPreSuite())
register(Ds4MhcPostSuite())
register(Ds4HcHeadSuite())
