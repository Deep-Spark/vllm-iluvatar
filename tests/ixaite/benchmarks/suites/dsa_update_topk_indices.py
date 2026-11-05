# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 indexer top-k indices (ixformer vs ixaite)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

TOPK = 512


@dataclass(frozen=True)
class DsaTopkCase:
    case_id: str
    t: int
    max_ctx: int
    topk: int = TOPK
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "indexer")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        width = (self.max_ctx + 1) & ~1
        full = torch.randn((self.t, width), device=device, dtype=dtype)
        logits = full[:, : self.max_ctx]
        ks = torch.zeros(self.t, device=device, dtype=torch.int32)
        ke = torch.full((self.t,), self.max_ctx, device=device, dtype=torch.int32)
        buf = torch.full((self.t, self.topk), -1, device=device, dtype=torch.int32)
        return Workload(
            args=(logits, ks, ke, self.topk, buf),
            meta={"buf0": buf.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[4].copy_(wl.meta["buf0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        # Only the selected set is fixed, not which of several equal-valued
        # columns represents it, so compare the selected logit values rather
        # than the indices. On the selection path (max_ctx > topk) bf16 ties
        # are common and the two providers may break them differently.
        logits = baseline_wl.args[0].float()

        def _values(ids: torch.Tensor) -> torch.Tensor:
            rows = []
            for row in range(ids.shape[0]):
                idx = ids[row]
                valid = idx >= 0
                safe = torch.where(valid, idx, torch.zeros_like(idx))
                vals = logits[row][safe]
                vals = torch.where(valid, vals, torch.full_like(vals, float("-inf")))
                rows.append(torch.sort(vals, descending=True).values)
            return torch.stack(rows)

        torch.testing.assert_close(
            _values(flydsl_wl.args[4]),
            _values(baseline_wl.args[4]),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            (flydsl_wl.args[4] >= 0).sum(dim=1),
            (baseline_wl.args[4] >= 0).sum(dim=1),
            rtol=0,
            atol=0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        logits = wl.args[0]
        return logits.numel() * logits.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class DsaUpdateTopkIndicesSuite:
    name = "dsa_update_topk_indices"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        op = importlib.import_module("ixaite.ops.dsa_update_topk_indices")
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.dsa_update_topk_indices",
            baseline=ixf_ops.dsa_update_topk_indices,
            flydsl=op.dsa_update_topk_indices,
            flydsl_name="ixaite.dsa_update_topk_indices",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[DsaTopkCase]:
        spec = l2_spec(model, self.name)
        topk = int(spec["topk"])
        cases = [
            DsaTopkCase(
                f"dsa_update_topk@{model};T={t};L={width}", int(t), int(width), topk
            )
            for t, width in spec["shapes"]
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(DsaUpdateTopkIndicesSuite())
