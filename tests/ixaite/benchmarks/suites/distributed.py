# SPDX-License-Identifier: Apache-2.0
"""TP all_reduce suite (torch.distributed vs ixaite facade).

Both sides reach the same IXCCL transport, but the baseline goes through
ProcessGroupNCCL, which runs the collective on a private stream and
synchronises with events, while ixaite issues it on the current stream. That
bookkeeping is what the comparison measures at decode sizes.

Single-process L2 is world-size 1 (identity). Multi-rank serve coverage lives
on the 27B TP e2e profiles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


@dataclass(frozen=True)
class DistributedCase:
    case_id: str
    num_tokens: int
    hidden_size: int
    tags: tuple[str, ...] = ("comm", "all_reduce", "tp")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        tensor = torch.randn(
            (self.num_tokens, self.hidden_size),
            device=device,
            dtype=dtype,
        )
        return Workload(
            args=(tensor,),
            meta={"T": self.num_tokens, "H": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(wl.args[0])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"],
            baseline_wl.outputs["out"],
            rtol=0,
            atol=0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        num_bytes = wl.args[0].numel() * wl.args[0].element_size()
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


def _torch_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return tensor
    dist.all_reduce(tensor, async_op=True)
    return tensor


class DistributedSuite:
    name = "distributed"
    supports_cuda_graph = False

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixaite.ops.distributed as ixaite_dist

        return Providers(
            baseline_name="torch.distributed.all_reduce",
            baseline=_torch_all_reduce,
            flydsl=ixaite_dist.all_reduce,
            flydsl_name="ixaite.distributed.all_reduce",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[DistributedCase]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        cases = [
            DistributedCase(
                case_id=f"all_reduce@{model};T={num_tokens}",
                num_tokens=num_tokens,
                hidden_size=hidden_size,
            )
            for num_tokens in (int(value) for value in spec["num_tokens"])
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 0.0


register(DistributedSuite())
