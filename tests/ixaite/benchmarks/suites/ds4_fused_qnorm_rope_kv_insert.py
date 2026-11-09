# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 fused Q-norm / RoPE / KV insert (ixformer vs ixaite).

0731 W4A8 uses BF16 SWA cache, D=512, rotary=64, page=64. TP8 is 8 local
query heads; TP1 is 64. The suite compares contiguous cache only -- ixformer
rejects the padded vLLM SWA view that ixaite accepts.
"""

from __future__ import annotations

import math
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
class Ds4InsertCase:
    case_id: str
    t: int
    heads: int
    head_dim: int
    rotary_dim: int
    block_size: int
    tags: tuple[str, ...] = ("attention", "ds4", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 insert models the BF16 serving path, got {dtype}")
        q = torch.randn((self.t, self.heads, self.head_dim), device=device, dtype=dtype)
        kv = torch.randn((self.t, self.head_dim), device=device, dtype=dtype)
        num_blocks = math.ceil(self.t / self.block_size)
        cache = torch.zeros(
            (num_blocks, self.block_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        slot_mapping = torch.arange(self.t, device=device, dtype=torch.int32)
        positions = torch.arange(self.t, device=device, dtype=torch.int32)
        cos_sin = torch.randn(
            (self.t + 8, self.rotary_dim), device=device, dtype=torch.float32
        )
        return Workload(
            args=(
                q,
                kv,
                cache,
                slot_mapping,
                positions,
                cos_sin,
                1e-6,
                self.block_size,
            ),
            meta={
                "T": self.t,
                "H": self.heads,
                "D": self.head_dim,
                "q0": q.clone(),
                "cache0": cache.clone(),
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["q0"])
        wl.args[2].copy_(wl.meta["cache0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[0].float(),
            baseline_wl.args[0].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.args[2].float(),
            baseline_wl.args[2].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        kv = wl.args[1]
        num_bytes = (
            2 * q.numel() + kv.numel() + q.numel() // self.heads
        ) * q.element_size()
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4FusedQnormRopeKvInsertSuite:
    name = "ds4_fused_qnorm_rope_kv_insert"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_fused_qnorm_rope_kv_insert",
            baseline=ixf_ops.ds4_fused_qnorm_rope_kv_insert,
            flydsl=ixaite.ds4_fused_qnorm_rope_kv_insert,
            flydsl_name="ixaite.ds4_fused_qnorm_rope_kv_insert",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4InsertCase]:
        spec = l2_spec(model, self.name)
        heads = int(spec["num_query_heads"])
        d = int(spec["head_dim"])
        rotary = int(spec["rotary_dim"])
        block = int(spec["block_size"])
        cases = [
            Ds4InsertCase(
                f"ds4_insert@{model};T={t}",
                t,
                heads,
                d,
                rotary,
                block,
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4FusedQnormRopeKvInsertSuite())
