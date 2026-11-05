# SPDX-License-Identifier: Apache-2.0
"""HND KV-cache update suite (ixformer vs ixaite).

Qwen3.5-9B expands TP-local ``Hkv`` via ``num_kv_heads_tp``. Qwen3.6-27B
TP2/TP4 profiles pin the full-attn shard (``Hkv=2`` / ``Hkv=1``).
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
class ReshapeAndCacheFlashCase:
    case_id: str
    num_tokens: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    tags: tuple[str, ...] = ("attention", "kv_cache", "bf16", "tp1")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"reshape_and_cache_flash models the BF16 serving path, got {dtype}"
            )
        num_blocks = math.ceil(self.num_tokens / self.page_size)
        key = torch.randn(
            self.num_tokens,
            self.num_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        value = torch.randn_like(key)
        key_cache = torch.zeros(
            num_blocks,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        value_cache = torch.zeros_like(key_cache)
        slot_mapping = torch.arange(self.num_tokens, device=device, dtype=torch.int64)
        return Workload(
            args=(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                "auto",
                1.0,
                1.0,
            ),
            meta={
                "T": self.num_tokens,
                "Hkv": self.num_kv_heads,
                "D": self.head_dim,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(flydsl_wl.args[2], baseline_wl.args[2])
        torch.testing.assert_close(flydsl_wl.args[3], baseline_wl.args[3])

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        element_size = wl.args[0].element_size()
        num_bytes = (
            self.num_tokens * 4 * self.num_kv_heads * self.head_dim * element_size
        )
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class ReshapeAndCacheFlashSuite:
    name = "reshape_and_cache_flash"

    def resolve_providers(self) -> Providers:
        import ixformer.inference.functions as ixf_ops

        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="ixformer.reshape_and_cache_flash",
            baseline=ixf_ops.reshape_and_cache_flash,
            flydsl=ixaite.reshape_and_cache_flash,
            flydsl_name="ixaite.reshape_and_cache_flash",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[ReshapeAndCacheFlashCase]:
        spec = l2_spec(model, self.name)
        kv_heads_list = [
            int(kv_heads)
            for kv_heads in spec.get("num_kv_heads_tp", (spec["num_kv_heads"],))
        ]
        cases = [
            ReshapeAndCacheFlashCase(
                case_id=(
                    f"reshape_and_cache_flash@{model};T={num_tokens};"
                    f"Hkv={kv_heads};D={spec['head_dim']};"
                    f"page={spec['page_size']}"
                ),
                num_tokens=num_tokens,
                num_kv_heads=kv_heads,
                head_dim=int(spec["head_dim"]),
                page_size=int(spec["page_size"]),
                tags=(
                    "attention",
                    "kv_cache",
                    "bf16",
                    {4: "tp1", 2: "tp2", 1: "tp4"}.get(kv_heads, f"hkv{kv_heads}"),
                ),
            )
            for kv_heads in kv_heads_list
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(ReshapeAndCacheFlashSuite())
