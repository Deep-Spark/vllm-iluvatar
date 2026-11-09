# SPDX-License-Identifier: Apache-2.0
"""Int8 paged KV-cache writer suites (plugin CUDA vs ixaite)."""

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


def _random_int8(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randint(-128, 128, shape, dtype=torch.int8, device=device)


@dataclass(frozen=True)
class ReshapeAndCacheInt8Case:
    case_id: str
    op_name: str
    num_tokens: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    mixed_value: bool
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"{self.op_name} models the BF16 serving path, got {dtype}"
            )
        shape = (self.num_tokens, self.num_kv_heads, self.head_dim)
        cache_shape = (
            math.ceil(self.num_tokens / self.page_size),
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        )
        key = _random_int8(shape, device)
        value = (
            torch.randn(shape, dtype=torch.bfloat16, device=device)
            if self.mixed_value
            else _random_int8(shape, device)
        )
        key_scale = torch.randn(
            self.num_tokens,
            self.num_kv_heads,
            dtype=torch.float32,
            device=device,
        )
        key_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)
        value_cache = torch.zeros(cache_shape, dtype=value.dtype, device=device)
        key_scale_cache = torch.zeros(
            cache_shape[:3],
            dtype=torch.float32,
            device=device,
        )
        slot_mapping = torch.arange(
            self.num_tokens,
            dtype=torch.int64,
            device=device,
        )
        return Workload(
            args=(
                key,
                value,
                key_scale,
                key_cache,
                value_cache,
                key_scale_cache,
                slot_mapping,
            ),
            meta={
                "T": self.num_tokens,
                "Hkv": self.num_kv_heads,
                "D": self.head_dim,
                "page": self.page_size,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        for index in (3, 4, 5):
            torch.testing.assert_close(
                flydsl_wl.args[index],
                baseline_wl.args[index],
                rtol=0,
                atol=0,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        value_bytes = 2 if self.mixed_value else 1
        bytes_per_token_head = 2 * self.head_dim * (1 + value_bytes) + 2 * 4
        num_bytes = self.num_tokens * self.num_kv_heads * bytes_per_token_head
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class _ReshapeAndCacheInt8Suite:
    name: str
    mixed_value: bool

    def resolve_providers(self) -> Providers:
        import vllm_iluvatar._C  # noqa: F401

        ixaite = initialize_ixaite()
        return Providers(
            baseline_name=f"plugin._C.{self.name}",
            baseline=getattr(torch.ops._C, self.name),
            flydsl=getattr(ixaite, self.name),
            flydsl_name=f"ixaite.{self.name}",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[ReshapeAndCacheInt8Case]:
        spec = l2_spec(model, self.name)
        kv_heads_list = [
            int(kv_heads)
            for kv_heads in spec.get("num_kv_heads_tp", (spec["num_kv_heads"],))
        ]
        tags_base = ("attention", "kv_cache", "int8", "prefill")
        cases = [
            ReshapeAndCacheInt8Case(
                case_id=(
                    f"{self.name}@{model};T={num_tokens};"
                    f"Hkv={kv_heads};D={spec['head_dim']};"
                    f"page={spec['page_size']}"
                ),
                op_name=self.name,
                num_tokens=num_tokens,
                num_kv_heads=kv_heads,
                head_dim=int(spec["head_dim"]),
                page_size=int(spec["page_size"]),
                mixed_value=self.mixed_value,
                tags=tags_base
                + ({4: "tp1", 2: "tp2", 1: "tp4"}.get(kv_heads, f"hkv{kv_heads}"),),
            )
            for kv_heads in kv_heads_list
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 0.0


class ReshapeAndCacheWithScaleSuite(_ReshapeAndCacheInt8Suite):
    name = "reshape_and_cache_with_scale"
    mixed_value = False


class ReshapeAndCacheMixSuite(_ReshapeAndCacheInt8Suite):
    name = "reshape_and_cache_mix"
    mixed_value = True


register(ReshapeAndCacheWithScaleSuite())
register(ReshapeAndCacheMixSuite())
