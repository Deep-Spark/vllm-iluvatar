# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 compressor save / pool (ixformer vs ixaite)."""

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

_HEAD = 512
_ROPE = 64
_EPS = 1e-6


def _geom(model: str, suite: str) -> tuple[int, int, int, int]:
    spec = l2_spec(model, suite)
    return (
        int(spec["head_dim"]),
        int(spec["compress_ratio"]),
        int(spec["state_block_size"]),
        int(spec["kv_block_size"]),
    )


def _width(cr: int, head_dim: int) -> int:
    return (2 if cr == 4 else 1) * head_dim


@dataclass(frozen=True)
class Ds4CompressorSaveCase:
    case_id: str
    t: int
    head_dim: int
    compress_ratio: int
    state_block_size: int
    tags: tuple[str, ...] = ("ds4", "compressor", "fp32")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        del dtype
        width = _width(self.compress_ratio, self.head_dim)
        kv_score = torch.randn((self.t, 2 * width), device=device, dtype=torch.float32)
        kv, score = kv_score.split([width, width], dim=-1)
        ape = torch.randn((self.compress_ratio, width), device=device, dtype=torch.float32)
        positions = torch.arange(self.t, device=device, dtype=torch.int64)
        slot_mapping = torch.arange(self.t, device=device, dtype=torch.int64)
        num_blocks = max(1, (self.t + self.state_block_size - 1) // self.state_block_size)
        cache = torch.zeros(
            (num_blocks, self.state_block_size, 2 * width),
            device=device,
            dtype=torch.float32,
        )
        return Workload(
            args=(
                kv,
                score,
                ape,
                positions,
                slot_mapping,
                cache,
                self.state_block_size,
                self.compress_ratio,
            ),
            meta={"T": self.t, "cache0": cache.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["cache"] = fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[5].copy_(wl.meta["cache0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["cache"],
            baseline_wl.outputs["cache"],
            rtol=0,
            atol=0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        cache = wl.args[5]
        bytes_moved = cache.numel() * cache.element_size()
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


@dataclass(frozen=True)
class Ds4CompressorPoolCase:
    case_id: str
    t: int
    head_dim: int
    compress_ratio: int
    state_block_size: int
    kv_block_size: int
    tags: tuple[str, ...] = ("ds4", "compressor", "bf16")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 compressor pool models the BF16 serving path, got {dtype}")
        width = _width(self.compress_ratio, self.head_dim)
        positions = torch.tensor(
            [(t + 1) * self.compress_ratio - 1 for t in range(self.t)],
            device=device,
            dtype=torch.int64,
        )
        max_src = int(positions.max().item())
        num_state_blocks = (max_src + self.state_block_size) // self.state_block_size
        state = torch.randn(
            (num_state_blocks, self.state_block_size, 2 * width),
            device=device,
            dtype=torch.float32,
        )
        block_table = torch.arange(num_state_blocks, device=device, dtype=torch.int32).view(1, -1)
        token_to_req = torch.zeros((self.t,), device=device, dtype=torch.int32)
        slot_mapping = torch.arange(self.t, device=device, dtype=torch.int64)
        kv_slot = torch.arange(self.t, device=device, dtype=torch.int64)
        weight = torch.randn((self.head_dim,), device=device, dtype=torch.float32)
        cos_sin = torch.randn((max_src + 1, _ROPE), device=device, dtype=torch.float32)
        k_cache = torch.zeros(
            (max(1, (self.t + self.kv_block_size - 1) // self.kv_block_size), self.kv_block_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        return Workload(
            args=(
                state,
                token_to_req,
                positions,
                slot_mapping,
                block_table,
                weight,
                cos_sin,
                k_cache,
                kv_slot,
                self.compress_ratio,
                _ROPE,
                _EPS,
            ),
            meta={"T": self.t, "k0": k_cache.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["k"] = fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[7].copy_(wl.meta["k0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["k"].float(),
            baseline_wl.outputs["k"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        k = wl.args[7]
        bytes_moved = k.numel() * k.element_size() * 2
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4CompressorSaveSuite:
    name = "ds4_compressor_save_partial_states"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_compressor_save_partial_states",
            baseline=ixf_ops.ds4_compressor_save_partial_states,
            flydsl=ixaite.ds4_compressor_save_partial_states,
            flydsl_name="ixaite.ds4_compressor_save_partial_states",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4CompressorSaveCase]:
        head_dim, cr, state_bs, _kv_bs = _geom(model, self.name)
        cases = [
            Ds4CompressorSaveCase(
                f"ds4_compressor_save@{model};tok={t:04d}",
                t,
                head_dim,
                cr,
                state_bs,
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


class Ds4CompressorPoolSuite:
    name = "ds4_compressor_pool_norm_rope_insert"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.ds4_compressor_pool_norm_rope_insert",
            baseline=ixf_ops.ds4_compressor_pool_norm_rope_insert,
            flydsl=ixaite.ds4_compressor_pool_norm_rope_insert,
            flydsl_name="ixaite.ds4_compressor_pool_norm_rope_insert",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[Ds4CompressorPoolCase]:
        head_dim, cr, state_bs, kv_bs = _geom(model, self.name)
        cases = [
            Ds4CompressorPoolCase(
                f"ds4_compressor_pool@{model};tok={t:04d}",
                t,
                head_dim,
                cr,
                state_bs,
                kv_bs,
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4CompressorSaveSuite())
register(Ds4CompressorPoolSuite())
