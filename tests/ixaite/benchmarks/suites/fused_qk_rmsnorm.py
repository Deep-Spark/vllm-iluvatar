# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 fused Q/K RMSNorm(+RoPE) gate suites (plugin CUDA vs ixaite)."""

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

_EPS = 1e-6

# D=256 BF16 can miss the D=128 provider-test band by one or two ulps.
_ROPE_QK_RTOL = 8e-3
_ROPE_QK_ATOL = 4e-2
_GATE_QK_RTOL = 8e-3
_GATE_QK_ATOL = 4e-2


def _require_bf16(dtype: torch.dtype, suite: str) -> None:
    if dtype is not torch.bfloat16:
        raise ValueError(f"{suite} models the BF16 serving path, got {dtype}")


def _make_qk_inputs(
    num_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_gate = torch.randn(
        num_tokens,
        num_query_heads * 2 * head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        num_tokens,
        num_kv_heads * head_dim,
        device=device,
        dtype=dtype,
    )
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    return q_gate, k, q_weight, k_weight


def _store_outputs(wl: Workload, result: tuple[torch.Tensor, ...]) -> None:
    q, k, gate = result
    wl.outputs["q"] = q
    wl.outputs["k"] = k
    wl.outputs["gate"] = gate


def _check_qk_gate(
    baseline_wl: Workload,
    flydsl_wl: Workload,
    *,
    qk_rtol: float,
    qk_atol: float,
) -> None:
    torch.testing.assert_close(
        flydsl_wl.outputs["q"],
        baseline_wl.outputs["q"],
        rtol=qk_rtol,
        atol=qk_atol,
    )
    torch.testing.assert_close(
        flydsl_wl.outputs["k"],
        baseline_wl.outputs["k"],
        rtol=qk_rtol,
        atol=qk_atol,
    )
    torch.testing.assert_close(
        flydsl_wl.outputs["gate"],
        baseline_wl.outputs["gate"],
        atol=0,
        rtol=0,
    )


@dataclass(frozen=True)
class FusedQkRmsnormRopeGateCase:
    case_id: str
    num_tokens: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    cache_rows: int
    tags: tuple[str, ...] = ("attention", "norm", "rope", "bf16", "tp1")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        _require_bf16(dtype, "fused_qk_rmsnorm_rope_gate")
        q_gate, k, q_weight, k_weight = _make_qk_inputs(
            self.num_tokens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            device,
            dtype,
        )
        cache = torch.randn(
            self.cache_rows, self.rotary_dim, device=device, dtype=dtype
        )
        positions = torch.arange(self.num_tokens, device=device, dtype=torch.int64)
        return Workload(
            args=(
                q_gate,
                k,
                q_weight,
                k_weight,
                cache,
                positions,
                _EPS,
                self.num_query_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_dim,
            ),
            meta={
                "T": self.num_tokens,
                "Hq": self.num_query_heads,
                "Hkv": self.num_kv_heads,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        _store_outputs(wl, fn(*wl.args))  # type: ignore[arg-type]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        _check_qk_gate(
            baseline_wl,
            flydsl_wl,
            qk_rtol=_ROPE_QK_RTOL,
            qk_atol=_ROPE_QK_ATOL,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


@dataclass(frozen=True)
class FusedQkRmsnormGateCase:
    case_id: str
    num_tokens: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    tags: tuple[str, ...] = ("attention", "norm", "bf16", "tp1")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        _require_bf16(dtype, "fused_qk_rmsnorm_gate")
        q_gate, k, q_weight, k_weight = _make_qk_inputs(
            self.num_tokens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            device,
            dtype,
        )
        return Workload(
            args=(
                q_gate,
                k,
                q_weight,
                k_weight,
                _EPS,
                self.num_query_heads,
                self.num_kv_heads,
                self.head_dim,
            ),
            meta={
                "T": self.num_tokens,
                "Hq": self.num_query_heads,
                "Hkv": self.num_kv_heads,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        _store_outputs(wl, fn(*wl.args))  # type: ignore[arg-type]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        _check_qk_gate(
            baseline_wl,
            flydsl_wl,
            qk_rtol=_GATE_QK_RTOL,
            qk_atol=_GATE_QK_ATOL,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


class FusedQkRmsnormRopeGateSuite:
    name = "fused_qk_rmsnorm_rope_gate"

    def resolve_providers(self) -> Providers:
        import vllm_iluvatar._C  # noqa: F401

        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="plugin CUDA fused_qk_rmsnorm_rope_gate",
            baseline=torch.ops._C.fused_qk_rmsnorm_rope_gate,
            flydsl=ixaite.fused_qk_rmsnorm_rope_gate,
            flydsl_name="ixaite.fused_qk_rmsnorm_rope_gate",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[FusedQkRmsnormRopeGateCase]:
        spec = l2_spec(model, self.name)
        cases = [
            FusedQkRmsnormRopeGateCase(
                case_id=(
                    f"fused_qk_rmsnorm_rope_gate@{model};T={num_tokens};"
                    f"Hq={spec['num_query_heads']};Hkv={spec['num_kv_heads']};"
                    f"D={spec['head_dim']};rotary={spec['rotary_dim']}"
                ),
                num_tokens=num_tokens,
                num_query_heads=int(spec["num_query_heads"]),
                num_kv_heads=int(spec["num_kv_heads"]),
                head_dim=int(spec["head_dim"]),
                rotary_dim=int(spec["rotary_dim"]),
                cache_rows=int(spec["cache_rows"]),
            )
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


class FusedQkRmsnormGateSuite:
    name = "fused_qk_rmsnorm_gate"

    def resolve_providers(self) -> Providers:
        import vllm_iluvatar._C  # noqa: F401

        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="plugin CUDA fused_qk_rmsnorm_gate",
            baseline=torch.ops._C.fused_qk_rmsnorm_gate,
            flydsl=ixaite.fused_qk_rmsnorm_gate,
            flydsl_name="ixaite.fused_qk_rmsnorm_gate",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[FusedQkRmsnormGateCase]:
        spec = l2_spec(model, self.name)
        cases = [
            FusedQkRmsnormGateCase(
                case_id=(
                    f"fused_qk_rmsnorm_gate@{model};T={num_tokens};"
                    f"Hq={spec['num_query_heads']};Hkv={spec['num_kv_heads']};"
                    f"D={spec['head_dim']}"
                ),
                num_tokens=num_tokens,
                num_query_heads=int(spec["num_query_heads"]),
                num_kv_heads=int(spec["num_kv_heads"]),
                head_dim=int(spec["head_dim"]),
            )
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(FusedQkRmsnormRopeGateSuite())
register(FusedQkRmsnormGateSuite())
