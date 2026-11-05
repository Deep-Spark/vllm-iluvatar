# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 GDN chunk suite (ixformer infer vs ixaite FlyDSL)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


def _ixformer_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> None:
    import ixformer._C as ixf_c_ops

    ixf_c_ops.infer.chunk_gated_delta_rule(
        q,
        k,
        v,
        out,
        g,
        beta,
        state,
        has_initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        state_indices,
        scale,
        None,
    )


def _adapt_ixaite_chunk(implementation: Callable[..., object]) -> Callable[..., None]:
    def run(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float,
    ) -> None:
        chunk_out, final_state = implementation(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=state,
            initial_state_indices=state_indices,
            has_initial_state=has_initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=64,
            use_qk_l2norm_in_kernel=False,
            out=out,
        )
        if chunk_out is not out:
            out.copy_(chunk_out)
        if final_state is None:
            raise RuntimeError("ixaite chunk did not return final_state")

    return run


@dataclass(frozen=True)
class ChunkGatedDeltaRuleCase:
    case_id: str
    lengths: tuple[int, ...]
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        total = sum(self.lengths)
        q = torch.randn(
            total, self.num_k_heads, self.head_k_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        q.div_(q.float().norm(dim=-1, keepdim=True).to(dtype))
        k.div_(k.float().norm(dim=-1, keepdim=True).to(dtype))
        v = torch.randn(
            total, self.num_v_heads, self.head_v_dim, device=device, dtype=dtype
        )
        out = torch.empty_like(v)
        g = -torch.rand(total, self.num_v_heads, device=device, dtype=torch.float32)
        beta = torch.rand(total, self.num_v_heads, device=device, dtype=dtype)
        state = torch.randn(
            len(self.lengths) + 2,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=torch.float32,
        )
        has_initial_state = torch.ones(
            len(self.lengths), device=device, dtype=torch.int32
        )
        bounds = [0]
        for length in self.lengths:
            bounds.append(bounds[-1] + length)
        cu_seqlens = torch.tensor(bounds, device=device, dtype=torch.int32)
        cu_seqlens_cpu = torch.tensor(bounds, device="cpu", dtype=torch.int32)
        state_indices = torch.arange(
            len(self.lengths), 0, -1, device=device, dtype=torch.int32
        )
        return Workload(
            args=(
                q,
                k,
                v,
                out,
                g,
                beta,
                state,
                has_initial_state,
                cu_seqlens,
                cu_seqlens_cpu,
                state_indices,
                self.head_k_dim**-0.5,
            )
        )

    def run(self, fn: Callable[..., None], wl: Workload) -> None:
        fn(*wl.args)
        wl.outputs["y"] = wl.args[3]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["y"], baseline_wl.outputs["y"], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            flydsl_wl.args[6], baseline_wl.args[6], rtol=3e-2, atol=3e-2
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


class ChunkGatedDeltaRuleSuite:
    name = "chunk_gated_delta_rule"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="ixformer infer chunk",
            baseline=_ixformer_chunk,
            flydsl=_adapt_ixaite_chunk(ixaite.chunk_gated_delta_rule),
            flydsl_name="ixaite.chunk_gated_delta_rule",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[ChunkGatedDeltaRuleCase]:
        specs = l2_spec(model, self.name)
        lengths = [tuple(int(value) for value in item) for item in specs["lengths"]]
        cases = [
            ChunkGatedDeltaRuleCase(
                (
                    f"gdn_{'dense' if len(item) == 1 else 'packed'}@{model};"
                    f"lengths={'+'.join(str(value) for value in item)}"
                ),
                item,
                int(specs["num_k_heads"]),
                int(specs["head_k_dim"]),
                int(specs["num_v_heads"]),
                int(specs["head_v_dim"]),
                ("gdn", "prefill", "dense" if len(item) == 1 else "packed"),
            )
            for item in lengths
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(ChunkGatedDeltaRuleSuite())
