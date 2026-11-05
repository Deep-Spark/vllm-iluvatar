# SPDX-License-Identifier: Apache-2.0
"""GDN post-conv prep suite (plugin three-stage path vs ixaite FlyDSL)."""

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


def _plugin_three_stage(
    conv_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    apply_l2norm: bool,
) -> tuple[torch.Tensor, ...]:
    from vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn import (
        fused_gdn_gating,
        l2norm_fwd,
    )

    num_tokens = conv_out.shape[0]
    qk_size = num_k_heads * head_k_dim
    q = conv_out[:, :qk_size].reshape(num_tokens, num_k_heads, head_k_dim)
    k = conv_out[:, qk_size : 2 * qk_size].reshape(num_tokens, num_k_heads, head_k_dim)
    v = conv_out[:, 2 * qk_size :].reshape(num_tokens, a.shape[-1], head_v_dim)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    if apply_l2norm:
        l2norm_fwd(q, k)
    return q, k, v, g, beta


@dataclass(frozen=True)
class FusedPostConvPrepCase:
    case_id: str
    t: int
    apply_l2norm: bool
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    tags: tuple[str, ...] = ("gdn", "fusion")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        packed_dim = (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )
        conv_out = torch.randn(self.t, packed_dim, device=device, dtype=dtype)
        ba = torch.randn(self.t, 2 * self.num_v_heads, device=device, dtype=dtype)
        b, a = ba[:, : self.num_v_heads], ba[:, self.num_v_heads :]
        A_log = torch.randn(self.num_v_heads, device=device, dtype=torch.float32)
        dt_bias = torch.randn(self.num_v_heads, device=device, dtype=torch.float32)
        # Both providers hand back head-slice views of conv_out, so neither
        # allocates: preallocating an out tuple would charge only one side for
        # a copy the GDN layer never asks for.
        return Workload(args=(conv_out, a, b, A_log, dt_bias))

    def run(self, fn: Callable[..., tuple[torch.Tensor, ...]], wl: Workload) -> None:
        conv_out, a, b, A_log, dt_bias = wl.args
        result = fn(
            conv_out,
            a,
            b,
            A_log,
            dt_bias,
            self.num_k_heads,
            self.head_k_dim,
            self.head_v_dim,
            apply_l2norm=self.apply_l2norm,
        )
        for index, tensor in enumerate(result):
            wl.outputs[str(index)] = tensor

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        for index in range(5):
            rtol, atol = (1e-4, 1e-5) if index == 3 else (1e-2, 1e-2)
            torch.testing.assert_close(
                flydsl_wl.outputs[str(index)],
                baseline_wl.outputs[str(index)],
                rtol=rtol,
                atol=atol,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


class FusedPostConvPrepSuite:
    name = "fused_post_conv_prep"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="plugin rearrange+Triton gating+l2norm",
            baseline=_plugin_three_stage,
            flydsl=ixaite.fused_post_conv_prep,
            flydsl_name="ixaite.fused_post_conv_prep",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[FusedPostConvPrepCase]:
        spec = l2_spec(model, self.name)
        l2norm_min_t = int(spec["l2norm_min_t"])
        cases = [
            FusedPostConvPrepCase(
                f"gdn_post_conv@{model};T={t};l2norm={t >= l2norm_min_t}",
                t,
                apply_l2norm=t >= l2norm_min_t,
                num_k_heads=int(spec["num_k_heads"]),
                head_k_dim=int(spec["head_k_dim"]),
                num_v_heads=int(spec["num_v_heads"]),
                head_v_dim=int(spec["head_v_dim"]),
            )
            for t in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(FusedPostConvPrepSuite())
