# SPDX-License-Identifier: Apache-2.0
"""GDN speculative multi-token delta update cross-provider suite."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload


@dataclass(frozen=True)
class GdnSpecDeltaUpdateCase:
    case_id: str
    num_seqs: int
    tokens_per_seq: int
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    tags: tuple[str, ...] = ("gdn", "decode", "speculative")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        num_tokens = self.num_seqs * self.tokens_per_seq
        gen = torch.Generator(device=device).manual_seed(
            self.num_seqs * 100 + self.tokens_per_seq
        )
        q = torch.randn(
            (num_tokens, self.num_k_heads, self.head_k_dim),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            (num_tokens, self.num_v_heads, self.head_v_dim),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        g = -torch.rand(
            (num_tokens, self.num_v_heads),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        beta = torch.rand(
            (num_tokens, self.num_v_heads),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        out = torch.empty(
            (num_tokens, self.num_v_heads, self.head_v_dim),
            device=device,
            dtype=dtype,
        )
        state = torch.randn(
            (
                num_tokens + 1,
                self.num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        cu_seqlens = torch.arange(
            0,
            num_tokens + 1,
            self.tokens_per_seq,
            device=device,
            dtype=torch.int32,
        )
        state_indices = torch.arange(
            1, num_tokens + 1, device=device, dtype=torch.int32
        ).view(self.num_seqs, self.tokens_per_seq)
        num_accepted_tokens = (
            torch.arange(self.num_seqs, device=device, dtype=torch.int32)
            % self.tokens_per_seq
        ) + 1
        return Workload(
            args=(
                q,
                k,
                v,
                out,
                g,
                beta,
                state,
                cu_seqlens,
                state_indices,
                self.head_k_dim**-0.5,
                True,
                num_accepted_tokens,
                None,
            ),
            outputs={"out": out, "state": state},
            meta={"num_tokens": num_tokens, "state0": state.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)
        wl.outputs["out"] = wl.args[3]
        wl.outputs["state"] = wl.args[6]

    # Spec delta writes the SSM cache in place. Restore the captured state
    # before the post-timing replay so that last step is one defined update.
    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[6].copy_(wl.meta["state0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"],
            baseline_wl.outputs["out"],
            rtol=1e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["state"].float(),
            baseline_wl.outputs["state"].float(),
            rtol=1e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        elem = wl.outputs["state"].element_size()
        num_bytes = (
            wl.meta["num_tokens"]
            * 2
            * self.num_v_heads
            * self.head_v_dim
            * self.head_k_dim
            * elem
        )
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class GdnSpecDeltaUpdateSuite:
    name = "gdn_spec_delta_update"

    def resolve_providers(self) -> Providers:
        import ixaite
        import ixformer._C as ixf_c_ops

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)
        fly, reason = load_ixaite_python_op(
            "gdn_spec_delta_update", "fused_recurrent_gated_delta_rule"
        )
        if fly is None:
            raise RuntimeError(reason)
        return Providers(
            baseline_name="ixformer",
            baseline=ixf_c_ops.infer.fused_recurrent_gated_delta_rule,
            flydsl=fly,
            flydsl_name="ixaite.gdn_spec_delta_update",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[GdnSpecDeltaUpdateCase]:
        spec = l2_spec(model, self.name)
        cases = [
            GdnSpecDeltaUpdateCase(
                case_id=f"gdn_spec_delta_update@{model};B={num_seqs};T={tokens_per_seq}",
                num_seqs=num_seqs,
                tokens_per_seq=tokens_per_seq,
                num_k_heads=int(spec["num_k_heads"]),
                head_k_dim=int(spec["head_k_dim"]),
                num_v_heads=int(spec["num_v_heads"]),
                head_v_dim=int(spec["head_v_dim"]),
            )
            for num_seqs, tokens_per_seq in (
                (int(item[0]), int(item[1])) for item in spec["shapes"]
            )
        ]
        if case_filter:
            cases = [case for case in cases if case_filter in case.case_id]
        return cases

    def gate_min_speedup(self) -> float:
        return 1.0


register(GdnSpecDeltaUpdateSuite())
