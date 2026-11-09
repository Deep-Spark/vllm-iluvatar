# SPDX-License-Identifier: Apache-2.0
"""fused_recurrent_gated_delta_rule_packed_decode cross-provider suite.

Baseline is the community FLA Triton packed decode (what branch A of the GDN
attention calls today); the FlyDSL side is the ixaite op bound through the
plugin provider. Both share the same ABI, so one workload drives both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload


@dataclass(frozen=True)
class GdnPackedDecodeCase:
    case_id: str
    num_tokens: int
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    tags: tuple[str, ...] = ("gdn", "decode")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        qkv_dim = (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )
        gen = torch.Generator(device=device).manual_seed(self.num_tokens)
        mixed_qkv = torch.randn(
            (self.num_tokens, qkv_dim), device=device, dtype=dtype, generator=gen
        )
        a = torch.randn(
            (self.num_tokens, self.num_v_heads),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        b = torch.randn_like(a, generator=gen)
        A_log = torch.randn(
            (self.num_v_heads,), device=device, dtype=torch.float32, generator=gen
        )
        dt_bias = torch.randn_like(A_log, generator=gen)
        # Slot 0 is the CUDA-graph NULL row; valid slots start at 1.
        indices = torch.arange(1, self.num_tokens + 1, device=device, dtype=torch.int32)
        state = torch.randn(
            (
                self.num_tokens + 1,
                self.num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        out = torch.empty(
            (self.num_tokens, 1, self.num_v_heads, self.head_v_dim),
            device=device,
            dtype=dtype,
        )
        return Workload(
            args=(mixed_qkv, a, b, A_log, dt_bias),
            kwargs={
                "scale": self.head_k_dim**-0.5,
                "initial_state": state,
                "out": out,
                "ssm_state_indices": indices,
                "use_qk_l2norm_in_kernel": True,
            },
            meta={"num_tokens": self.num_tokens, "state0": state.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args, **wl.kwargs)

    # Packed decode reads and overwrites the SSM cache slot. Back-to-back
    # graph replays therefore accumulate recurrence; BF16 drift vs FLA then
    # fails the post-timing check. Restore the captured state so that last
    # replay is one well-defined step, same as causal_conv1d_update.
    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.kwargs["initial_state"].copy_(wl.meta["state0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.kwargs["out"],
            baseline_wl.kwargs["out"],
            rtol=1e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.kwargs["initial_state"].float(),
            baseline_wl.kwargs["initial_state"].float(),
            rtol=1e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        # Effective bandwidth: the state tile dominates (read + write back),
        # plus the packed qkv row read and the output row write.
        elem = wl.kwargs["initial_state"].element_size()
        qkv_dim = (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )
        num_bytes = self.num_tokens * (
            2 * self.num_v_heads * self.head_v_dim * self.head_k_dim * elem
            + qkv_dim * elem
            + self.num_v_heads * self.head_v_dim * elem
        )
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class GdnPackedDecodeSuite:
    name = "gdn_packed_decode"

    def resolve_providers(self) -> Providers:
        import ixaite  # noqa: F401 -- initialize FlyDSL before vllm imports it
        from vllm.third_party.flash_linear_attention.ops import (
            fused_recurrent_gated_delta_rule_packed_decode as fla_packed_decode,
        )

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)
        fly, reason = load_ixaite_python_op(
            "gdn_packed_decode", "fused_recurrent_gated_delta_rule_packed_decode"
        )
        if fly is None:
            raise RuntimeError(reason)
        return Providers(
            baseline_name="fla_triton",
            baseline=fla_packed_decode,
            flydsl=fly,
            flydsl_name="ixaite.gdn_packed_decode",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[GdnPackedDecodeCase]:
        spec = l2_spec(model, self.name)
        cases = [
            GdnPackedDecodeCase(
                case_id=f"gdn_packed_decode@{model};B={num_tokens}",
                num_tokens=num_tokens,
                num_k_heads=int(spec["num_k_heads"]),
                head_k_dim=int(spec["head_k_dim"]),
                num_v_heads=int(spec["num_v_heads"]),
                head_v_dim=int(spec["head_v_dim"]),
            )
            for num_tokens in (int(value) for value in spec["num_tokens"])
        ]
        if case_filter:
            cases = [case for case in cases if case_filter in case.case_id]
        return cases

    def gate_min_speedup(self) -> float:
        return 1.0


register(GdnPackedDecodeSuite())
