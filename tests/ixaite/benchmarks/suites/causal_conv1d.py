# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 GDN causal-conv suites (plugin Triton vs ixaite FlyDSL)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import (
    DECODE_GRAPH_T,
    PREFILL_T,
    filter_cases,
    initialize_ixaite,
)


@dataclass(frozen=True)
class _ConvMetadata:
    """The GDN metadata fields both conv prefill paths consume.

    Without it each provider rebuilds the token-chunk schedule from a host
    copy of query_start_loc on every call. That host work dominates a prefill
    measurement, leaving the kernels it is meant to compare as a minor term,
    and its D2H sync also makes the call impossible to capture in a graph. The
    served path always supplies this metadata, so the suite does too.
    """

    nums_dict: dict
    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor


@dataclass(frozen=True)
class CausalConv1dFnCase:
    case_id: str
    t: int
    dim: int
    width: int
    tags: tuple[str, ...] = ("conv1d", "prefill", "gdn")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        x = torch.randn(self.t, self.dim, device=device, dtype=dtype).transpose(0, 1)
        weight = torch.randn(self.dim, self.width, device=device, dtype=dtype)
        bias = torch.randn(self.dim, device=device, dtype=dtype)
        state = torch.randn(
            1, self.width - 1, self.dim, device=device, dtype=dtype
        ).transpose(1, 2)
        query_start_loc = torch.tensor([0, self.t], device=device, dtype=torch.int32)
        cache_indices = torch.zeros(1, device=device, dtype=torch.int32)
        has_initial_state = torch.ones(1, device=device, dtype=torch.int32)
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            query_start_loc.to("cpu"), device=device
        )
        return Workload(
            args=(
                x,
                weight,
                bias,
                state,
                query_start_loc,
                cache_indices,
                has_initial_state,
            ),
            meta={
                "metadata": _ConvMetadata(nums_dict, batch_ptr, token_chunk_offset_ptr)
            },
        )

    def run(self, fn: Callable[..., torch.Tensor], wl: Workload) -> None:
        x, weight, bias, state, query_start_loc, cache_indices, has_initial_state = (
            wl.args
        )
        wl.outputs["y"] = fn(
            x,
            weight,
            bias,
            state,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
            metadata=wl.meta["metadata"],
        )

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["y"], baseline_wl.outputs["y"], rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            flydsl_wl.args[3], baseline_wl.args[3], rtol=1e-2, atol=1e-2
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


@dataclass(frozen=True)
class CausalConv1dUpdateCase:
    case_id: str
    batch: int
    dim: int
    width: int
    tags: tuple[str, ...] = ("conv1d", "decode", "gdn")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = torch.randn(self.batch, self.dim, device=device, dtype=dtype)
        weight = torch.randn(self.dim, self.width, device=device, dtype=dtype)
        bias = torch.randn(self.dim, device=device, dtype=dtype)
        state = torch.randn(
            self.batch, self.width - 1, self.dim, device=device, dtype=dtype
        ).transpose(1, 2)
        state_indices = torch.arange(self.batch, device=device, dtype=torch.int32)
        return Workload(
            args=(x, state, weight, bias, state_indices),
            meta={"x0": x.clone(), "state0": state.clone()},
        )

    def run(self, fn: Callable[..., torch.Tensor], wl: Workload) -> None:
        x, state, weight, bias, state_indices = wl.args
        wl.outputs["y"] = fn(
            x,
            state,
            weight,
            bias,
            "silu",
            conv_state_indices=state_indices,
            validate_data=False,
        )

    # A decode step consumes x and conv_state and overwrites both (out aliases
    # x), so back-to-back replays feed each step its own output and diverge to
    # inf/NaN on both providers within the timing loop. Restore the captured
    # inputs so the post-timing replay is one well-defined step.
    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[0].copy_(wl.meta["x0"])
        wl.args[1].copy_(wl.meta["state0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["y"], baseline_wl.outputs["y"], rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            flydsl_wl.args[1], baseline_wl.args[1], rtol=1e-2, atol=1e-2
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> None:
        del median_ms, wl
        return None


class CausalConv1dFnSuite:
    name = "causal_conv1d_fn"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar.model_executor.layers.mamba.ops import ilu_causal_conv1d

        return Providers(
            baseline_name="plugin Triton causal_conv1d_fn",
            baseline=ilu_causal_conv1d._causal_conv1d_fn_triton,
            flydsl=ixaite.causal_conv1d_fn,
            flydsl_name="ixaite.causal_conv1d_fn",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[CausalConv1dFnCase]:
        spec = l2_spec(model, self.name)
        return filter_cases(
            [
                CausalConv1dFnCase(
                    f"gdn_prefill@{model};T={t}",
                    t,
                    dim=int(spec["dim"]),
                    width=int(spec["width"]),
                )
                for t in PREFILL_T
            ],
            case_filter,
        )

    def gate_min_speedup(self) -> float:
        return 1.0


class CausalConv1dUpdateSuite:
    name = "causal_conv1d_update"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar.model_executor.layers.mamba.ops import ilu_causal_conv1d

        return Providers(
            baseline_name="plugin Triton causal_conv1d_update",
            baseline=ilu_causal_conv1d._causal_conv1d_update_triton,
            flydsl=ixaite.causal_conv1d_update,
            flydsl_name="ixaite.causal_conv1d_update",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[CausalConv1dUpdateCase]:
        spec = l2_spec(model, self.name)
        return filter_cases(
            [
                CausalConv1dUpdateCase(
                    f"gdn_decode@{model};T={t}",
                    t,
                    dim=int(spec["dim"]),
                    width=int(spec["width"]),
                )
                for t in DECODE_GRAPH_T
            ],
            case_filter,
        )

    def gate_min_speedup(self) -> float:
        return 1.0


register(CausalConv1dFnSuite())
register(CausalConv1dUpdateSuite())
