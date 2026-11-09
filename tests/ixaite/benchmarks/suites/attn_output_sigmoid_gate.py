# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 attention-output sigmoid-gate suite (Torch vs ixaite)."""

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


def _torch_gate(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    del out
    return attn_output * torch.sigmoid(gate)


@dataclass(frozen=True)
class AttnOutputSigmoidGateCase:
    case_id: str
    num_tokens: int
    hidden_size: int
    tags: tuple[str, ...] = ("attention", "activation", "bf16", "tp1")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"attn_output_sigmoid_gate models the BF16 serving path, got {dtype}"
            )
        attn_output = torch.randn(
            self.num_tokens,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        gate = torch.randn_like(attn_output)
        return Workload(
            args=(attn_output, gate, attn_output),
            meta={"T": self.num_tokens, "D": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["result"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["result"],
            baseline_wl.outputs["result"],
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        num_bytes = self.num_tokens * self.hidden_size * wl.args[0].element_size() * 3
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class AttnOutputSigmoidGateSuite:
    name = "attn_output_sigmoid_gate"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="torch.mul_sigmoid",
            baseline=_torch_gate,
            flydsl=ixaite.attn_output_sigmoid_gate,
            flydsl_name="ixaite.attn_output_sigmoid_gate",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[AttnOutputSigmoidGateCase]:
        spec = l2_spec(model, self.name)
        hidden_size = int(spec["hidden_size"])
        cases = [
            AttnOutputSigmoidGateCase(
                case_id=(
                    f"attn_output_sigmoid_gate@{model};T={num_tokens};D={hidden_size}"
                ),
                num_tokens=num_tokens,
                hidden_size=hidden_size,
            )
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(AttnOutputSigmoidGateSuite())
