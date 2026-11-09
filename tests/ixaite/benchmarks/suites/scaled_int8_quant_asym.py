# SPDX-License-Identifier: Apache-2.0
"""W8A8 asymmetric activation INT8 quant suite (torch eager vs ixaite).

The baseline is torch eager rather than ixformer because ixformer has no
asymmetric INT8 quant: ``dynamic_scaled_int8_quant`` takes only
``(output, input, scales)``, so the zero point can only be computed on the
host. That absence is the reason this ixaite path exists.
"""

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

_IMAX = 127
_IMIN = -128


def _torch_dynamic_asym(x: torch.Tensor):
    """Per-token min/max -> uint8-range scale and int32 zero point."""
    f = x.float()
    x_min = f.amin(dim=-1, keepdim=True)
    x_max = f.amax(dim=-1, keepdim=True)
    scales = (x_max - x_min) / (_IMAX - _IMIN)
    # Round the whole zero-point expression, not just the quotient; rounding
    # min/scale first shifts azp by an LSB and the quant by two.
    azp = torch.round(float(_IMIN) - x_min / scales)
    quant = torch.clamp(torch.round(f / scales) + azp, _IMIN, _IMAX).to(torch.int8)
    return quant, scales, azp.to(torch.int32)


@dataclass(frozen=True)
class ScaledInt8QuantAsymCase:
    case_id: str
    num_tokens: int
    hidden_size: int
    tags: tuple[str, ...] = ("quant", "int8", "w8a8", "azp")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("scaled_int8_quant models FP16/BF16 activations")
        # Offset the distribution so the zero point is not incidentally ~0 and
        # a dropped AZP term would still look correct.
        x = (
            torch.randn(self.num_tokens, self.hidden_size, device=device, dtype=dtype)
            + 2.0
        )
        return Workload(
            args=(x,),
            meta={"T": self.num_tokens, "K": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        quant, scales, azp = fn(*wl.args)
        wl.outputs["quant"] = quant
        wl.outputs["scales"] = scales
        wl.outputs["azp"] = azp

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["scales"].reshape(-1),
            baseline_wl.outputs["scales"].reshape(-1),
            rtol=1e-4,
            atol=1e-7,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["azp"].reshape(-1),
            baseline_wl.outputs["azp"].reshape(-1),
            atol=1,
            rtol=0.0,
        )
        # Codes are not comparable bit-for-bit: the two scales differ in the last
        # fp32 bits, so a row whose zero point sits on a .5 tie rounds the other
        # way and shifts that whole row's codes by one. Each (code, zero point)
        # pair is still self-consistent, so compare what W8A8 consumes -- the
        # dequantized value, which must stay within half a quantization step.
        scales = flydsl_wl.outputs["scales"].reshape(-1, 1)
        azp = flydsl_wl.outputs["azp"].reshape(-1, 1)
        deq = (flydsl_wl.outputs["quant"].float() - azp.float()) * scales
        steps = (deq - flydsl_wl.args[0].float()).abs() / scales
        max_steps = float(steps.max())
        if max_steps > 0.5 + 1e-4:
            raise AssertionError(
                f"dequantized round-trip off by {max_steps:.4f} quantization "
                "steps, expected <= 0.5"
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        num_bytes = self.num_tokens * self.hidden_size * (
            wl.args[0].element_size() + 1
        ) + self.num_tokens * 8
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class ScaledInt8QuantAsymSuite:
    name = "scaled_int8_quant_asym"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()

        def ixaite_dynamic_asym(x: torch.Tensor):
            return ixaite.scaled_int8_quant(x, symmetric=False)

        return Providers(
            baseline_name="torch.dynamic_asym_int8_quant",
            baseline=_torch_dynamic_asym,
            flydsl=ixaite_dynamic_asym,
            flydsl_name="ixaite.scaled_int8_quant[asym]",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[ScaledInt8QuantAsymCase]:
        spec = l2_spec(model, self.name)
        hidden_sizes = [int(k) for k in spec["hidden_sizes"]]
        cases = [
            ScaledInt8QuantAsymCase(
                case_id=f"{self.name}@{model};T={num_tokens};K={hidden}",
                num_tokens=num_tokens,
                hidden_size=hidden,
            )
            for hidden in hidden_sizes
            for num_tokens in RUNTIME_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(ScaledInt8QuantAsymSuite())
