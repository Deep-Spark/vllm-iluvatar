# SPDX-License-Identifier: Apache-2.0
"""Dense unquantized linear suite (IXF.linear vs ixaite.linear)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.profiles import linear_required_t
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload

# Call sites whose activation reaches the linear through a row-aligned buffer.
_ROW_ALIGNED_ACTIVATION_SLUGS = frozenset({"mlp_down_proj"})

# Decode reads a different layer's weight on every call, so a timed unit that
# replays one weight can report a cache hit rate no deployment sees. On the
# k=12288 down_proj brick that flattered FlyDSL by 1.5x. Two weights are
# enough to reproduce the deployed number; the cap keeps replicas of the
# widest weights (lm_head) out of device memory.
_ROTATED_WEIGHTS = 2
_ROTATION_MAX_T = 16
_ROTATION_MAX_WEIGHT_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class LinearCase:
    case_id: str
    slug: str
    T: int
    K: int
    N: int
    bias: bool = False
    tags: tuple[str, ...] = ()

    def _make_activation(self, device: torch.device, dtype: torch.dtype):
        """Build the activation the way the deployed layer allocates it.

        down_proj is fed the SiluAndMul output, which ``IluSiluAndMul`` backs
        with a row-aligned buffer for short decode batches so the GEMM reads
        its brick in place. A tightly sized tensor here would instead charge
        every measurement a staged copy that production does not pay.
        """
        if self.slug in _ROW_ALIGNED_ACTIVATION_SLUGS:
            from vllm_iluvatar.backends.ixaite.layers.activation import _ACT_ROW_ALIGN

            if self.T < _ACT_ROW_ALIGN:
                buffer = torch.randn(_ACT_ROW_ALIGN, self.K, device=device, dtype=dtype)
                return buffer[: self.T]
        return torch.randn(self.T, self.K, device=device, dtype=dtype)

    @property
    def calls_per_run(self) -> int:
        """Operator calls in one timed unit (see ``_ROTATED_WEIGHTS``)."""
        if self.T > _ROTATION_MAX_T:
            return 1
        if self.N * self.K * 2 > _ROTATION_MAX_WEIGHT_BYTES:
            return 1
        return _ROTATED_WEIGHTS

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = self._make_activation(device, dtype)
        weight = torch.randn(self.N, self.K, device=device, dtype=dtype)
        bias = torch.randn(self.N, device=device, dtype=dtype) if self.bias else None
        cold = [
            torch.randn(self.N, self.K, device=device, dtype=dtype)
            for _ in range(self.calls_per_run - 1)
        ]
        return Workload(
            kwargs={"x": x, "weight": weight, "bias": bias, "cold_weights": cold},
            meta={
                "M": self.T,
                "N": self.N,
                "K": self.K,
                "flops": 2 * self.T * self.N * self.K,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        x = wl.kwargs["x"]
        bias = wl.kwargs["bias"]
        for cold in wl.kwargs["cold_weights"]:
            fn(x, cold, bias)
        # The primary weight goes last so the retained output is the one
        # ``check`` references.
        y = fn(x, wl.kwargs["weight"], bias)
        wl.outputs["y"] = y  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        del baseline_wl
        # Reference in fp32 with a single rounding. torch's bf16 F.linear adds
        # bias as a second pass over an already-rounded GEMM result, which
        # doubles its own error and penalises a provider that fuses bias into
        # the store.
        x = flydsl_wl.kwargs["x"]
        bias = flydsl_wl.kwargs["bias"]
        expected = F.linear(
            x.float(),
            flydsl_wl.kwargs["weight"].float(),
            None if bias is None else bias.float(),
        ).to(x.dtype)
        torch.testing.assert_close(
            flydsl_wl.outputs["y"],
            expected,
            rtol=1.6e-2,
            atol=1e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        flops = float(wl.meta["flops"])
        tflops = flops / (median_ms * 1e-3) / 1e12
        return tflops, "TFLOPS"


class LinearSuite:
    name = "linear"

    def __init__(self, *, full_profile: bool = False) -> None:
        self._full_profile = full_profile

    def resolve_providers(self) -> Providers:
        import ixaite
        import ixformer.inference.functions as ixf_ops

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_torch_op
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)
        fly, reason = load_ixaite_torch_op("linear")
        if fly is None:
            raise RuntimeError(
                f"ixaite linear is not available yet ({reason}); "
                "implement ixaite.ops.linear before gating this suite"
            )

        def baseline(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            return ixf_ops.linear(x, weight, bias)

        return Providers(
            baseline_name="IXF.linear",
            baseline=baseline,
            flydsl=fly,
            flydsl_name="ixaite.linear",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[LinearCase]:
        specs = l2_spec(model, self.name)

        t_values = linear_required_t(full_profile=self._full_profile)
        cases: list[LinearCase] = []
        for slug, spec in specs.items():
            tags = tuple(spec.get("tags", ()))  # type: ignore[arg-type]
            max_t = spec.get("max_t")
            for t in t_values:
                if max_t is not None and t > int(max_t):  # type: ignore[arg-type]
                    continue
                cases.append(
                    LinearCase(
                        case_id=f"{slug}@{model};T={t}",
                        slug=slug,
                        T=t,
                        K=int(spec["K"]),
                        N=int(spec["N"]),
                        bias=bool(spec["bias"]),
                        tags=tags,
                    )
                )
        if case_filter:
            cases = [case for case in cases if case_filter in case.case_id]
        return cases

    def gate_min_speedup(self) -> float:
        return 1.0


# Default registration uses the first-version required T set.
register(LinearSuite(full_profile=False))
