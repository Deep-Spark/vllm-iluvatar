# SPDX-License-Identifier: Apache-2.0
"""ViT LayerNorm suite (ixformer vs ixaite FlyDSL)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

# Vision-tower row counts, taken from the resolutions the qwen35_9b_vision
# e2e profile serves. patch_size is 16, so one image is (px / 16) ** 2 patch
# rows: 448 -> 784, 896 -> 3136. The encoder batches the images in a
# scheduler step, so concurrency multiplies the row count -- 12544 is the
# profile's four concurrent 896x896 requests.
VISION_T: tuple[int, ...] = (784, 3136, 12544)


@dataclass(frozen=True)
class LayerNormCase:
    case_id: str
    t: int
    hidden: int
    tags: tuple[str, ...] = ("norm", "vision")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        x = torch.randn((self.t, self.hidden), device=device, dtype=dtype)
        weight = torch.randn((self.hidden,), device=device, dtype=dtype)
        bias = torch.randn((self.hidden,), device=device, dtype=dtype)
        return Workload(
            args=(x, weight, bias, 1e-6),
            meta={"T": self.t, "H": self.hidden},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args)  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        # x reads and out writes, plus the shared weight and bias rows.
        elem = wl.args[0].element_size()
        num_bytes = (2 * self.t * self.hidden + 2 * self.hidden) * elem
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class LayerNormSuite:
    name = "layer_norm"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        from vllm_iluvatar import _custom_ops as ilu_ops

        return Providers(
            baseline_name="IXF.layer_norm",
            baseline=ilu_ops.layer_norm,
            flydsl=ixaite.layer_norm,
            flydsl_name="ixaite.layer_norm",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[LayerNormCase]:
        spec = l2_spec(model, self.name)
        hidden = int(spec["hidden_size"])
        cases = [
            LayerNormCase(f"vis_block_norm@{model};T={t}", t, hidden) for t in VISION_T
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(LayerNormSuite())
