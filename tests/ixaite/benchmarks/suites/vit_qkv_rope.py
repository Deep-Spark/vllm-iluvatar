# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 Vision packed-QKV split plus NeoX RoPE suite."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite
from tests.ixaite.benchmarks.suites.vit_qkv_layout import _einops_split


def _ref_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot_dim = cos.shape[-1] * 2
    x_rot = x[..., :rot_dim].float()
    x_pass = x[..., rot_dim:]
    cos_ = cos.unsqueeze(-2).float()
    sin_ = sin.unsqueeze(-2).float()
    x1, x2 = x_rot.chunk(2, dim=-1)
    o1 = x1 * cos_ - x2 * sin_
    o2 = x2 * cos_ + x1 * sin_
    return torch.cat((torch.cat((o1, o2), dim=-1).to(x.dtype), x_pass), dim=-1)


def _einops_split_and_rope(
    packed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = _einops_split(packed, num_heads, head_dim)
    return _ref_neox(q, cos, sin), _ref_neox(k, cos, sin), v


@dataclass(frozen=True)
class VitQkvRopeCase:
    case_id: str
    seq_len: int
    num_heads: int
    head_dim: int
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"vit_qkv_rope models the BF16 serving path, got {dtype}")
        packed = torch.randn(
            (self.seq_len, 1, 3 * self.num_heads * self.head_dim),
            device=device,
            dtype=dtype,
        )
        rot_half = self.head_dim // 4
        angles = torch.randn(
            (self.seq_len, rot_half), device=device, dtype=torch.float32
        )
        cos = angles.cos()
        sin = angles.sin()
        return Workload(
            args=(packed, cos, sin, self.num_heads, self.head_dim),
            meta={
                "S": self.seq_len,
                "H": self.num_heads,
                "D": self.head_dim,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["q"], wl.outputs["k"], wl.outputs["v"] = fn(*wl.args)

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        for name in ("q", "k", "v"):
            torch.testing.assert_close(
                flydsl_wl.outputs[name],
                baseline_wl.outputs[name],
                rtol=2e-2,
                atol=2e-2,
            )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        elements = wl.args[0].numel()
        return elements / (median_ms * 1e-3) / 1e9, "GElements/s"


class VitQkvRopeSuite:
    name = "vit_qkv_rope"

    def resolve_providers(self) -> Providers:
        ixaite = initialize_ixaite()
        return Providers(
            baseline_name="eager split + NeoX RoPE",
            baseline=_einops_split_and_rope,
            flydsl=ixaite.vit_qkv_rope,
            flydsl_name="ixaite.vit_qkv_rope",
        )

    def iter_cases(
        self,
        model: str,
        case_filter: str | None,
    ) -> Sequence[VitQkvRopeCase]:
        spec = l2_spec(model, self.name)
        head_dim = int(spec["head_dim"])
        cases = []
        for seq_len in spec["sequence_lengths"]:
            for tp_index, num_heads in enumerate(spec["num_heads"]):
                tp = 1 << tp_index
                cases.append(
                    VitQkvRopeCase(
                        case_id=(
                            f"vit_qkv_rope@{model};S={int(seq_len)};"
                            f"H={int(num_heads)};D={head_dim};tp={tp}"
                        ),
                        seq_len=int(seq_len),
                        num_heads=int(num_heads),
                        head_dim=head_dim,
                        tags=("vision", "rope", "bf16", f"tp{tp}"),
                    )
                )
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 0.0


register(VitQkvRopeSuite())
