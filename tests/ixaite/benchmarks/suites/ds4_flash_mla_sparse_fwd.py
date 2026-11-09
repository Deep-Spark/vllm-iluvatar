# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 sparse MLA prefill (ixformer default vs ixaite)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

_HEADS = 8
_HEAD_DIM = 512
_PREFILL_T = (8, 128, 512)


@dataclass(frozen=True)
class Ds4SparseMlaCase:
    case_id: str
    t: int
    topk: int
    s_kv: int
    use_topk_length: bool = False
    use_sink: bool = False
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "prefill")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 sparse MLA models the BF16 serving path, got {dtype}")
        s_kv = ((self.s_kv + 31) // 32) * 32
        q = torch.randn((self.t, _HEADS, _HEAD_DIM), device=device, dtype=dtype)
        kv = torch.randn((s_kv, 1, _HEAD_DIM), device=device, dtype=dtype)
        indices = torch.randint(0, s_kv, (self.t, 1, self.topk), device=device, dtype=torch.int32)
        if self.topk >= 4:
            indices[:, 0, -1] = -1
        output = torch.empty_like(q)
        sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
        topk_length = None
        if self.use_topk_length:
            topk_length = torch.clamp(
                torch.arange(self.t, device=device, dtype=torch.int32) // 4
                + torch.arange(self.t, device=device, dtype=torch.int32)
                + 1,
                max=self.topk,
            )
        attn_sink = None
        if self.use_sink:
            attn_sink = torch.zeros((_HEADS,), device=device, dtype=torch.float32)
            attn_sink[0] = float("-inf")
        return Workload(
            args=(q, kv, indices, sm_scale),
            kwargs={
                "d_v": _HEAD_DIM,
                "attn_sink": attn_sink,
                "topk_length": topk_length,
                "output": output,
            },
            meta={"T": self.t, "topk": self.topk, "s_kv": s_kv, "out0": output.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["out"] = fn(*wl.args, **wl.kwargs)  # type: ignore[misc]

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.kwargs["output"].copy_(wl.meta["out0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        indices = wl.args[2]
        bytes_moved = (q.numel() * 2 + indices.numel()) * q.element_size()
        return bytes_moved / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4FlashMlaSparseFwdSuite:
    name = "ds4_flash_mla_sparse_fwd"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_flash_mla_sparse_fwd import (
            ds4_flash_mla_sparse_fwd as ixaite_op,
        )

        return Providers(
            baseline_name="IXF.ds4_flash_mla_sparse_fwd",
            baseline=ixf_ops.ds4_flash_mla_sparse_fwd,
            flydsl=ixaite_op,
            flydsl_name="ixaite.ds4_flash_mla_sparse_fwd",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4SparseMlaCase]:
        del model
        cases = [
            Ds4SparseMlaCase(
                f"ds4_flash_mla_sparse_fwd@0731;tok={t:04d};topk=0064",
                t,
                64,
                max(t * 4, 128),
            )
            for t in _PREFILL_T
        ]
        cases.extend(
            [
                Ds4SparseMlaCase(
                    f"ds4_flash_mla_sparse_fwd@0731;tok={t:04d};topk=0256;tlen",
                    t,
                    256,
                    512,
                    use_topk_length=True,
                )
                for t in (128, 512)
            ]
        )
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4FlashMlaSparseFwdSuite())
