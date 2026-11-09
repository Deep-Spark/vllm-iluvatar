# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 SWA paged MLA attention (ixformer vs ixaite)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

HEAD_DIM = 512
HEADS = 8
BLOCK = 64
WINDOW = 512


def _make_tensors(device, q_len: int, kv_len: int, with_sink: bool):
    max_blocks = (kv_len + BLOCK - 1) // BLOCK
    q = torch.randn((q_len, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    kv_cache = torch.randn((max_blocks, BLOCK, HEAD_DIM), device=device, dtype=torch.bfloat16)
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    cu_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    sink = torch.randn((HEADS,), device=device, dtype=torch.float32) if with_sink else None
    out = torch.empty_like(q)
    return q, kv_cache, block_table, cu_q, cu_k, sink, out


@dataclass(frozen=True)
class Ds4SwaAttnCase:
    case_id: str
    q_len: int
    kv_len: int
    window_size: int = WINDOW
    with_sink: bool = False
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "swa")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 SWA models the BF16 serving path, got {dtype}")
        q, kv, table, cu_q, cu_k, sink, out = _make_tensors(
            device, self.q_len, self.kv_len, self.with_sink
        )
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        return Workload(
            args=(
                q,
                kv,
                table,
                cu_q,
                cu_k,
                self.q_len,
                self.kv_len,
                self.window_size,
                sm_scale,
                sink,
                out,
            ),
            meta={"T": self.q_len, "out0": out.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        out, lse = fn(*wl.args)
        wl.outputs["out"] = out
        wl.outputs["lse"] = lse

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[10].copy_(wl.meta["out0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["out"].float(),
            baseline_wl.outputs["out"].float(),
            rtol=3e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(
            flydsl_wl.outputs["lse"],
            baseline_wl.outputs["lse"],
            rtol=3e-2,
            atol=3e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        kv_len = int(wl.args[6])
        flops = 4.0 * q.shape[0] * q.shape[1] * kv_len * q.shape[2]
        return flops / (median_ms * 1e-3) / 1e12, "TFLOPS"


class Ds4SwaAttnSuite:
    name = "ds4_swa_attn_fwd"

    def resolve_providers(self) -> Providers:
        ixaite_mod = initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.ds4_swa_attn_fwd import ds4_swa_attn_fwd

        del ixaite_mod
        return Providers(
            baseline_name="IXF.ds4_swa_attn_fwd",
            baseline=ixf_ops.ds4_swa_attn_fwd,
            flydsl=ds4_swa_attn_fwd,
            flydsl_name="ixaite.ds4_swa_attn_fwd",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4SwaAttnCase]:
        del model
        cases = [
            Ds4SwaAttnCase(f"ds4_swa_attn_fwd;tok={t:04d};sink={int(sink)}", t, t, WINDOW, sink)
            for t in (128, 512)
            for sink in (False, True)
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4SwaAttnSuite())
