# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 indexer paged MQA logits (ixformer vs ixaite)."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

HEADS = 64
HEAD_DIM = 128
BLOCK = 256


@dataclass(frozen=True)
class DsaMqaLogitsCase:
    case_id: str
    t: int
    kv_len: int
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    block: int = BLOCK
    tags: tuple[str, ...] = ("attention", "ds4", "bf16", "indexer")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError("indexer MQA logits models the BF16 serving path")
        pages = max(1, math.ceil(self.kv_len / self.block))
        q = torch.randn((self.t, self.heads, self.head_dim), device=device, dtype=dtype)
        weights = torch.randn((self.t, self.heads), device=device, dtype=dtype)
        kv_cache = torch.randn((pages, self.block, self.head_dim), device=device, dtype=dtype)
        block_table = torch.arange(pages, device=device, dtype=torch.int32).unsqueeze(0)
        cu_q = torch.tensor([0, self.t], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, self.kv_len], device=device, dtype=torch.int32)
        alloc = (self.kv_len + 1) & ~1
        full = torch.empty((self.t, alloc), device=device, dtype=dtype)
        logits = full[:, : self.kv_len]
        return Workload(
            args=(
                q,
                cu_q,
                cu_kv,
                kv_cache,
                block_table,
                weights,
                logits,
                self.t,
                self.kv_len,
                self.kv_len,
                2,
            ),
            meta={"logits0": logits.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[6].copy_(wl.meta["logits0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[6][:, : self.kv_len].float(),
            baseline_wl.args[6][:, : self.kv_len].float(),
            rtol=5e-2,
            atol=2.0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        flops = 2 * self.t * self.heads * self.head_dim * self.kv_len
        return flops / (median_ms * 1e-3) / 1e12, "TFLOP/s"


class DsaIndexerMqaLogitsSuite:
    name = "dsa_indexer_mqa_logits_bf16"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        op = importlib.import_module("ixaite.ops.dsa_indexer_mqa_logits_bf16")
        import ixformer.inference.functions as ixf_ops

        return Providers(
            baseline_name="IXF.dsa_indexer_mqa_logits_with_blocks_bf16",
            baseline=ixf_ops.dsa_indexer_mqa_logits_with_blocks_bf16,
            flydsl=op.dsa_indexer_mqa_logits_with_blocks_bf16,
            flydsl_name="ixaite.dsa_indexer_mqa_logits_with_blocks_bf16",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[DsaMqaLogitsCase]:
        shapes = ((1, 256), (1, 512), (8, 256), (128, 128), (128, 512))
        cases = [
            DsaMqaLogitsCase(f"dsa_mqa_logits@{model};T={t};L={l}", t, l) for t, l in shapes
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(DsaIndexerMqaLogitsSuite())
