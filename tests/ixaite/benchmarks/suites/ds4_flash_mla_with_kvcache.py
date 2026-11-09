# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 decode sparse MLA with paged KV (ixformer vs ixaite).

0731 W4A8: H=8, D=512, SWA page=64 window=128; C4A extra page=16 topk=2048;
C128A extra page=2 topk=2048. T=1 decode and T=8 MTP.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite

HEAD_DIM = 512
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
HEADS = 8


@dataclass(frozen=True)
class Ds4FlashMlaCase:
    case_id: str
    t: int
    flavor: str
    tags: tuple[str, ...] = ("attention", "ds4", "bf16")

    def _extra_geom(self) -> tuple[int, int] | None:
        if self.flavor == "swa":
            return None
        if self.flavor == "c4a":
            return 16, 2048
        if self.flavor == "c128a":
            return 2, 2048
        raise ValueError(self.flavor)

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(f"ds4 flash mla models the BF16 serving path, got {dtype}")
        gen = torch.Generator(device=device)
        gen.manual_seed(7)
        q = torch.randn(
            (self.t, 1, HEADS, HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        page_m = 64
        topk_m = 128
        num_slots_m = 256
        num_blocks_m = num_slots_m // page_m
        k_cache = torch.randn(
            (num_blocks_m, page_m, 1, HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        indices = torch.randint(
            0, num_slots_m, (self.t, 1, topk_m), device=device, dtype=torch.int32
        )
        topk_length = torch.full((self.t,), topk_m, device=device, dtype=torch.int32)
        extra = self._extra_geom()
        extra_k = extra_idx = extra_len = None
        if extra is not None:
            page_e, topk_e = extra
            num_slots_e = max(4096, topk_e)
            num_blocks_e = (num_slots_e + page_e - 1) // page_e
            extra_k = torch.randn(
                (num_blocks_e, page_e, 1, HEAD_DIM),
                device=device,
                dtype=dtype,
                generator=gen,
            )
            extra_idx = torch.randint(
                0,
                num_blocks_e * page_e,
                (self.t, 1, topk_e),
                device=device,
                dtype=torch.int32,
            )
            extra_len = torch.full((self.t,), topk_e, device=device, dtype=torch.int32)
        out = torch.empty_like(q)
        return Workload(
            args=(
                q,
                k_cache,
                indices,
                topk_length,
                SM_SCALE,
                None,
                extra_k,
                extra_idx,
                extra_len,
                out,
                3,
            ),
            meta={"T": self.t, "flavor": self.flavor, "out0": out.clone()},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(*wl.args)

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.args[9].copy_(wl.meta["out0"])

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.args[9].float(),
            baseline_wl.args[9].float(),
            rtol=2e-2,
            atol=2e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        q = wl.args[0]
        return q.numel() * q.element_size() / (median_ms * 1e-3) / 1e9, "GB/s"


class Ds4FlashMlaSuite:
    name = "ds4_flash_mla_with_kvcache"

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops

        op = importlib.import_module("ixaite.ops.ds4_flash_mla_with_kvcache")
        return Providers(
            baseline_name="IXF.ds4_flash_mla_with_kvcache",
            baseline=ixf_ops.ds4_flash_mla_with_kvcache,
            flydsl=op.ds4_flash_mla_with_kvcache,
            flydsl_name="ixaite.ds4_flash_mla_with_kvcache",
        )

    def iter_cases(self, model: str, case_filter: str | None) -> Sequence[Ds4FlashMlaCase]:
        del model
        cases = [
            Ds4FlashMlaCase(f"ds4_flash_mla@{flavor};tok={t:04d}", t, flavor)
            for flavor in ("swa", "c4a", "c128a")
            for t in (1, 8)
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(Ds4FlashMlaSuite())
