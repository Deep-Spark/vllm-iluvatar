# SPDX-License-Identifier: Apache-2.0
"""MoE token-index suite (plain / EP / decode; ixformer vs ixaite)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases, initialize_ixaite


def _assert_token_index(
    topk_ids: torch.Tensor,
    src_dst: torch.Tensor,
    dst_src: torch.Tensor,
    expert_sizes: torch.Tensor,
    start_id: int,
    end_id: int,
    *,
    expand_tokens: int | None = None,
) -> None:
    eids = topk_ids.reshape(-1)
    valid = (eids >= start_id) & (eids < end_id)
    dst_tokens = int(valid.sum().item())
    if expand_tokens is not None:
        assert expand_tokens == dst_tokens
    if dst_tokens == 0:
        assert torch.all(src_dst == -1)
        return
    assert torch.equal(src_dst == -1, ~valid)
    dst_eids = torch.empty(dst_tokens, dtype=torch.int32, device=topk_ids.device)
    placed = src_dst[valid]
    dst_eids[placed.long()] = eids[valid]
    assert torch.equal(eids[valid].sort().values, dst_eids)
    assert torch.equal(eids[dst_src.long()], dst_eids)
    for local, eid in enumerate(range(start_id, end_id)):
        assert int(expert_sizes[local].item()) == int((topk_ids == eid).sum().item())


@dataclass(frozen=True)
class MoeComputeTokenIndexCase:
    case_id: str
    variant: str
    num_tokens: int
    num_experts: int
    topk: int
    start_expert_id: int
    end_expert_id: int
    tags: tuple[str, ...] = ("moe", "w4a8", "routing")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        del dtype
        torch.manual_seed(self.num_tokens * 17 + self.num_experts + self.topk)
        topk_ids = torch.randint(
            0,
            self.num_experts,
            (self.num_tokens, self.topk),
            device=device,
            dtype=torch.int32,
        )
        return Workload(
            args=(self.variant, topk_ids, self.num_experts),
            kwargs={
                "start_expert_id": self.start_expert_id,
                "end_expert_id": self.end_expert_id,
            },
            meta={
                "T": self.num_tokens,
                "E": self.num_experts,
                "k": self.topk,
                "variant": self.variant,
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        variant, topk_ids, num_experts = wl.args
        payload = fn(
            variant,
            topk_ids,
            num_experts,
            start_expert_id=wl.kwargs["start_expert_id"],
            end_expert_id=wl.kwargs["end_expert_id"],
        )
        wl.outputs["src_dst"] = payload[0]
        wl.outputs["dst_src"] = payload[1]
        wl.outputs["expert_sizes"] = payload[2]
        if len(payload) > 3:
            expand = payload[3]
            wl.outputs["expand_tokens"] = (
                int(expand.item()) if torch.is_tensor(expand) else int(expand)
            )

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.outputs["expert_sizes"].cpu(),
            baseline_wl.outputs["expert_sizes"].cpu(),
            rtol=0,
            atol=0,
        )
        topk_ids = flydsl_wl.args[1]
        start_id = int(flydsl_wl.kwargs["start_expert_id"])
        end_id = int(flydsl_wl.kwargs["end_expert_id"])
        expand = flydsl_wl.outputs.get("expand_tokens")
        expand_tokens = int(expand) if expand is not None else None
        _assert_token_index(
            topk_ids,
            flydsl_wl.outputs["src_dst"],
            flydsl_wl.outputs["dst_src"][
                : int(flydsl_wl.outputs["expert_sizes"].sum())
            ],
            flydsl_wl.outputs["expert_sizes"],
            start_id,
            end_id,
            expand_tokens=expand_tokens,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str]:
        tokens = wl.args[1].numel()
        return tokens / (median_ms * 1e-3) / 1e6, "MTok/s"


def _dispatch(table: Mapping[str, Callable[..., object]]):
    def _call(
        variant: str,
        topk_ids: torch.Tensor,
        num_experts: int,
        *,
        start_expert_id: int,
        end_expert_id: int,
    ):
        if variant == "ep":
            src, dst, sizes, _cpu, expand = table["ep"](
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_expert_id,
                end_expert_id=end_expert_id,
            )
            return src, dst, sizes, expand
        if variant == "decode":
            src, dst, sizes = table["decode"](
                topk_ids=topk_ids,
                num_experts=num_experts,
            )
            return src, dst, sizes
        out = table["plain"](
            topk_ids=topk_ids,
            num_experts=num_experts,
        )
        return out[0], out[1], out[2]

    return _call


class MoeComputeTokenIndexSuite:
    name = "moe_compute_token_index"
    supports_cuda_graph = False

    def resolve_providers(self) -> Providers:
        initialize_ixaite()
        import ixformer.inference.functions as ixf_ops
        from ixaite.ops.moe_compute_token_index import (
            moe_compute_token_index as ixaite_plain,
        )
        from ixaite.ops.moe_compute_token_index import (
            moe_compute_token_index_ep as ixaite_ep,
        )
        from ixaite.ops.moe_compute_token_index import (
            moe_compute_token_index_for_decode as ixaite_decode,
        )

        return Providers(
            baseline_name="IXF.moe_compute_token_index",
            baseline=_dispatch(
                {
                    "plain": ixf_ops.moe_compute_token_index,
                    "ep": ixf_ops.moe_compute_token_index_ep,
                    "decode": ixf_ops.moe_compute_token_index_for_decode,
                }
            ),
            flydsl=_dispatch(
                {
                    "plain": ixaite_plain,
                    "ep": ixaite_ep,
                    "decode": ixaite_decode,
                }
            ),
            flydsl_name="ixaite.moe_compute_token_index",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[MoeComputeTokenIndexCase]:
        spec = l2_spec(model, self.name)
        num_experts = int(spec["num_experts"])
        topk = int(spec["topk"])
        start_id = int(spec.get("start_expert_id", 0))
        end_id = int(spec.get("end_expert_id", num_experts))
        cases = [
            MoeComputeTokenIndexCase(
                case_id=(f"{variant}@{model};T={num_tokens};E={num_experts};k={topk}"),
                variant=variant,
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                start_expert_id=start_id,
                end_expert_id=end_id,
            )
            for variant in spec["variants"]
            for num_tokens in spec["num_tokens"]
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 0.0


register(MoeComputeTokenIndexSuite())
