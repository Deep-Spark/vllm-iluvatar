# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 paged-attention suite (ixformer vs ixaite)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload


@dataclass(frozen=True)
class AttentionCase:
    case_id: str
    operation: str
    context_len: int
    batch: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    graph_capacity: int
    tags: tuple[str, ...]
    query_len: int | None = None
    softmax_head_dim: int | None = None

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"attention suite models the BF16 serving path, got {dtype}"
            )

        scale_dim = self.softmax_head_dim or self.head_dim
        softmax_scale = scale_dim**-0.5
        if "packed_encoder" in self.tags:
            total_q = self.batch * self.context_len
            q = torch.randn(
                total_q,
                self.num_query_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            k = torch.randn(
                total_q,
                self.num_kv_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.randn_like(k)
            cu_seqlens = torch.arange(
                0,
                total_q + 1,
                self.context_len,
                device=device,
                dtype=torch.int32,
            )
            out = torch.empty_like(q)
            return Workload(
                kwargs={
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": self.context_len,
                    "max_seqlen_k": self.context_len,
                    "softmax_scale": softmax_scale,
                    "causal": False,
                    "out": out,
                },
                meta={
                    "B": self.batch,
                    "Sq": self.context_len,
                    "Sk": self.context_len,
                    "Hq": self.num_query_heads,
                    "Hkv": self.num_kv_heads,
                    "D": self.head_dim,
                    "kv_layout": "packed",
                },
            )

        num_blocks_per_seq = math.ceil(self.context_len / self.page_size)
        num_blocks = self.batch * num_blocks_per_seq
        table_cols = (
            math.ceil(self.graph_capacity / self.page_size)
            if self.operation == "flash_attn_with_kvcache"
            else num_blocks_per_seq
        )
        block_table = torch.zeros(
            (self.batch, table_cols), device=device, dtype=torch.int32
        )
        block_table[:, :num_blocks_per_seq] = torch.arange(
            num_blocks, device=device, dtype=torch.int32
        ).reshape(self.batch, num_blocks_per_seq)
        cache_seqlens = torch.full(
            (self.batch,), self.context_len, device=device, dtype=torch.int32
        )
        # Iluvatar vLLM HND paged layout: [blocks, Hkv, page, D].
        k_cache = torch.randn(
            num_blocks,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.randn_like(k_cache)

        if self.operation == "flash_attn_varlen_func":
            query_len = self.query_len or self.context_len
            total_q = self.batch * query_len
            q = torch.randn(
                total_q,
                self.num_query_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            cu_seqlens = torch.arange(
                0,
                total_q + 1,
                query_len,
                device=device,
                dtype=torch.int32,
            )
            out = torch.empty_like(q)
            kwargs: dict[str, object] = {
                "q": q,
                "k": k_cache,
                "v": v_cache,
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": torch.arange(
                    self.batch + 1, device=device, dtype=torch.int32
                ) * self.context_len,
                "max_seqlen_q": query_len,
                "max_seqlen_k": self.context_len,
                "softmax_scale": softmax_scale,
                "causal": True,
                "block_table": block_table,
                "out": out,
            }
        else:
            q = torch.randn(
                self.batch,
                1,
                self.num_query_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            out = torch.empty_like(q)
            cu_query_lens = torch.arange(
                self.batch + 1,
                device=device,
                dtype=torch.int32,
            )
            kwargs = {
                "q": q,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "cache_seqlens": cache_seqlens,
                "cu_query_lens": cu_query_lens,
                "max_query_len": 1,
                "block_table": block_table,
                "softmax_scale": softmax_scale,
                "causal": True,
                "max_context_len": self.context_len,
                "out": out,
            }

        return Workload(
            kwargs=kwargs,
            meta={
                "B": self.batch,
                "Sq": self.query_len or self.context_len
                if self.operation == "flash_attn_varlen_func"
                else 1,
                "Sk": self.context_len,
                "Hq": self.num_query_heads,
                "Hkv": self.num_kv_heads,
                "D": self.head_dim,
                "page": self.page_size,
                "kv_layout": "HND",
            },
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        fn(self.operation, wl.kwargs)

    def prepare_cuda_graph(self, wl: Workload) -> None:
        if self.operation == "flash_attn_with_kvcache":
            active_blocks = math.ceil(self.context_len / self.page_size)
            capacity_blocks = math.ceil(self.graph_capacity / self.page_size)
            total_capacity_blocks = self.batch * capacity_blocks
            for name in ("k_cache", "v_cache"):
                source = wl.kwargs[name]
                expanded = torch.zeros(
                    (total_capacity_blocks, *source.shape[1:]),
                    device=source.device,
                    dtype=source.dtype,
                )
                for batch_index in range(self.batch):
                    source_start = batch_index * active_blocks
                    target_start = batch_index * capacity_blocks
                    expanded[target_start : target_start + active_blocks].copy_(
                        source[source_start : source_start + active_blocks]
                    )
                wl.kwargs[name] = expanded
            wl.kwargs["block_table"].copy_(
                torch.arange(
                    total_capacity_blocks,
                    device=wl.kwargs["block_table"].device,
                    dtype=torch.int32,
                ).reshape(self.batch, capacity_blocks)
            )
            wl.kwargs["use_cuda_graph"] = True
            wl.kwargs["max_context_len"] = self.graph_capacity

    def prepare_cuda_graph_replay(self, wl: Workload) -> None:
        wl.kwargs["q"].neg_()
        replay_context = (
            max(1, self.context_len // 2)
            if self.context_len >= self.graph_capacity
            else min(
                self.graph_capacity,
                max(self.context_len + 1, self.context_len * 2),
            )
        )
        wl.kwargs["cache_seqlens"].fill_(replay_context)

    def cleanup_cuda_graph(self) -> None:
        # ixaite intentionally pins capture-time staging/workspace allocations
        # until the graph that owns their pointers has been destroyed.
        import ixaite

        ixaite.release_cuda_graph_pins()

    def check_cuda_graph_replay(
        self, reference: Callable[..., object], wl: Workload
    ) -> None:
        graph_out = wl.kwargs["out"]
        reference_out = torch.empty_like(graph_out)
        reference_kwargs = {
            **wl.kwargs,
            "out": reference_out,
            "max_context_len": self.graph_capacity,
        }
        reference_kwargs.pop("use_cuda_graph", None)
        reference(self.operation, reference_kwargs)
        torch.testing.assert_close(
            graph_out,
            reference_out,
            rtol=3e-2,
            atol=3e-2,
        )

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        torch.testing.assert_close(
            flydsl_wl.kwargs["out"],
            baseline_wl.kwargs["out"],
            rtol=3e-2,
            atol=3e-2,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        del median_ms, wl
        return None


class AttentionSuite:
    name = "attention"

    @staticmethod
    def supports_case_cuda_graph(case: AttentionCase) -> bool:
        return "eager_decode" in case.tags

    def resolve_providers(self) -> Providers:
        import ixaite
        import ixformer.contrib.vllm_flash_attn as ixf_attention

        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)

        def make_dispatch(
            module: object,
            provider_name: str,
            *,
            use_production_decode: bool = False,
        ) -> Callable[..., object]:
            def dispatch(operation: str, kwargs: dict[str, object]) -> object:
                if use_production_decode and operation == "flash_attn_with_kvcache":
                    operation = "flash_attn_qwen35_decode"
                    kwargs = {
                        **kwargs,
                        "window_size": (-1, -1),
                        "alibi_slopes": None,
                        "softcap": 0.0,
                        "sinks": None,
                        "use_cuda_graph": bool(kwargs.get("use_cuda_graph", False)),
                    }
                fn = getattr(module, operation, None)
                if not callable(fn):
                    raise RuntimeError(
                        f"{provider_name} does not expose callable {operation}"
                    )
                return fn(**kwargs)

            return dispatch

        return Providers(
            baseline_name="ixformer.contrib.vllm_flash_attn",
            baseline=make_dispatch(ixf_attention, "ixformer.contrib.vllm_flash_attn"),
            flydsl=make_dispatch(
                ixaite,
                "ixaite",
                use_production_decode=True,
            ),
            flydsl_name="ixaite.flash_attn (production decode)",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[AttentionCase]:
        spec = l2_spec(model, self.name)
        tp_tag = model.rsplit("_", 1)[-1] if "_tp" in model else "tp1"
        if spec.get("mode") == "packed_encoder":
            layouts = spec.get("layouts")
            if layouts is None:
                layouts = (
                    {
                        "num_query_heads": spec["num_query_heads"],
                        "num_kv_heads": spec["num_kv_heads"],
                        "head_dim": spec["head_dim"],
                        "softmax_head_dim": spec.get("softmax_head_dim"),
                    },
                )
            cases = [
                AttentionCase(
                    case_id=(
                        f"packed_encoder@{model};B={spec['batch']};S={context};"
                        f"Hq={layout['num_query_heads']};"
                        f"Hkv={layout['num_kv_heads']};"
                        f"D={layout['head_dim']};layout=packed"
                    ),
                    operation="flash_attn_varlen_func",
                    context_len=context,
                    batch=spec["batch"],
                    num_query_heads=layout["num_query_heads"],
                    num_kv_heads=layout["num_kv_heads"],
                    head_dim=layout["head_dim"],
                    page_size=0,
                    graph_capacity=context,
                    tags=("attention", "packed_encoder", "bf16", tp_tag),
                    softmax_head_dim=layout.get(
                        "softmax_head_dim", spec.get("softmax_head_dim")
                    ),
                )
                for layout in layouts
                for context in (int(value) for value in spec["contexts"])
            ]
            if case_filter:
                cases = [case for case in cases if case_filter in case.case_id]
            return cases
        decode_contexts = tuple(
            int(value)
            for value in spec.get("decode_contexts", spec.get("contexts", ()))
        )
        graph_capacity = int(spec.get("graph_capacity", max(decode_contexts)))

        cases = [
            AttentionCase(
                case_id=(
                    f"{kind}@{model};B={spec['batch']};Sk={context};"
                    f"Hq={spec['num_query_heads']};Hkv={spec['num_kv_heads']};"
                    f"D={spec['head_dim']};page={spec['page_size']};layout=HND"
                ),
                operation=operation,
                context_len=context,
                batch=spec["batch"],
                num_query_heads=spec["num_query_heads"],
                num_kv_heads=spec["num_kv_heads"],
                head_dim=spec["head_dim"],
                page_size=spec["page_size"],
                graph_capacity=graph_capacity,
                tags=("attention", kind, "paged_kv", "bf16", tp_tag),
            )
            for kind, operation, contexts_key in (
                ("paged_prefill", "flash_attn_varlen_func", "prefill_contexts"),
                ("eager_decode", "flash_attn_with_kvcache", "decode_contexts"),
            )
            for context in (
                int(value) for value in spec.get(contexts_key, spec.get("contexts", ()))
            )
        ]
        # Suffix queries see both cached prefix and the newly written tokens.
        for query_len, context in spec.get("continuations", ()):
            cases.append(AttentionCase(
                case_id=f"paged_continuation@{model};Sq={query_len};Sk={context}",
                operation="flash_attn_varlen_func", context_len=int(context),
                query_len=int(query_len), batch=spec["batch"],
                num_query_heads=spec["num_query_heads"], num_kv_heads=spec["num_kv_heads"],
                head_dim=spec["head_dim"], page_size=spec["page_size"],
                graph_capacity=graph_capacity,
                tags=("attention", "paged_continuation", "bf16", tp_tag),
            ))
        if case_filter:
            cases = [case for case in cases if case_filter in case.case_id]
        return cases

    def gate_min_speedup(self) -> float:
        return 1.0


register(AttentionSuite())
