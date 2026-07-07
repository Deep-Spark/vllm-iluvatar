# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT GDN chunk custom op.

Corresponds to:
    vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py

This module uses the official ``ChunkGatedDeltaRule.register_oot`` extension
point to own the GDN chunk calculation entry on Iluvatar. The community attention
layer still instantiates ``ChunkGatedDeltaRule()``, and ``CustomOp.__new__``
selects this OOT subclass after registration.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,  # pyright: ignore[reportMissingImports]
)


@ChunkGatedDeltaRule.register_oot()
class IluChunkGatedDeltaRule(ChunkGatedDeltaRule):
    """Use the Iluvatar FLA chunk path for GDN Triton prefill."""

    def __init__(self) -> None:
        super().__init__()
        if self.gdn_prefill_backend == "flashinfer":
            # Community selects FlashInfer on SM90. Iluvatar keeps the same
            # public resolver but routes the chunk op to the plugin FLA path.
            self.gdn_prefill_backend = "triton"
        if self.gdn_prefill_backend == "triton":
            self._forward_method = self.forward_native

    def forward_native(
        self,
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        initial_state: Any,
        output_final_state: bool,
        cu_seqlens: Any = None,
        chunk_indices: Any = None,
        chunk_offsets: Any = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: Any = None,
    ):
        from vllm_iluvatar.model_executor.layers.fla.ops.chunk import (
            chunk_gated_delta_rule,
        )

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )
