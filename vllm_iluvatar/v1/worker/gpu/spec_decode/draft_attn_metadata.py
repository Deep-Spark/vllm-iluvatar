# SPDX-License-Identifier: Apache-2.0
"""Draft attention metadata that forwards persistent ``positions``.

Corresponds to: vllm/v1/worker/gpu/spec_decode/speculator.py
Target   : DraftModelSpeculator._build_attn_metadata
Baseline : v0.30.0
Why      : community omits ``positions``. Iluvatar V4.1 compressor / C128A
           require them.
How      : community body plus
           ``positions=self.input_buffers.positions[:num_tokens]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _build_attn_metadata(
    self,
    num_reqs: int,
    batch_desc: BatchExecutionDescriptor,
    query_start_loc_np: np.ndarray,
    seq_lens_cpu_upper_bound: torch.Tensor,
    step: int,
    causal: bool | Mapping[int, bool] = True,
    dcp_local_seq_lens: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    num_reqs_padded = batch_desc.num_reqs or num_reqs
    # A FULL graph replays a captured shape whose padded requests each hold
    # a full query width, so attention must see the padded token count.
    # PIECEWISE/eager needs the actual token count, because batch_desc may
    # carry graph or DP padding that no request owns, which would desync it
    # from query_start_loc.
    num_tokens = (
        batch_desc.num_tokens
        if batch_desc.cg_mode == CUDAGraphMode.FULL
        else int(query_start_loc_np[-1])
    )
    query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
    query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(
        query_start_loc_np[: num_reqs + 1]
    )
    query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]
    max_query_len = int((query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max())
    block_tables = [x[:num_reqs_padded] for x in self.block_tables.input_block_tables]
    slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]
    draft_seq_lens_cpu_upper_bound = torch.zeros(
        num_reqs_padded, dtype=torch.int32, device="cpu"
    )
    torch.add(
        seq_lens_cpu_upper_bound[:num_reqs],
        step,
        out=draft_seq_lens_cpu_upper_bound[:num_reqs],
    )
    draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
    if dcp_local_seq_lens is None and self.block_tables.cp_size > 1:
        # Draft steps advance and rewind their own global sequence lengths,
        # so the target model's DCP-local lengths may already be stale.
        dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
            self.input_buffers.dcp_local_seq_lens,
            self.input_buffers.seq_lens,
            num_reqs,
            self.block_tables.cp_size,
            self.block_tables.cp_rank,
            self.block_tables.cp_interleave,
        )
    attn_metadata = build_attn_metadata(
        attn_groups=self.attn_groups,
        num_reqs=num_reqs_padded,
        num_tokens=num_tokens,
        query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
        dcp_local_seq_lens=(
            None if dcp_local_seq_lens is None else dcp_local_seq_lens[:num_reqs_padded]
        ),
        max_seq_len=self.draft_max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=self.kv_cache_config,
        causal=causal,
        seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
        # >>> ILU MODIFICATION >>>
        positions=self.input_buffers.positions[:num_tokens],
        # <<< ILU MODIFICATION <<<
        is_prefilling=self.draft_is_prefilling[:num_reqs],
    )
    return attn_metadata
