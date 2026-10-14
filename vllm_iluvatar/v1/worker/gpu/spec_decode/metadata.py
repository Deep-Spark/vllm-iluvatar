"""Draft metadata with device positions for V4.1 SWA."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

BASELINE = '    def _build_attn_metadata(\n        self,\n        num_reqs: int,\n        batch_desc: BatchExecutionDescriptor,\n        query_start_loc_np: np.ndarray,\n        seq_lens_cpu_upper_bound: torch.Tensor,\n        step: int,\n        causal: bool | Mapping[int, bool] = True,\n        dcp_local_seq_lens: torch.Tensor | None = None,\n        slot_mappings: torch.Tensor | None = None,\n    ) -> dict[str, Any] | None:\n        num_reqs_padded = batch_desc.num_reqs or num_reqs\n        # A FULL graph replays a captured shape whose padded requests each hold\n        # a full query width, so attention must see the padded token count.\n        # PIECEWISE/eager needs the actual token count, because batch_desc may\n        # carry graph or DP padding that no request owns, which would desync it\n        # from query_start_loc.\n        num_tokens = (\n            batch_desc.num_tokens\n            if batch_desc.cg_mode == CUDAGraphMode.FULL\n            else int(query_start_loc_np[-1])\n        )\n        query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)\n        query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(\n            query_start_loc_np[: num_reqs + 1]\n        )\n        query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]\n        max_query_len = int((query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max())\n        block_tables = [\n            x[:num_reqs_padded] for x in self.block_tables.input_block_tables\n        ]\n        if slot_mappings is None:\n            slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]\n        draft_seq_lens_cpu_upper_bound = torch.zeros(\n            num_reqs_padded, dtype=torch.int32, device="cpu"\n        )\n        torch.add(\n            seq_lens_cpu_upper_bound[:num_reqs],\n            step,\n            out=draft_seq_lens_cpu_upper_bound[:num_reqs],\n        )\n        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)\n        if dcp_local_seq_lens is None and self.dcp_size > 1:\n            # Draft steps advance and rewind their own global sequence lengths,\n            # so the target model\'s DCP-local lengths may already be stale.\n            dcp_local_seq_lens = prepare_dcp_local_seq_lens(\n                self.input_buffers.dcp_local_seq_lens,\n                self.input_buffers.seq_lens,\n                num_reqs,\n                self.dcp_size,\n                self.block_tables.cp_rank,\n                self.block_tables.cp_interleave,\n            )\n        attn_metadata = build_attn_metadata(\n            attn_groups=self.attn_groups,\n            num_reqs=num_reqs_padded,\n            num_tokens=num_tokens,\n            query_start_loc_gpu=self.input_buffers.query_start_loc[\n                : num_reqs_padded + 1\n            ],\n            query_start_loc_cpu=query_start_loc_cpu,\n            max_query_len=max_query_len,\n            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],\n            dcp_local_seq_lens=(\n                None\n                if dcp_local_seq_lens is None\n                else dcp_local_seq_lens[:num_reqs_padded]\n            ),\n            max_seq_len=self.draft_max_seq_len,\n            block_tables=block_tables,\n            slot_mappings=slot_mappings,\n            kv_cache_config=self.kv_cache_config,\n            causal=causal,\n            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,\n            is_prefilling=self.draft_is_prefilling[:num_reqs_padded],\n        )\n        return attn_metadata\n'


def _build_attn_metadata(
    self,
    num_reqs: int,
    batch_desc: BatchExecutionDescriptor,
    query_start_loc_np: np.ndarray,
    seq_lens_cpu_upper_bound: torch.Tensor,
    step: int,
    causal: bool | Mapping[int, bool] = True,
    dcp_local_seq_lens: torch.Tensor | None = None,
    slot_mappings: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    num_reqs_padded = batch_desc.num_reqs or num_reqs
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
    if slot_mappings is None:
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
    if dcp_local_seq_lens is None and self.dcp_size > 1:
        dcp_local_seq_lens = prepare_dcp_local_seq_lens(
            self.input_buffers.dcp_local_seq_lens,
            self.input_buffers.seq_lens,
            num_reqs,
            self.dcp_size,
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
        is_prefilling=self.draft_is_prefilling[:num_reqs_padded],
        positions=self.input_buffers.positions[:num_tokens],
    )
    return attn_metadata
