# SPDX-License-Identifier: Apache-2.0
"""DFlash / DSpark CUDA-graph capture that forwards dummy ``positions``.

Corresponds to: vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py
Target   : _prepare_dflash_inputs_to_capture
Baseline : v0.30.0
Why      : community capture omits ``positions`` /
           ``seq_lens_cpu_upper_bound``. Iluvatar V4.1 needs both.
How      : community body plus those two kwargs from the dummy InputBatch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        input_batch.dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            input_batch.num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            # >>> ILU MODIFICATION >>>
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            positions=input_batch.positions,
            # <<< ILU MODIFICATION <<<
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
