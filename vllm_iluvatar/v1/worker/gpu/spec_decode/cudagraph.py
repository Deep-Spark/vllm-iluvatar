# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

BASELINE = 'def _prepare_dflash_inputs_to_capture(\n    num_reqs: int,\n    num_tokens: int,\n    input_buffers: InputBuffers,\n    block_tables: BlockTables,\n    attn_groups: list[list[AttentionGroup]],\n    kv_cache_config: KVCacheConfig,\n    max_model_len: int,\n    skip_attn: bool,\n    causal: bool | Mapping[int, bool],\n) -> AttentionState:\n    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)\n    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)\n    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)\n    slot_mappings_by_layer = build_slot_mappings_by_layer(\n        slot_mappings, kv_cache_config\n    )\n\n    attn_metadata = None\n    if not skip_attn:\n        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)\n        if block_tables.cp_size > 1:\n            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(\n                input_buffers.dcp_local_seq_lens,\n                input_batch.seq_lens,\n                input_batch.num_reqs,\n                block_tables.cp_size,\n                block_tables.cp_rank,\n                block_tables.cp_interleave,\n                num_reqs_padded=input_batch.num_reqs_after_padding,\n            )\n        attn_metadata = build_attn_metadata(\n            attn_groups=attn_groups,\n            num_reqs=num_reqs,\n            num_tokens=num_tokens,\n            query_start_loc_gpu=input_batch.query_start_loc,\n            query_start_loc_cpu=query_start_loc_cpu,\n            max_query_len=num_tokens // num_reqs,\n            seq_lens=input_batch.seq_lens,\n            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,\n            max_seq_len=max_model_len,\n            block_tables=input_block_tables,\n            slot_mappings=slot_mappings,\n            kv_cache_config=kv_cache_config,\n            for_cudagraph_capture=True,\n            causal=causal,\n        )\n    return AttentionState(attn_metadata, slot_mappings_by_layer)\n'


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
        if block_tables.cp_size > 1:
            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(
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
            for_cudagraph_capture=True,
            causal=causal,
            positions=input_buffers.positions[:num_tokens],
            seq_lens_cpu_upper_bound=torch.full(
                (num_reqs,), max_model_len, dtype=torch.int32
            ),
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
