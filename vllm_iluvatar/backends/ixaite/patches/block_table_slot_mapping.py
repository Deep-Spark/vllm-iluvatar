# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Upstream reference: vLLM v0.27.1 (6e448d0),
# vllm/v1/worker/block_table.py.
# Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. for the ixaite backend.
"""Torch fallback for v1 BlockTable.compute_slot_mapping on Iluvatar.

Community ``_compute_slot_mapping_kernel`` is Triton-only. Triton backends are
unavailable on CoreX, so the decorated symbol stays a plain function and
``kernel[(grid,)]`` raises TypeError during real requests (not just warmup).

This implementation is vectorized (no per-request ``.item()``) so decode can
enter CUDA graphs once the rest of the Flash hot path is sync-free.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode

logger = init_logger(__name__)


def _compute_slot_mapping_torch(
    self: BlockTable, num_reqs: int, query_start_loc, positions
) -> None:
    if self.slot_mapping_mode == SlotMappingMode.NONE:
        return
    assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

    num_tokens = int(positions.shape[0])
    max_num_tokens = int(self.max_num_batched_tokens)
    slot_mapping = self.slot_mapping.gpu
    if num_tokens < max_num_tokens:
        slot_mapping[num_tokens:max_num_tokens].fill_(PAD_SLOT_ID)
    if num_tokens == 0 or num_reqs <= 0:
        return

    # Virtual block, local context-parallel slice, then kernel-block index.
    # One allocator block covers blocks_per_kv_block kernel blocks.
    block_size = int(self.block_size)
    blocks_per_kv_block = int(self.blocks_per_kv_block)
    cp_world = int(self.dcp_world_size)
    cp_rank = int(self.dcp_rank)
    interleave = int(self.cp_kv_cache_interleave_size)
    block_table = self.block_table.gpu
    device = positions.device
    pos = positions.to(device=device, dtype=torch.long)
    # query_start_loc is [num_reqs + 1]. Map tokens to requests without .item().
    ends = query_start_loc.to(device=device, dtype=torch.long)[1 : num_reqs + 1]
    tok = torch.arange(num_tokens, device=device, dtype=torch.long)
    req_ids = torch.searchsorted(ends, tok, right=True).clamp(max=num_reqs - 1)

    virtual_block_size = int(self.kv_cache_block_size) * cp_world
    virtual_block_indices = pos // virtual_block_size
    virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
    if cp_world == 1:
        local_block_offsets = virtual_block_offsets
    else:
        group = cp_world * interleave
        local_block_offsets = (virtual_block_offsets // group) * interleave + (
            virtual_block_offsets % interleave
        )
    block_indices = (
        virtual_block_indices * blocks_per_kv_block + local_block_offsets // block_size
    )
    block_numbers = block_table[req_ids, block_indices].to(dtype=torch.int64)
    slots = block_numbers * block_size + local_block_offsets % block_size
    if cp_world == 1:
        slot_mapping[:num_tokens] = slots
        return
    is_local = (virtual_block_offsets // interleave) % cp_world == cp_rank
    slot_mapping[:num_tokens] = torch.where(is_local, slots, PAD_SLOT_ID)


def register() -> None:
    if getattr(BlockTable, "_ilu_torch_slot_mapping", False):
        return
    BlockTable.compute_slot_mapping = _compute_slot_mapping_torch  # type: ignore[method-assign]
    BlockTable._ilu_torch_slot_mapping = True  # type: ignore[attr-defined]
    logger.info(
        "ILU: using torch fallback for BlockTable.compute_slot_mapping "
        "(Triton unavailable on Iluvatar)"
    )
