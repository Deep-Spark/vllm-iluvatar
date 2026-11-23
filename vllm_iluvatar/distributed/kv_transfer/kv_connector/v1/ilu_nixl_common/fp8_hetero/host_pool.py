# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host wire pool: P's 1:1 mirror of its HBM KV in D's byte layout."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from .wire_layout import WirePlan

logger = init_logger(__name__)


class HostKVPool:
    """Wire-layout mirror of every physical KV block held by this worker."""

    def __init__(self, plan: WirePlan, num_blocks: int, *, pinned: bool = True) -> None:
        self.plan = plan
        self.num_blocks = num_blocks
        self._backing = torch.zeros(
            num_blocks * plan.block_stride,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pinned,
        )
        self.rows = torch.as_strided(
            self._backing,
            size=(num_blocks, plan.block_stride),
            stride=(plan.block_stride, 1),
        )
        self.layer_views = plan.layer_views(self._backing, num_blocks)
        logger.info(
            "Allocated host wire KV pool: num_blocks=%d, block_stride=%d, "
            "total_bytes=%d, pinned=%s, layers=%d",
            num_blocks,
            plan.block_stride,
            self._backing.nbytes,
            pinned,
            len(self.layer_views),
        )

    def nixl_reg_entry(self) -> tuple[int, int, int, str]:
        """One ``ibv_reg_mr`` covering the whole pool; layers are offsets."""
        return (self._backing.data_ptr(), self._backing.nbytes, 0, "")

    def storage(self) -> torch.UntypedStorage:
        """Backing storage of the whole pool."""
        return self._backing.untyped_storage()
