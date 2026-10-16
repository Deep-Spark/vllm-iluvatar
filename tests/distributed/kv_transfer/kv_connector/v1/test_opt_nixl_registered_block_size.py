# SPDX-License-Identifier: Apache-2.0
"""Handshake must use the registered page, not the engine-lowered block size."""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
import pytest

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.scheduler import (
    NixlConnectorScheduler,
)


def _handshake_payload(block_size: int) -> NixlHandshakePayload:
    agent = NixlAgentMetadata(
        engine_id="e",
        agent_metadata=b"",
        kv_caches_base_addr=[0],
        device_id=0,
        num_blocks=1,
        kv_cache_layout="BLHNC",
        block_size=block_size,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHMLA",
        physical_blocks_per_logical_kv_block=1,
        physical_block_size=block_size,
        logical_num_blocks=1,
        layer_ptr_mapping={},
        layer_block_len_mapping={},
        layer_desc_mapping={},
        layer_num_blocks_mapping={},
        layer_group_mapping={},
        layer_kernel_blocks_mapping={},
        layer_block_tokens_mapping={},
        layer_backend_mapping={},
        global_rank=0,
        pp_rank=0,
        tp_rank=0,
    )
    return NixlHandshakePayload(
        compatibility_hash="h",
        agent_metadata_bytes=msgspec.msgpack.Encoder().encode(agent),
    )


def _handshake_scheduler(group_block_sizes: list[int]) -> NixlConnectorScheduler:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler.group_block_sizes = group_block_sizes
    # What the engine core leaves in cache_config: the smallest prefix-cacheable
    # group, not the page the workers registered.
    scheduler.block_size = min(group_block_sizes)
    scheduler.registered_block_size = max(group_block_sizes)
    scheduler.node_info = {}
    # Non-None keeps the listener thread out of the test.
    scheduler._scheduler_loop_t = SimpleNamespace()
    return scheduler


def test_handshake_expects_the_registered_page_not_the_lowered_one():
    # V4.1 INT8 pages the ratio-1 indexer at 64 while everything else stays on
    # the negotiated 128, so the workers register 128.
    scheduler = _handshake_scheduler([128, 128, 64])

    scheduler.set_xfer_handshake_metadata({0: _handshake_payload(128)})

    assert scheduler.node_info["physical_block_size"] == 128


def test_handshake_still_rejects_ranks_that_disagree():
    scheduler = _handshake_scheduler([128, 128, 64])

    with pytest.raises(ValueError, match="must agree"):
        scheduler.set_xfer_handshake_metadata({0: _handshake_payload(64)})
