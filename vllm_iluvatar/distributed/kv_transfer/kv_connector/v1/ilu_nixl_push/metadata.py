# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

ReqId = str

PUT_NODE_META_MSG = b"put_node_meta_msg"
GET_NODE_META_MSG = b"get_node_meta_msg"
GET_META_MSG = b"get_meta_msg"
DISCONNECT_MSG = msgspec.msgpack.encode(("disconnect", "test-id"))[
    : msgspec.msgpack.encode(("disconnect", "test-id")).index(b"d") + len("disconnect")
]

#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#   3: Add physical_blocks_per_logical_kv_block to NixlAgentMetadata
#   4: Add KV block lease renewal through heartbeats
#
NIXL_CONNECTOR_VERSION: int = 4


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int
    physical_block_size: int
    logical_num_blocks: int
    layer_ptr_mapping: dict[str, list[int]]
    layer_block_len_mapping: dict[str, list[int]]
    layer_desc_mapping: dict[str, list[list[tuple[int, int]]]]
    global_rank: int
    pp_rank: int  # for check use
    tp_rank: int


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get compatibility_hash
    2. Compute local hash and compare
    3. Only if hashes match, decode agent_metadata_bytes

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    factors = {
        # Version compatibility
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    return compat_hash


@dataclass
class PeerMeta:
    addr: str
    block_ratio: int
    block_size: list[int]
    socket: Any = None


@dataclass
class NodeMeta:
    pp_size: int
    tp_size: int
    engine_id: str
    pp_split: list[int]
    dp_rank: int = 0
    expire_time: int = 0
    kv_group_infos: list[str] = None
    physical_block_size: int = 0


@dataclass
class ReqMeta:
    local_block_ids: list[list[int]]
    local_block_ids_np: list[np.ndarray] = None
    engine_id: str = None
    need_permute_after_recv: bool = False
    prompt_token_len: int = -1
    start_transfer: bool = False


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send_layerwise: dict[ReqId, ReqMeta] = {}

        self.reqs_to_abort: set[ReqId] = set()
        self.reqs_to_send_time_record: dict[ReqId, float] = {}

    def _add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
    ) -> ReqMeta:
        _req = ReqMeta(
            local_block_ids=local_block_ids,
        )
        self.reqs_to_recv[request_id] = _req
