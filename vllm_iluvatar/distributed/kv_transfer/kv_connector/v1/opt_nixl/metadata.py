# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import msgspec
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

ReqId = str

PUT_NODE_META_MSG = b"put_node_meta_msg"
GET_NODE_META_MSG = b"get_node_meta_msg"
GET_META_MSG = b"get_meta_msg"
TEARDOWN_COMPLETE_MSG = "teardown_complete"
TEARDOWN_RESULT_OK = b"teardown_complete"
TEARDOWN_RESULT_FAILED = b"teardown_failed"
OWNERSHIP_CANCEL_MSG = "ownership_cancel"
OWNERSHIP_RESULT_ACCEPTED = "accepted"
OWNERSHIP_RESULT_CANCELED = "canceled"
OWNERSHIP_RESULT_TERMINAL = "terminal"
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
#   5: Separate packed inter-block stride from per-layer payload length
#   6: Add explicit D-to-P transfer intent and notification status
#   7: Replace use_packed_slab_xfer bool with kv_registration_kind
#   8: Add explicit sparse receive notification plans and v8 notify wire
#   9: Supersede v8 receive plans with producer-declared notification totals
#  10: Per-pointer block counts and KV group ids in NixlAgentMetadata; layers
#      marked skip_register are absent from every layer_* mapping; the notify
#      group field is the producer's own group id (opaque to the consumer);
#      KvRegistrationKind.CROSS removed
#  11: Register at scheduler-block granularity with per-pointer kernel blocks
#      (no block-id remap); one transfer per (layer | pp, remote tp) across
#      all groups; notify wire
#      <src_pp>-<pp_start>-<pp_end>:<src_tp>-<tp_start>-<tp_end>-<count>:<unit>
#      counted per (source tp, source pp)
#  12: kv_registration_kind removed: every view is registered the same way
#      (storage base + per-pointer offset/len/stride), compatibility is
#      checked per pointer; notify wire
#      <pp>-<pp_start>-<pp_end>:<pp_layer>-<start>-<end>:<tp>-<start>-<end>-<count>
#  13: attention backends are checked per paired layer (layer_backend_mapping)
#      instead of hashing the rank-local backend set, which differs between
#      PP stages
#  14: reconnect teardown tokens and worker-complete acknowledgements
#  15: request-generation ownership fencing for producer WRITE destinations
#  16: delimiter-safe request generations for native completion notifications
#  17: explicit main KV format tags in envelope and agent metadata;
#      compressed peers require matching tags and compatibility hashes
#
NIXL_CONNECTOR_VERSION: int = 17


class TransferIntent(str, Enum):
    """D-to-P request-level command carried with destination block IDs."""

    RECEIVE_KV = "receive_kv"
    NO_TRANSFER = "no_transfer"
    CANCEL = "cancel"


class TransferNotificationStatus(str, Enum):
    """P-to-D result encoded in a NIXL completion notification."""

    SUCCESS = "success"
    FAILED = "failed"


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
    # Inter-block byte stride for each pointer used by a layer.  This is
    # intentionally not the payload length for packed layouts.
    layer_block_len_mapping: dict[str, list[int]]
    # Per pointer, the (offset-within-block, payload-length) segments to xfer.
    layer_desc_mapping: dict[str, list[list[tuple[int, int]]]]
    # Per pointer, the number of blocks registered behind it.  Descriptor ids
    # are laid out [pointer][segment][block], so the first descriptor of a
    # region is the running sum of these counts, not region * num_blocks.
    layer_num_blocks_mapping: dict[str, list[int]]
    # Per pointer, the KV cache group that owns it in *this* worker's
    # numbering.  The writer indexes the peer's block-id tuple with it.
    layer_group_mapping: dict[str, list[int]]
    # Per pointer, kernel blocks per registered (scheduler) block; the
    # registered stride and payload already span all of them.
    layer_kernel_blocks_mapping: dict[str, list[int]]
    # Per pointer, the owning group's scheduler block size in tokens.
    layer_block_tokens_mapping: dict[str, list[int]]
    # Attention backend name of every registered layer.
    layer_backend_mapping: dict[str, str]
    global_rank: int
    pp_rank: int  # for check use
    tp_rank: int
    # Engine-wide setting, including ranks with no registered main KV views.
    main_kv_format: str = "none"


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get the format and compatibility_hash
    2. Require matching format and hash if either peer uses quantized main KV
    3. Decode agent_metadata_bytes and require its format tag to agree

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded
    # Defaults preserve decoding of old, unquantized handshakes.
    main_kv_format: str = "none"


def compute_nixl_compatibility_hash(vllm_config: VllmConfig) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data, and that every rank of both
    engines shares.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache dtype

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in the kind-specific handshake prepare path and are not
    included in this hash to support heterogeneous deployments.  The
    attention backend set depends on which layers a PP stage holds, so
    backends are compared per paired layer by ``LayerMappingPlanner``.

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
        # KV cache dtype affects memory layout
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": is_hma_enabled,
    }
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import format_factors

    factors.update(format_factors())

    compat_hash = hash_factors(factors)
    logger.info(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
    )
    return compat_hash


@dataclass
class PeerMeta:
    addr: str
    pp_size: int = 1
    tp_size: int = 1
    pp_split: list[int] = None
    dp_rank: int = 0
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
    # Stable across the P and D legs, normally the Router transfer_id.
    # It fences reused request IDs and stale control/notification messages.
    generation: str = ""


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send_layerwise: dict[ReqId, ReqMeta] = {}

        self.reqs_to_abort: set[ReqId] = set()
        self.reqs_to_send_time_record: dict[ReqId, float] = {}
        # D-side release proofs returned by P after every native WRITE is
        # terminal, or after P tombstones a request before any WRITE starts.
        self.reqs_to_release: dict[ReqId, tuple[str, str]] = {}

    def _add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        engine_id: str | None = None,
        generation: str = "",
    ) -> ReqMeta:
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            engine_id=engine_id,
            generation=generation,
        )
        self.reqs_to_recv[request_id] = _req
        return _req
