# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar extensions for the official NIXL push connector metadata."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo as NixlHeartbeatInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ReqMeta as NixlReqMeta,
)

from ..ilu_nixl_common.types import (
    HmaTransferRoute,
    RegionPairing,
    SemanticTransferRegion,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Preserve exact public type identity for msgspec serialization & external callers.
IluNixlPushTransferRegion = SemanticTransferRegion


@dataclass
class PushObservabilityStats:
    """Lightweight observability counters for push overlap and control plane."""

    wait_push_reg_count: int = 0
    wait_push_reg_total_ms: float = 0.0
    wait_push_reg_max_ms: float = 0.0
    tail_chunk_exposed_count: int = 0
    tail_chunk_exposed_total_ms: float = 0.0
    tail_chunk_exposed_max_ms: float = 0.0
    compiled_xfer_submissions: int = 0
    fast_err_notifs_sent: int = 0
    duplicate_notifs_ignored: int = 0
    skipped_redundant_push_regs: int = 0
    completed_early_buffered: int = 0
    lease_renewal_capped: int = 0

    def record_wait_push_reg(self, wait_ms: float) -> None:
        if wait_ms < 0.0:
            wait_ms = 0.0
        self.wait_push_reg_count += 1
        self.wait_push_reg_total_ms += wait_ms
        if wait_ms > self.wait_push_reg_max_ms:
            self.wait_push_reg_max_ms = wait_ms

    def record_tail_chunk_exposed(self, exposed_ms: float) -> None:
        if exposed_ms < 0.0:
            exposed_ms = 0.0
        self.tail_chunk_exposed_count += 1
        self.tail_chunk_exposed_total_ms += exposed_ms
        if exposed_ms > self.tail_chunk_exposed_max_ms:
            self.tail_chunk_exposed_max_ms = exposed_ms

    def snapshot(self) -> dict[str, int | float]:
        return {
            "wait_push_reg_count": self.wait_push_reg_count,
            "wait_push_reg_avg_ms": (
                self.wait_push_reg_total_ms / self.wait_push_reg_count
                if self.wait_push_reg_count
                else 0.0
            ),
            "wait_push_reg_max_ms": self.wait_push_reg_max_ms,
            "tail_chunk_exposed_count": self.tail_chunk_exposed_count,
            "tail_chunk_exposed_avg_ms": (
                self.tail_chunk_exposed_total_ms / self.tail_chunk_exposed_count
                if self.tail_chunk_exposed_count
                else 0.0
            ),
            "tail_chunk_exposed_max_ms": self.tail_chunk_exposed_max_ms,
            "compiled_xfer_submissions": self.compiled_xfer_submissions,
            "fast_err_notifs_sent": self.fast_err_notifs_sent,
            "duplicate_notifs_ignored": self.duplicate_notifs_ignored,
            "skipped_redundant_push_regs": self.skipped_redundant_push_regs,
            "completed_early_buffered": self.completed_early_buffered,
            "lease_renewal_capped": self.lease_renewal_capped,
        }


@dataclass
class HeartbeatInfo(NixlHeartbeatInfo):
    # Remote P PP topology (from kv_transfer_params on decode requests).
    pp_size: int = 1
    pp_split: str = ""


@dataclass
class ReqMeta(NixlReqMeta):
    # Remote P PP topology (from kv_transfer_params pp_size / pp_split).
    remote_pp_size: int = 1
    remote_pp_split: str = ""
    # Per D-side transfer-group start offset in P's full prompt block list
    # (-1 for clipped groups such as SWA, CircularBuffer, or Mamba).
    remote_block_starts: list[int] | None = None
    # Per P-side transfer-group chunk offset (-1 for tail-only state/ring groups).
    chunk_block_offsets: list[int] | None = None


@dataclass
class IluNixlPushHandshakePayload(NixlHandshakePayload):
    main_kv_format: str = "none"
    indexer_kv_layout: str = "legacy"


@dataclass
class IluNixlPushAgentMetadata(NixlAgentMetadata):
    pp_split: str = ""
    pp_rank: int = 0
    tp_rank: int = 0
    layer_region_ids: dict[str, list[int]] | None = None
    region_ids_by_group: list[list[int]] | None = None
    transfer_regions: list[IluNixlPushTransferRegion] | None = None
    # Distance between consecutive blocks in one packed slab; 0 if unpacked.
    packed_block_stride: int = 0
    wire_cache_dtype: str = ""
    kv_buffer_memory_type: str = "VRAM"
    virtual_packed: bool = False
    kv_layer_end: int | None = None
    swa_bounded_replay: bool | None = None
    skip_layers: list[str] | None = None
    kv_source_layers: list[int] | None = None
    index_source_layers: list[int] | None = None
    compress_ratios: list[int] | None = None
    main_kv_format: str = "none"
    indexer_kv_layout: str = "legacy"
    supports_remote_block_starts: bool = True
    group_is_ssm: list[bool] | None = None


def compute_ilu_push_compatibility_hash(
    vllm_config: "VllmConfig",
    transfer_mode: str = "push",
) -> str:
    """Compute PP/speculator-agnostic compatibility hash including V4.1 KV codecs."""
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import format_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    factors: dict[str, Any] = {
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": is_hma_enabled,
        "transfer_mode": transfer_mode,
    }
    factors.update(format_factors())
    return hash_factors(factors)


@dataclass
class ChunkPushPlan:
    """Immutable, per-request chunked-bulk WRITE plan captured at arm time."""

    request_id: str
    # P local logical block ids, one list per transferable KV group.
    local_block_ids: list[list[int]]
    decode_engine_id: str
    remote_request_id: str
    prefill_tokens: int
    layer_start: int
    layer_end: int
    prefill_chunk_index: int = 0
    is_last_chunk: bool = True
    block_offsets: tuple[int, ...] = ()
    registration: dict[str, Any] = field(default_factory=dict)
    ordered_layers: list[str] = field(default_factory=list)
    armed_at: float = 0.0

    def key(self) -> tuple[str, int]:
        return self.request_id, self.prefill_chunk_index


@dataclass
class ChunkPushReqMeta:
    """P-side metadata for one prefill chunk of a non-layerwise push request."""

    local_block_ids: BlockIds
    decode_engine_id: str | None
    prefill_tokens: int
    block_offsets: tuple[int, ...] = ()
    chunk_index: int = 0
    is_last_chunk: bool = True
    protocol_version: int = 1
    # --- heterogeneous dtype (P BF16 -> D FP8) only --------------------- #
    quant_block_ids: BlockIds = ()
    quant_first_pages: tuple[int, ...] = ()
    num_valid_tokens: int = 0
    requantize_block_ids: BlockIds = ()


class IluNixlPushConnectorMetadata(NixlConnectorMetadata):
    """NIXL metadata carrying remote PP topology and chunked-bulk send requests."""

    def __init__(self) -> None:
        super().__init__()
        self.reqs_to_send_chunked: dict[ReqId, ChunkPushReqMeta] = {}
        self.disconnected_engines: tuple[str, ...] = ()

    def _add_new_req(
        self,
        local_block_ids,
        kv_transfer_params,
        local_num_computed_blocks: tuple[int, ...] = (),
        awaiting_kvs: bool = False,
        remote_block_starts: list[int] | None = None,
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
            dcp_size=kv_transfer_params.get("dcp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            pp_size=kv_transfer_params.get("pp_size", 1),
            local_num_computed_blocks=local_num_computed_blocks,
            awaiting_kvs=awaiting_kvs,
            remote_pp_size=kv_transfer_params.get("pp_size", 1),
            remote_pp_split=kv_transfer_params.get("pp_split", ""),
            remote_block_starts=remote_block_starts,
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
        local_num_computed_blocks: tuple[int, ...] = (),
        awaiting_kvs: bool = False,
        remote_block_starts: list[int] | None = None,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            RemoteMeta,
        )

        req = self._add_new_req(
            local_block_ids,
            kv_transfer_params,
            local_num_computed_blocks,
            awaiting_kvs,
            remote_block_starts=remote_block_starts,
        )
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get("remote_blocks_expiry_time"),
            num_tokens=kv_transfer_params.get("remote_num_tokens"),
        )
        self.reqs_to_recv[request_id] = req

    def add_chunk_req(
        self,
        request_id: ReqId,
        chunk_meta: ChunkPushReqMeta,
    ) -> None:
        self.reqs_to_send_chunked[request_id] = chunk_meta
