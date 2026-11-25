# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar extensions for the NIXL connector metadata."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo as NixlHeartbeatInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ReqMeta as NixlReqMeta,
)

from ..ilu_nixl_common.types import (
    SemanticTransferRegion,
    resolve_remote_block_ids,
)

ILU_STRUCTURED_NOTIF_PREFIX = "ILU_N1:"
GET_META_BATCH_MSG = b"get_meta_batch_msg"


@dataclass
class HeartbeatInfo(NixlHeartbeatInfo):
    pp_size: int = 1
    pp_split: str = ""


@dataclass
class ReqMeta(NixlReqMeta):
    remote_pp_size: int = 1
    remote_pp_split: str = ""
    num_tokens: int | None = None
    # Per-group start index in P's block list; -1 marks clipped groups (SWA,
    # CircularBuffer, Mamba) that retain head/tail alignment.
    remote_block_starts: list[int] | None = None
    # Prompt length used by E25 ConsumerPrefixState validation (-1 = unspecified).
    prompt_token_len: int = -1
    # Whether wait_for_save must synchronize the host staging stream on this step.
    sync_save: bool = True

    @property
    def engine_id(self) -> str | None:
        """Expose remote producer engine_id for E25 ConsumerPrefixState."""
        remote = getattr(self, "remote", None)
        return getattr(remote, "engine_id", None) if remote is not None else None


@dataclass
class IluNixlTransferRegion(SemanticTransferRegion):
    """Wire-compatible semantic transfer region descriptor for IluNixl pull."""


@dataclass
class IluNixlHandshakePayload(NixlHandshakePayload):
    main_kv_format: str = "none"
    indexer_kv_layout: str = "legacy"


@dataclass
class IluNixlAgentMetadata(NixlAgentMetadata):
    pp_split: str = ""
    pp_rank: int = 0
    tp_rank: int = 0
    layer_region_ids: dict[str, list[int]] | None = None
    region_ids_by_group: list[list[int]] | None = None
    transfer_regions: list[IluNixlTransferRegion] | None = None
    # Keep old peers compatible: their registered backing was device memory.
    memory_type: str = "VRAM"
    virtual_packed: bool = False
    # DeepSeek-V4.1 topology & truncation metadata.
    kv_layer_end: int | None = None
    swa_bounded_replay: bool | None = None
    skip_layers: list[str] | None = None
    kv_source_layers: list[int] | None = None
    index_source_layers: list[int] | None = None
    compress_ratios: list[int] | None = None
    main_kv_format: str = "none"
    indexer_kv_layout: str = "legacy"
    supports_remote_block_starts: bool = False
    supports_structured_notif: bool = False


def encode_structured_notif(
    req_id: str,
    *,
    status: str = "ok",
    remote_key: int = 0,
    consumer_pp_rank: int = 0,
    consumer_tp_rank: int = 0,
    expected_consumers: int = 1,
) -> bytes:
    """Encode an IluNixl notification carrying sender tag, count, and status."""
    if status not in ("ok", "fail"):
        raise ValueError(f"Unsupported notification status: {status!r}")
    if expected_consumers < 0:
        raise ValueError(
            f"expected_consumers must be non-negative, got {expected_consumers}"
        )
    sender_tag = f"{remote_key}@{consumer_pp_rank},{consumer_tp_rank}"
    return (
        f"{ILU_STRUCTURED_NOTIF_PREFIX}{status}:{sender_tag}:"
        f"{expected_consumers}:{req_id}"
    ).encode()


def decode_structured_notif(
    msg: str,
) -> tuple[str, str, str, int] | None:
    """Decode ``ILU_N1:<status>:<sender_tag>:<expected>:<req_id>`` or None."""
    if not msg.startswith(ILU_STRUCTURED_NOTIF_PREFIX):
        return None
    rest = msg[len(ILU_STRUCTURED_NOTIF_PREFIX) :]
    parts = rest.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Malformed structured notification: {msg!r}")
    status, sender_tag, expected_raw, req_id = parts
    if status not in ("ok", "fail"):
        raise ValueError(f"Invalid structured notification status: {status!r}")
    return req_id, status, sender_tag, int(expected_raw)


def format_kv_group_page_log(
    kv_cache_groups,
    block_id_groups: Sequence[Sequence[int]] | None = None,
    *,
    blocks_per_sw: Sequence[int] | None = None,
    extra: Sequence[str] | None = None,
) -> str:
    """Return a compact per-group page summary for PD alignment logs."""
    parts: list[str] = []
    for group_idx, group in enumerate(kv_cache_groups):
        spec = getattr(group, "kv_cache_spec", group)
        layer_names = getattr(group, "layer_names", ())
        inner_specs = getattr(spec, "kv_cache_specs", None)
        if inner_specs:
            inner = next(iter(inner_specs.values()))
            n_layers = len(inner_specs)
        else:
            inner = spec
            n_layers = len(layer_names)
        pages = (
            len(block_id_groups[group_idx])
            if block_id_groups is not None and group_idx < len(block_id_groups)
            else None
        )
        clip = (
            blocks_per_sw[group_idx]
            if blocks_per_sw is not None and group_idx < len(blocks_per_sw)
            else None
        )
        sample = ",".join(layer_names[:2]) if layer_names else ""
        piece = (
            f"g{group_idx}:{type(inner).__name__}"
            f"(bs={getattr(inner, 'block_size', '?')}"
            f",sw={getattr(inner, 'sliding_window', None)}"
            f",tps={getattr(inner, 'tokens_per_state', None)}"
            f",layers={n_layers})"
        )
        if pages is not None:
            piece += f" pages={pages}"
        if clip is not None:
            piece += f" clip={clip}"
        if extra is not None and group_idx < len(extra) and extra[group_idx]:
            piece += f" {extra[group_idx]}"
        if sample:
            piece += f" [{sample}]"
        elif n_layers == 0:
            piece += " [empty]"
        parts.append(piece)
    return " | ".join(parts)


class IluNixlConnectorMetadata(NixlConnectorMetadata):
    """NIXL metadata carrying remote PP topology, block starts, and E25 receipts."""

    def __init__(self):
        super().__init__()
        self.source_full_ready_receipts: dict[str, Any] = {}
        self.source_state_endpoints: dict[str, Any] = {}

    def _add_new_req(
        self,
        local_block_ids,
        kv_transfer_params,
        local_num_computed_blocks: tuple[int, ...] = (),
        awaiting_kvs: bool = False,
        remote_block_starts: list[int] | None = None,
        prompt_token_len: int = -1,
        sync_save: bool = True,
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
            dcp_size=kv_transfer_params.get("dcp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            pp_size=kv_transfer_params.get("pp_size", 1),
            local_num_computed_blocks=local_num_computed_blocks,
            remote_pp_size=kv_transfer_params.get("pp_size", 1),
            remote_pp_split=kv_transfer_params.get("pp_split", ""),
            awaiting_kvs=awaiting_kvs,
            remote_block_starts=remote_block_starts,
            prompt_token_len=prompt_token_len,
            sync_save=sync_save,
        )

    def add_new_req_to_recv(
        self,
        request_id,
        local_block_ids,
        kv_transfer_params,
        local_num_computed_blocks: tuple[int, ...] = (),
        awaiting_kvs: bool = False,
        remote_block_starts: list[int] | None = None,
        prompt_token_len: int = -1,
    ) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            RemoteMeta,
        )

        remote_blocks = resolve_remote_block_ids(kv_transfer_params)
        if prompt_token_len < 0:
            raw_num = kv_transfer_params.get("remote_num_tokens")
            if isinstance(raw_num, int) and raw_num > 0:
                prompt_token_len = raw_num

        req = self._add_new_req(
            local_block_ids,
            kv_transfer_params,
            local_num_computed_blocks,
            awaiting_kvs=awaiting_kvs,
            remote_block_starts=remote_block_starts,
            prompt_token_len=prompt_token_len,
        )
        req.remote = RemoteMeta(
            block_ids=remote_blocks,
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get("remote_blocks_expiry_time"),
            num_tokens=kv_transfer_params.get("remote_num_tokens"),
        )
        self.reqs_to_recv[request_id] = req

    def add_new_req_to_save(
        self,
        request_id,
        local_block_ids,
        kv_transfer_params,
        num_tokens: int | None = None,
        *,
        sync_save: bool = True,
    ) -> None:
        req = self._add_new_req(
            local_block_ids,
            kv_transfer_params,
            sync_save=sync_save,
        )
        req.num_tokens = num_tokens
        self.reqs_to_save[request_id] = req
