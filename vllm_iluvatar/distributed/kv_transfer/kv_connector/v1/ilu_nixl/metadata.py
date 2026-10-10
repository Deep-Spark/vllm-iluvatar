# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar extensions for the official NIXL connector metadata."""

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo as NixlHeartbeatInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ReqMeta as NixlReqMeta,
)


@dataclass
class HeartbeatInfo(NixlHeartbeatInfo):
    pp_size: int = 1
    pp_split: str = ""


@dataclass
class ReqMeta(NixlReqMeta):
    remote_pp_size: int = 1
    remote_pp_split: str = ""
    num_tokens: int | None = None


@dataclass
class IluNixlTransferRegion:
    region_id: int
    block_len: int
    shared_by: list[str]
    layer_indices: list[int]
    group_ids: list[int]
    cache_kinds: list[str]
    identities: list[str]
    # A virtual region may be a slot inside one physically registered packed
    # allocation. Non-packed regions use physical_backing_id == region_id,
    # byte_offset == 0, and block_stride == block_len.
    physical_backing_id: int = 0
    byte_offset: int = 0
    block_stride: int = 0


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
    """NIXL metadata carrying remote PP topology."""

    def _add_new_req(
        self,
        local_block_ids,
        kv_transfer_params,
        local_num_computed_blocks: tuple[int, ...] = (),
        # >>> ILU MODIFICATION >>>
        # Forward community v0.31.0 ``awaiting_kvs`` to ReqMeta (inherited from
        # NixlReqMeta). Ilu adds remote PP topology fields on top.
        awaiting_kvs: bool = False,
        # <<< ILU MODIFICATION <<<
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
        )

    def add_new_req_to_save(
        self,
        request_id,
        local_block_ids,
        kv_transfer_params,
        num_tokens: int | None = None,
    ) -> None:
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.num_tokens = num_tokens
        self.reqs_to_save[request_id] = req
