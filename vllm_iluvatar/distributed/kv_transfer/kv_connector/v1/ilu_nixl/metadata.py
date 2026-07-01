# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar extensions for the official NIXL connector metadata."""

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


@dataclass
class IluNixlTransferRegion:
    region_id: int
    block_len: int
    shared_by: list[str]
    layer_indices: list[int]
    group_ids: list[int]
    cache_kinds: list[str]
    identities: list[str]


@dataclass
class IluNixlAgentMetadata(NixlAgentMetadata):
    pp_split: str = ""
    pp_rank: int = 0
    tp_rank: int = 0
    layer_region_ids: dict[str, list[int]] | None = None
    region_ids_by_group: list[list[int]] | None = None
    transfer_regions: list[IluNixlTransferRegion] | None = None


class IluNixlConnectorMetadata(NixlConnectorMetadata):
    """NIXL metadata carrying remote PP topology."""

    def _add_new_req(self, local_block_ids, kv_transfer_params) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            remote_pp_size=kv_transfer_params.get("pp_size", 1),
            remote_pp_split=kv_transfer_params.get("pp_split", ""),
        )
