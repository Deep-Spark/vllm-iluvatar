# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side logic for the NIXL connector."""

import asyncio
import os
import threading
import time
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import uvloop
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from .metadata import (
    DISCONNECT_MSG,
    GET_NODE_META_MSG,
    PUT_NODE_META_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NodeMeta,
    PeerMeta,
    ReqId,
    ReqMeta,
    TransferIntent,
)
from .utils import (
    NodeError,
    zmq_comm2proxy,
    zmq_ctx,
)

_SERVER_ADDR = os.getenv("VLLM_ILU_NIXL_SERVER_ADDR", "172.31.0.123:8000")
_PROXY_ADDR = os.getenv("VLLM_ILU_NIXL_PROXY_ADDR", "tcp://172.31.0.123:8079")
_SCHEDULER_ADDR = os.getenv("VLLM_ILU_NIXL_SCHEDULER_ADDR", "tcp://127.0.0.1:19001")
_FORCE_PER_RANK_SEND = os.getenv("VLLM_ILU_NIXL_FORCE_PER_RANK_SEND", "0").lower() in {
    "1",
    "true",
    "yes",
}

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger("vllm." + __name__)


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        engine_id: str,
    ):
        # ---- config ----
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        if current_platform.device_type == "cpu":
            raise AssertionError("iluvatar nixl only support cuda device")
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # ---- hybrid kv ----
        if not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        self.kv_group_infos = []
        self._all_specs = set()
        self.kv_group_nums = len(kv_cache_config.kv_cache_groups)
        self.group_block_sizes = []
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                self._all_specs.update(set(group.kv_cache_spec.kv_cache_specs.values()))
                specs = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
            else:
                self._all_specs.add(group.kv_cache_spec)
                specs = group.kv_cache_spec
            self.group_block_sizes.append(specs.block_size)

            self.kv_group_infos.append(specs.__class__.__name__)

        self._has_multi_size = len(set(self.group_block_sizes)) > 1
        self._has_mamba = any(isinstance(spec, MambaSpec) for spec in self._all_specs)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._force_rank_wise_sends = (
            str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            == "true"
        )

        self._apply_block_id_preprocess = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        if (
            self._has_mamba
            and len(self._all_specs) != 1
            and not self._apply_block_id_preprocess
        ):
            raise RuntimeError(
                "For model with mamba but not only with mamba, do not support disable_hybrid_kv_cache_manager is True"
            )

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        self.blocks_per_sw = []
        if self._apply_block_id_preprocess:
            for g in kv_cache_config.kv_cache_groups:
                if isinstance(g.kv_cache_spec, SlidingWindowSpec):
                    transfer_block_nums = (
                        cdiv(g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
                        + 1
                    )
                    self.blocks_per_sw.append(
                        (-transfer_block_nums, transfer_block_nums)
                    )
                elif isinstance(g.kv_cache_spec, MambaSpec):
                    # for none or all
                    self.blocks_per_sw.append((0, 1))
                else:
                    self.blocks_per_sw.append((0, 0))

        self.remap_to_kernel_size = []
        for g in kv_cache_config.kv_cache_groups:
            if isinstance(g.kv_cache_spec, MambaSpec):
                self.remap_to_kernel_size.append(0)
            else:
                self.remap_to_kernel_size.append(1)

        is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        assert not is_bidirectional_kv_xfer_enabled, (
            "iluvatar not support Bi-directional KV transfer"
        )

        # ---- Metadata ----
        assert vllm_config.kv_transfer_config.kv_role in [
            "kv_producer",
            "kv_consumer",
        ], f"Unsupported kv role: {vllm_config.kv_transfer_config.kv_role}"
        self.is_consumer = vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        pp_split: list[int] = []
        pp_world_size = vllm_config.parallel_config.pipeline_parallel_size
        num_hidden_layers = (
            vllm_config.model_config.hf_text_config.num_hidden_layers
            if vllm_config.model_config.hf_text_config is not None
            else vllm_config.model_config.hf_config.num_hidden_layers
        )
        for i in range(pp_world_size):
            pp_split.append(
                get_pp_indices(
                    num_hidden_layers,
                    i,
                    pp_world_size,
                )[0]
            )
        self.pp_split = pp_split
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 480
            )
        )
        self.local_addr = f"{self.side_channel_host}:{self.side_channel_port}"
        self.node_info = {
            "server_addr": _SERVER_ADDR,  # vllm openai server ip:port
            "side_channel_addr": self.local_addr,  # p/d direct information exchange ip:port
            "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
            "tp_size": vllm_config.parallel_config.tensor_parallel_size,
            "engine_id": self.engine_id,
            "pp_split": pp_split,
            "dp_rank": vllm_config.parallel_config.data_parallel_rank,
            "kv_role": vllm_config.kv_transfer_config.kv_role,
            "kv_group_infos": self.kv_group_infos,  # the p/d kv group must be the same
        }

        # ---- transferring ----
        # Background thread for handling new handshake requests.
        self.ctx = zmq.asyncio.Context()
        self._async_loop = uvloop.new_event_loop()
        self._scheduler_loop_t: threading.Thread | None = None
        self._encoded_xfer_handshake_metadata: dict[int, Any] = {}
        self._stop_event = threading.Event()
        self.physical_blocks_per_logical_kv_block: int = 1
        # a socket connect to remote prefill node, use this socket to send local block ids to remote[D use]
        self.peer_node_metas: dict[str, PeerMeta] = {}
        # the block infos will be send to remote[D use]
        # for now, "Any" could be: 1:(req_id, list[int]) for block_ids, 2: NodeMeta for connect
        self.send_to_peer_infos: asyncio.Queue[tuple[str, tuple[str, Any]]] = (
            asyncio.Queue()
        )

        # recv/send used
        self._reqs_to_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_to_send: dict[ReqId, tuple[int, bool, str]] = {}
        # Reqs to send and their expiration time
        self._reqs_to_send_time_record: dict[ReqId, float] = {}
        # The block info of local requests
        self._reqs_blocks: dict[ReqId, list[list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed_recv: set[ReqId] = set()
        self._reqs_not_send_to_worker: set[ReqId] = set()
        # Requests whose blocks are owned by the connector until workers
        # report finished_sending.  This keeps a producer-only engine stepping
        # even after its local prefill request has left the scheduler.
        self._pending_worker_sends: set[ReqId] = set()

        # Default per-layer unless cross or the env forces per-rank.
        self.prefer_layer_wise = not (
            _FORCE_PER_RANK_SEND or self._force_rank_wise_sends
        )

    def shutdown(self):
        self._stop_event.set()
        if self._scheduler_loop_t is not None:
            self._scheduler_loop_t.join()
            self._scheduler_loop_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        # NOTE(gyf): write not used heartbeat for now, it doesn't seem useful
        return

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        # NOTE(gyf): write not used heartbeat for now, it doesn't seem useful
        return

    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        """
        Clip the number of blocks to the sliding window size for each kv cache group
        that employs SWA.
        This is necessary because the KV Cache manager initially allocates blocks for
        the entire sequence length, and successively cleans up blocks that are outside
        the window prior to the `request_finished_all_groups` hook.
        """
        if len(block_ids) == 0 or not self._apply_block_id_preprocess:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        # For non-SWA groups, blocks_per_sw is (0,0) so we return all block_ids unchanged
        # for SWA groups, return [-start:][:len], where start == len
        # for Mamba groups, return [0:][:1], the official impl is ok, but we only need transfer the first block.
        return tuple(
            [
                blocks[self.blocks_per_sw[i][0] :][: self.blocks_per_sw[i][1]]
                if self.blocks_per_sw[i][1] > 0
                else blocks
                for i, blocks in enumerate(block_ids)
            ]
        )

    def _remote_block_starts(self, blocks: "KVCacheBlocks", ratio: int) -> list[int]:
        """Per group, index in the prompt's block list of the first block D asks for.

        D may load fewer tokens than the prompt (prefix hit at the head, and
        the scheduler aligns external tokens down to its block size under SWA
        bounded replay), so P cannot align on the tail of its own list.  -1
        marks groups whose lists are clipped (SWA, Mamba): P tail/head-aligns
        those as before.
        """
        starts: list[int] = []
        for i, group in enumerate(blocks.blocks):
            if self._apply_block_id_preprocess and self.blocks_per_sw[i][1] > 0:
                starts.append(-1)
                continue
            start = next(
                (
                    j
                    for j, block in enumerate(group)
                    if block.block_hash is None and not block.is_null
                ),
                len(group),
            )
            if ratio != 1 and self.remap_to_kernel_size[i] > 0:
                start *= ratio
            starts.append(start)
        return starts

    def remap_to_kernel_blocks(
        self, block_ids: BlockIds, token_lens: int, ratio: int, block_sizes: list[int]
    ) -> list[list[int]]:
        """
        Re-map the logical block ids to physical block ids for kernel use when physical_blocks_per_logical_kv_block > 1.
        for mamba, return.
        for others:
                eg: block_size 64, ratio 4, token_lens 65, block_ids original [1, 2], return [4, 5, 6, 7, 8] for cdiv(65, 64//4) = 5.
        """
        if ratio == 1:
            return block_ids

        kernel_block_offsets = np.arange(ratio)
        return [
            (
                np.array(blocks, dtype=np.int32).reshape(-1, 1) * ratio
                + kernel_block_offsets
            )
            .reshape(-1)[: cdiv(token_lens, block_sizes[i])]
            .tolist()
            if self.remap_to_kernel_size[i] > 0
            else list(blocks)
            for i, blocks in enumerate(block_ids)
        ]

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[int, bytes] = {}
        decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
        physical_block_size = self.block_size
        block_size = None
        for global_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            nixl_meta: NixlAgentMetadata = decoder.decode(
                rank_metadata.agent_metadata_bytes
            )
            if (
                self.physical_blocks_per_logical_kv_block
                != nixl_meta.physical_blocks_per_logical_kv_block
            ):
                self.physical_blocks_per_logical_kv_block = (
                    nixl_meta.physical_blocks_per_logical_kv_block
                )
            else:
                assert (
                    self.physical_blocks_per_logical_kv_block
                    == nixl_meta.physical_blocks_per_logical_kv_block
                )

            if physical_block_size != nixl_meta.physical_block_size:
                physical_block_size = nixl_meta.physical_block_size
            else:
                assert physical_block_size == nixl_meta.physical_block_size
            encoded_data[global_rank] = rank_metadata.agent_metadata_bytes
            logger.info(
                "global rank %d: encoded NixlHandshakePayload size: %s bytes",
                global_rank,
                str(len(encoded_data[global_rank])),
            )
            if not self._has_multi_size or self._has_mamba:
                if block_size is None:
                    block_size = (
                        self.block_size // self.physical_blocks_per_logical_kv_block
                    )
                else:
                    assert (
                        block_size
                        == self.block_size // self.physical_blocks_per_logical_kv_block
                    )

        self.block_size = block_size or self.block_size
        if self._has_multi_size and not self._has_mamba:
            assert self.physical_blocks_per_logical_kv_block == 1, (
                "physical block size != logical block size is not supported when kv spec has multi block size"
            )

        for index, remap in enumerate(self.remap_to_kernel_size):
            if remap:
                self.group_block_sizes[index] = (
                    self.group_block_sizes[index]
                    // self.physical_blocks_per_logical_kv_block
                )

        logger.info("the block size for each group are: %s", self.group_block_sizes)
        self.node_info["physical_block_size"] = physical_block_size

        # Only start the listener when we have metadata to serve.
        if self._scheduler_loop_t is None:
            self._scheduler_loop_t = threading.Thread(
                target=self.scheduler_loop,
                args=(metadata,),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._scheduler_loop_t.start()

    def _mamba_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.info(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        # D side
        if self.is_consumer:
            if params is not None and params.get("do_remote_prefill"):
                if not (params.get("transfer_id") or params.get("remote_request_id")):
                    logger.warning(
                        "PD request %s is missing transfer_id; P/D request IDs "
                        "may diverge and KV notifications may not match",
                        request.request_id,
                    )
                # Remote prefill: get all prompt blocks from remote.
                token_ids = request.prompt_token_ids or []
                actual = self._mamba_prefill_token_count(len(token_ids))
                count = actual - num_computed_tokens
                logger.info(
                    "NIXL D matched tokens for %s: prompt=%d, local_computed=%d, "
                    "external=%d, load_async=%s",
                    request.request_id,
                    len(token_ids),
                    num_computed_tokens,
                    max(count, 0),
                    count > 0,
                )
                if count > 0:
                    return count, True
                return 0, False
            else:
                return 0, False
        # P side
        else:
            if params is not None and params.get("do_remote_decode"):
                if not (params.get("transfer_id") or params.get("remote_request_id")):
                    logger.warning(
                        "PD request %s is missing transfer_id; P/D request IDs "
                        "may diverge and KV notifications may not match",
                        request.request_id,
                    )
                if self._has_mamba:
                    self._truncate_mamba_request_for_prefill(request)
                # Remote prefill: get all prompt blocks from remote.
                token_ids = request.prompt_token_ids or []
                layerwise = not (_FORCE_PER_RANK_SEND or self._force_rank_wise_sends)
                self._reqs_to_send[request.request_id] = (
                    len(token_ids),
                    layerwise,
                    params.get("remote_engine_id"),
                )
                # The producer computes KV locally. The async-load flag is only
                # for consumers waiting on remote KV and would make vLLM assert
                # that the producer has external cached tokens.
                return 0, False
            return 0, False

    def _get_or_connect_peer(self, remote_engine_id: str) -> PeerMeta:
        """Resolve one remote engine and enqueue CONNECT before block commands."""

        if peer_meta := self.peer_node_metas.get(remote_engine_id):
            return peer_meta

        with zmq_ctx(zmq.REQ, _PROXY_ADDR) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            peer_infos: dict[str, Any] = msgspec.msgpack.decode(
                zmq_comm2proxy(
                    sock,
                    msgspec.msgpack.encode((GET_NODE_META_MSG, remote_engine_id)),
                    need_recv=True,
                    logger=logger,
                )
            )

        # Both workers register scheduler blocks (kernel blocks live inside a
        # registered block), so destination ids travel unmodified and the two
        # engines must agree on the block size.  Multi-size groups are checked
        # per pointer by the worker handshake (layer_block_tokens_mapping).
        remote_physical_block_size = peer_infos["physical_block_size"]
        if not self._has_multi_size and remote_physical_block_size != self.block_size:
            raise ValueError(
                f"remote engine {remote_engine_id} block size "
                f"{remote_physical_block_size} != local block size "
                f"{self.block_size}; P and D must use the same block size"
            )
        remote_ratio = 1
        remote_block_sizes = list(self.group_block_sizes)

        remote_addr = peer_infos["side_channel_addr"]
        assert remote_addr != f"{self.side_channel_host}:{self.side_channel_port}"
        peer_socket = make_zmq_socket(
            path=make_zmq_path("tcp", remote_addr),
            ctx=self.ctx,
            socket_type=zmq.REQ,
            bind=False,
        )
        node_meta = NodeMeta(
            pp_size=self.node_info["pp_size"],
            tp_size=self.node_info["tp_size"],
            engine_id=self.engine_id,
            dp_rank=self.node_info["dp_rank"],
            pp_split=self.node_info["pp_split"],
            expire_time=self._kv_lease_duration,
            kv_group_infos=self.kv_group_infos,
        )
        peer_meta = PeerMeta(
            addr=remote_addr,
            block_ratio=remote_ratio,
            block_size=remote_block_sizes,
            pp_size=peer_infos["pp_size"],
            tp_size=peer_infos["tp_size"],
            pp_split=peer_infos["pp_split"],
            dp_rank=peer_infos.get("dp_rank", 0),
            socket=peer_socket,
        )
        self.peer_node_metas[remote_engine_id] = peer_meta
        self._async_loop.call_soon_threadsafe(
            self.send_to_peer_infos.put_nowait,
            (remote_engine_id, ("connect", node_meta)),
        )
        return peer_meta

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.info(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        # D side
        if self.is_consumer:
            if params.get("do_remote_prefill"):
                remote_engine_id = params.get("remote_engine_id")
                if not remote_engine_id:
                    raise ValueError(
                        f"request {request.request_id} is missing remote_engine_id"
                    )
                peer_meta = self._get_or_connect_peer(remote_engine_id)
                block_ids_send_to_remote = [[] for _ in range(self.kv_group_nums)]
                block_starts = [-1] * self.kv_group_nums
                intent = TransferIntent.NO_TRANSFER

                # Remote P writes KV only when D actually allocated external-token
                # destinations. NO_TRANSFER still travels to P so it can release
                # its send/block ownership without emitting a NIXL notification.
                if num_external_tokens > 0:
                    intent = TransferIntent.RECEIVE_KV
                    unhashed_local_block_ids = (
                        blocks.get_unhashed_block_ids_all_groups()
                    )
                    block_ids_send_to_remote = self.get_sw_clipped_blocks(
                        unhashed_local_block_ids
                    )
                    block_ids_send_to_remote = self.remap_to_kernel_blocks(
                        block_ids_send_to_remote,
                        num_external_tokens,
                        peer_meta.block_ratio,
                        peer_meta.block_size,
                    )
                    block_starts = self._remote_block_starts(
                        blocks, peer_meta.block_ratio
                    )
                    self._reqs_to_recv[request.request_id] = (
                        request,
                        block_ids_send_to_remote,
                    )
                    assert all(
                        len(block_id) for block_id in block_ids_send_to_remote
                    ), "partial empty KV groups are not supported yet"
                else:
                    self._reqs_not_send_to_worker.add(request.request_id)

                logger.debug(
                    "Sending transfer request_id %s intent %s with block ids %s",
                    request.request_id,
                    intent.value,
                    block_ids_send_to_remote,
                )
                self._async_loop.call_soon_threadsafe(
                    self.send_to_peer_infos.put_nowait,
                    (
                        remote_engine_id,
                        (
                            "block_ids",
                            (
                                request.request_id,
                                intent.value,
                                block_ids_send_to_remote,
                                self.engine_id,
                                block_starts,
                            ),
                        ),
                    ),
                )
                # Only trigger 1 KV transfer per request.
                params["do_remote_prefill"] = False
        # P side
        else:
            if params.get("do_remote_decode"):
                params["do_remote_decode"] = False

    def update_prefill_req_chunk(
        self,
        meta: NixlConnectorMetadata,
        request_id: ReqId,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        new_block_ids: tuple[list[int], ...],
    ):
        pass

    def update_prefill_req(
        self,
        meta: NixlConnectorMetadata,
        request_id: ReqId,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        new_block_ids: tuple[list[int], ...],
    ):
        # Plain requests served by the producer itself have nothing to send.
        if request_id not in self._reqs_to_send:
            return

        if new_block_ids is not None:
            for index, new_block_id in enumerate(new_block_ids):
                if request_id not in self._reqs_blocks:
                    self._reqs_blocks[request_id] = [
                        [] for _ in range(self.kv_group_nums)
                    ]
                self._reqs_blocks[request_id][index].extend(
                    new_block_id if new_block_id is not None else []
                )

        prefill_tokens, layer_wise, remote_engine_id = self._reqs_to_send[request_id]

        if num_computed_tokens + num_scheduled_tokens < prefill_tokens:
            return

        send_dst = meta.reqs_to_send if not layer_wise else meta.reqs_to_send_layerwise
        block_ids = self.get_sw_clipped_blocks(self._reqs_blocks[request_id])
        # if block size > kernel block size, re-map the block ids.
        # But local block size may bigger than remote block size, need processed on worker side.
        block_ids = self.remap_to_kernel_blocks(
            # self._reqs_blocks[request_id],
            block_ids,
            prefill_tokens,
            self.physical_blocks_per_logical_kv_block,
            self.group_block_sizes,
        )

        logger.debug("block ids send to local worker is: %s", block_ids)
        assert all(len(block_id) for block_id in block_ids), (
            "some group do not need send kv, not support for now, fixing..."
        )
        # P only need local block and remote_engine_id
        send_dst[request_id] = ReqMeta(
            block_ids,
            None,
            remote_engine_id,
            False,
            prefill_tokens,  # use for worker re-calculate block nums if local_block_size != remote_block_size
        )
        self._pending_worker_sends.add(request_id)
        # for write, start time record after schedule
        self._reqs_to_send_time_record[request_id] = (
            time.perf_counter() + self._kv_lease_duration
        )
        self._reqs_blocks.pop(request_id)
        self._reqs_to_send.pop(request_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # D side
        if self.is_consumer:
            # Loop through _reqs_to_recv and convert to ReqMeta.
            for req_id, (req, block_ids) in self._reqs_to_recv.items():
                assert req.kv_transfer_params is not None
                meta._add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    engine_id=req.kv_transfer_params.get("remote_engine_id"),
                )
            # Clear need recv
            self._reqs_to_recv.clear()
            meta.reqs_to_abort = self._reqs_not_processed_recv
            self._reqs_not_processed_recv = set()
        # P side
        else:
            # try add waiting to running requests first
            for new_req in scheduler_output.scheduled_new_reqs:
                self.update_prefill_req(
                    meta,
                    new_req.req_id,
                    scheduler_output.num_scheduled_tokens[new_req.req_id],
                    new_req.num_computed_tokens,
                    new_req.block_ids,
                )

            # add chunked done requests
            scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
            for idx, request_id in enumerate(scheduled_cached_reqs.req_ids):
                if request_id not in scheduled_cached_reqs.resumed_req_ids:
                    self.update_prefill_req(
                        meta,
                        request_id,
                        scheduler_output.num_scheduled_tokens[request_id],
                        scheduled_cached_reqs.num_computed_tokens[idx],
                        scheduled_cached_reqs.new_block_ids[idx],
                    )

            meta.reqs_to_send_time_record = self._reqs_to_send_time_record
            self._reqs_to_send_time_record = {}

        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Release scheduler-side ownership after worker WRITE completion."""
        for req_id in connector_output.finished_sending or ():
            self._pending_worker_sends.discard(req_id)

    def has_pending_push_work(self) -> bool:
        """Keep the engine loop alive while push work can still make progress."""
        return bool(
            self._reqs_to_send
            or self._reqs_blocks
            or self._reqs_to_send_time_record
            or self._pending_worker_sends
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        # D side
        if self.is_consumer:
            if params.get("do_remote_prefill"):
                # If do_remote_prefill is still True when the request is finished,
                # update_state_after_alloc must not have been called (the request
                # must have been aborted before it was scheduled).
                remote_engine_id = params.get("remote_engine_id")
                if not remote_engine_id:
                    return False, {}
                try:
                    self._get_or_connect_peer(remote_engine_id)
                except NodeError:
                    return False, {}
                empty_block_ids = [[] for _ in range(self.kv_group_nums)]
                self._async_loop.call_soon_threadsafe(
                    self.send_to_peer_infos.put_nowait,
                    (
                        remote_engine_id,
                        (
                            "block_ids",
                            (
                                request.request_id,
                                TransferIntent.CANCEL.value,
                                empty_block_ids,
                                self.engine_id,
                            ),
                        ),
                    ),
                )
                params["do_remote_prefill"] = False
                return False, None

            if (
                request.status
                not in (
                    RequestStatus.FINISHED_LENGTH_CAPPED,
                    RequestStatus.FINISHED_STOPPED,
                )
                and params.get("do_remote_prefill") is not None
            ):
                assert not params.get("do_remote_prefill")
                # for D, there are 2 cases.
                if request.request_id in self._reqs_not_send_to_worker:
                    # this req is full cache by local, when aborted, return False
                    self._reqs_not_send_to_worker.pop(request.request_id)
                    return False, {}
                else:
                    # wait kv cache write done from P
                    self._reqs_not_processed_recv.add(request.request_id)
                    return True, {}

            return False, None
        # P side
        else:
            if params.get("do_remote_decode"):
                params["do_remote_decode"] = False
                # when a reqs aborted, p workers will always get d block info and only notify info.:
                # len(remote_block_ids) > 0 for normal abort
                # len(remote_block_ids) == 0 for others, full cache then abort or abort before schedule.
                self._reqs_to_send.pop(request.request_id, None)
                self._reqs_blocks.pop(request.request_id, None)
                return False, None

            if (
                request.status
                not in (
                    RequestStatus.FINISHED_LENGTH_CAPPED,
                    RequestStatus.FINISHED_STOPPED,
                )
                and params.get("do_remote_decode") is not None
            ):
                assert not params.get("do_remote_decode")
                # for P, when P send info to worker then aborted, all we can do is wait kv write(may write) done
                return True, {}

            return any(len(block_id) for block_id in block_ids), {}

    def scheduler_loop(self, metadata: dict[int, KVConnectorHandshakeMetadata]):
        asyncio.set_event_loop(self._async_loop)

        async def tasks():
            background_tasks = {
                asyncio.create_task(self._proxy_connector(metadata)),
                asyncio.create_task(self._worker_data_forward()),
                asyncio.create_task(self._run_send_infos()),
            }
            done, pending = await asyncio.wait(
                background_tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            try:
                for task in done:
                    if error := task.exception():
                        raise error
            finally:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            self._async_loop.run_until_complete(tasks())
        except Exception:
            logger.fatal("An exception occurred in the scheduler background thread")
            traceback.print_exc()

    async def _proxy_connector(self, metadata: dict[int, KVConnectorHandshakeMetadata]):
        """
        TODO(gyf): add a zmq based to keep alive
        """
        while not self._stop_event.is_set():
            addr = _PROXY_ADDR.replace("tcp://", "")
            host, port = addr.rsplit(":", 1)
            # this port is used for keep alive
            addr = host + ":" + str(int(port) + 1)
            try:
                connected = False
                reader, writer = await asyncio.open_connection(host, int(port) + 1)
                connected = True
                writer.write(
                    msgspec.msgpack.encode(
                        (PUT_NODE_META_MSG, self.node_info, metadata)
                    )
                )
                writer.write(b"info_done\0")
                await writer.drain()
                await reader.readexactly(1)
            except Exception as e:
                logger.error("%s", e)
                logger.error("Cannot connect to: %s(use PORT+1 to keep alive)", addr)
                await asyncio.sleep(1)
            finally:
                if connected:
                    writer.close()
                    await writer.wait_closed()

    async def _worker_data_forward(
        self,
    ):
        """Background thread for pub info to workers."""
        # a socket publish infos to workers, tcp://ip:port
        scheduler_addr = _SCHEDULER_ADDR
        path = make_zmq_path("tcp", self.local_addr)
        with (
            make_zmq_socket(
                path=path,
                ctx=self.ctx,
                socket_type=zmq.ROUTER,
                bind=True,
            ) as recv_from,
            make_zmq_socket(
                path=scheduler_addr,
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_to_connector,
        ):
            # get sub from worker
            for _ in range(self.node_info["tp_size"] * self.node_info["pp_size"]):
                if await publish_to_connector.recv() != b"\x01":
                    raise RuntimeError(
                        "NIXL handshake listener failed to connect to connector"
                    )
            await publish_to_connector.send(b"READY")
            recv_from.setsockopt(zmq.RCVTIMEO, 1000)

            while True:
                try:
                    identity, _, data = await recv_from.recv_multipart()
                    # do not want to decoding the data when get a msg from proxy,
                    # because scheduler only need "disconnect" msg, a little trick here.
                    if (
                        len(data) >= len(DISCONNECT_MSG)
                        and data[: len(DISCONNECT_MSG)] == DISCONNECT_MSG
                    ):
                        _, remote_engine_id = msgspec.msgpack.decode(data)
                        self.send_to_peer_infos.put_nowait((remote_engine_id, None))
                    await recv_from.send_multipart((identity, b"", b""))
                    await publish_to_connector.send(data)
                    logger.info(
                        "NIXL scheduler forwarded %d bytes to workers (op=%s)",
                        len(data),
                        msgspec.msgpack.decode(data)[0],
                    )
                except zmq.Again:
                    if self._stop_event.is_set():
                        break
                    continue
                except Exception as e:
                    logger.warning("NIXL scheduler forward failed: %r", e)

    async def _run_send_infos(self):
        while not self._stop_event.is_set():
            remote_engine_id, send_infos = await self.send_to_peer_infos.get()
            if send_infos is None:
                # A producer never opens a peer socket to its decoders, so the
                # disconnect of one may have nothing to close.
                peer_meta = self.peer_node_metas.pop(remote_engine_id, None)
                if peer_meta is not None:
                    with suppress(BaseException):
                        peer_meta.socket.close()
                continue
            # when remote disconnect, send_to_peer_infos may still have some infos to send, we drop these infos.
            peer_meta: PeerMeta = self.peer_node_metas.get(remote_engine_id, None)
            logger.info(
                "NIXL scheduler send %s to %s (%s)",
                send_infos[0],
                remote_engine_id,
                "dropped, peer gone" if peer_meta is None else peer_meta.addr,
            )
            if peer_meta is not None:
                await peer_meta.socket.send(msgspec.msgpack.encode(send_infos))
                with suppress(BaseException):
                    # remote may closed when before get disconnect msg, recv msg form a closed remote socket will
                    # get a error
                    await peer_meta.socket.recv()
