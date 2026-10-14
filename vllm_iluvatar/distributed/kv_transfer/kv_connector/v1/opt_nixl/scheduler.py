# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side logic for the NIXL connector."""

import asyncio
import hashlib
import os
import queue
import threading
import time
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import msgspec
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
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from .metadata import (
    GET_NODE_META_MSG,
    OWNERSHIP_CANCEL_MSG,
    OWNERSHIP_RESULT_ACCEPTED,
    OWNERSHIP_RESULT_CANCELED,
    OWNERSHIP_RESULT_TERMINAL,
    PUT_NODE_META_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NodeMeta,
    PeerMeta,
    ReqId,
    ReqMeta,
    TEARDOWN_COMPLETE_MSG,
    TEARDOWN_RESULT_FAILED,
    TEARDOWN_RESULT_OK,
    TransferIntent,
)
from .lifecycle import RequestOwnershipState, RequestTombstoneStore
from .utils import (
    NodeError,
    dp_scoped_addr,
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
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger("vllm." + __name__)


def transfer_slice(spec: "KVCacheSpec") -> slice | None:
    """The part of a group's block list P has to hand over, or None for all.

    Three groups are clipped:

    * sliding window: the KV cache manager allocates the whole sequence and
      frees the blocks that drop out of the window before the
      ``request_finished_all_groups`` hook, so only the last n survive.
    * SSM/Mamba: one block holds the whole state, so the first one is enough.
    * outside prefix caching (DeepSeek-V4.1 SWA under bounded replay, HiSparse
      replicas): D rebuilds the group locally, so nothing has to travel.

    An empty slice keeps the group's seat in the block-id tuple -- P and D
    index that tuple by KV cache group id -- it does not drop the group.
    """
    if not spec.prefix_cacheable:
        return slice(0, 0)
    if isinstance(spec, SlidingWindowSpec):
        return slice(-(cdiv(spec.sliding_window, spec.block_size) + 1), None)
    if isinstance(spec, MambaSpec):
        return slice(0, 1)
    return None


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
        dp_index = vllm_config.parallel_config.data_parallel_index
        self.side_channel_port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT + dp_index
        # Each DP rank binds its own scheduler->worker publisher.
        self.scheduler_addr = dp_scoped_addr(_SCHEDULER_ADDR, dp_index)
        self.scheduler_addr = dp_scoped_addr(
            _SCHEDULER_ADDR, vllm_config.parallel_config.data_parallel_index
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

        # Per KV cache group, the slice of a request's block list that P has
        # to transfer.  ``None`` means the whole list; see ``transfer_slice``.
        self.transfer_slices: list[slice | None] = [None] * self.kv_group_nums
        if self._apply_block_id_preprocess:
            self.transfer_slices = [
                transfer_slice(g.kv_cache_spec) for g in kv_cache_config.kv_cache_groups
            ]
        logger.info(
            "NIXL block clipping: %s of %d KV groups transfer nothing, %s are "
            "clipped to a window",
            [i for i, clip in enumerate(self.transfer_slices) if clip == slice(0, 0)],
            self.kv_group_nums,
            [
                i
                for i, clip in enumerate(self.transfer_slices)
                if clip is not None and clip != slice(0, 0)
            ],
        )

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
        # Partial prompt blocks are private receive destinations; the even hit
        # keeps CR2 replay aligned and its first SWA block live.
        replay_specs = [s for s in self._all_specs if s.prefix_replay_tokens]
        self.prefix_replay_alignment = (
            2
            if os.getenv("VLLM_ILU_V41_REMOTE_REPLAY_ALIGN2", "0") == "1"
            and self.is_consumer
            and vllm_config.model_config.hf_config.model_type == "deepseek_v41"
            and not self._has_mamba
            and replay_specs
            and all(
                isinstance(s, SlidingWindowMLASpec)
                and s.prefix_replay_tokens == 128
                and s.block_size == 64
                for s in replay_specs
            )
            and all(
                isinstance(
                    s,
                    (SlidingWindowMLASpec, MLAAttentionSpec, CircularBufferSpec),
                )
                and getattr(s, "tokens_per_state", 1) in (1, 2)
                for s in self._all_specs
            )
            else None
        )
        logger.info("Remote-prefix replay alignment: %s", self.prefix_replay_alignment)
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
        self._teardown_timeout = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                # A 256K in-flight transfer can delay a worker's connector
                # control loop for more than two minutes.
                "reconnect_teardown_timeout",
                300.0,
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
            "teardown_protocol_version": 1,
            "kv_group_infos": self.kv_group_infos,  # the p/d kv group must be the same
        }

        # ---- transferring ----
        # Background thread for handling new handshake requests.
        self.ctx = zmq.asyncio.Context()
        self._async_loop = uvloop.new_event_loop()
        self._scheduler_loop_t: threading.Thread | None = None
        self._shutdown_async_event: asyncio.Event | None = None
        self._encoded_xfer_handshake_metadata: dict[int, Any] = {}
        self._stop_event = threading.Event()
        # a socket connect to remote prefill node, use this socket to send local block ids to remote[D use]
        self.peer_node_metas: dict[str, PeerMeta] = {}
        # the block infos will be send to remote[D use]
        # for now, "Any" could be: 1:(req_id, list[int]) for block_ids, 2: NodeMeta for connect
        self.send_to_peer_infos: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()

        # recv/send used
        self._reqs_to_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_to_send: dict[ReqId, tuple[int, bool, str, str]] = {}
        # Reqs to send and their expiration time
        self._reqs_to_send_time_record: dict[ReqId, float] = {}
        # The block info of local requests
        self._reqs_blocks: dict[ReqId, list[list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed_recv: set[ReqId] = set()
        self._reqs_not_send_to_worker: set[ReqId] = set()
        # A P can disappear after the router selected it but before D asks the
        # proxy for its metadata. Report that request as a failed async receive
        # on the next connector output instead of killing the EngineCore.
        self._scheduler_failed_recvs: set[ReqId] = set()
        # Requests whose blocks are owned by the connector until workers
        # report finished_sending.  This keeps a producer-only engine stepping
        # even after its local prefill request has left the scheduler.
        self._pending_worker_sends: set[ReqId] = set()
        # Live D-side requests waiting for remote KV. This is ownership state,
        # not a lease renewal: a stuck transfer must still expire and fail.
        self._live_recv_reqs: set[ReqId] = set()
        # Request-generation ownership is scheduler-owned. A timeout requests
        # cancellation; it never releases D destination blocks by itself.
        self._ownership_generations: dict[ReqId, str] = {}
        self._ownership_remote_engines: dict[ReqId, str] = {}
        self._ownership_states: dict[ReqId, RequestOwnershipState] = {}
        self._ownership_deadlines: dict[ReqId, float] = {}
        self._ownership_cancel_pending: set[ReqId] = set()
        self._ownership_safe_releases: queue.SimpleQueue[
            tuple[ReqId, str, str]
        ] = queue.SimpleQueue()
        # P-side worker completion is delivered from the scheduler main thread
        # to the side-channel actor, which may be holding a CANCEL waiter.
        self._ownership_terminal_events: asyncio.Queue[
            tuple[ReqId, str, str]
        ] = asyncio.Queue()
        self._ownership_tombstone_ttl = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "ownership_tombstone_ttl", 86400.0
            )
        )
        self._ownership_tombstone_limit = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "ownership_tombstone_limit", 262144
            )
        )

        # Default per-layer unless cross or the env forces per-rank.
        self.prefer_layer_wise = not (
            _FORCE_PER_RANK_SEND or self._force_rank_wise_sends
        )

    def shutdown(self):
        self._stop_event.set()
        shutdown_event = self._shutdown_async_event
        if shutdown_event is not None:
            with suppress(RuntimeError):
                self._async_loop.call_soon_threadsafe(shutdown_event.set)
        scheduler_thread = self._scheduler_loop_t
        if scheduler_thread is not None:
            scheduler_thread.join(timeout=5.0)
            if scheduler_thread.is_alive():
                logger.error(
                    "NIXL scheduler background thread did not exit within 5 seconds"
                )
                return
            self._scheduler_loop_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track live D-side requests that are waiting for remote KV."""
        params = request.kv_transfer_params
        if self.is_consumer and params and params.get("do_remote_prefill"):
            self._live_recv_reqs.add(request.request_id)
            generation = self._request_generation(request.request_id, params)
            self._ownership_generations[request.request_id] = generation
            remote_engine_id = params.get("remote_engine_id")
            if remote_engine_id:
                self._ownership_remote_engines[request.request_id] = str(
                    remote_engine_id
                )
            self._ownership_states[request.request_id] = (
                RequestOwnershipState.PREPARED
            )

    def _stop_recv_tracking(self, req_id: ReqId) -> None:
        """Stop tracking a completed or cancelled local receive."""
        self._live_recv_reqs.discard(req_id)
        self._ownership_deadlines.pop(req_id, None)
        self._ownership_cancel_pending.discard(req_id)
        self._ownership_generations.pop(req_id, None)
        self._ownership_remote_engines.pop(req_id, None)
        self._ownership_states.pop(req_id, None)

    @staticmethod
    def _request_generation(req_id: ReqId, params: dict[str, Any]) -> str:
        if generation := params.get("_opt_nixl_generation"):
            return str(generation)
        identity = str(
            params.get("transfer_id")
            or params.get("remote_request_id")
            or req_id
        )
        # Router request IDs contain host:port pairs. Keep the generation
        # delimiter-safe for the native notification wire while preserving
        # the request ID's UUID-level uniqueness on both P and D.
        generation = hashlib.sha256(identity.encode()).hexdigest()
        params["_opt_nixl_generation"] = generation
        return generation

    def _queue_peer_command(
        self, remote_engine_id: str, command: tuple[str, Any]
    ) -> None:
        """Enqueue one side-channel command from the scheduler main thread."""

        self._async_loop.call_soon_threadsafe(
            self.send_to_peer_infos.put_nowait,
            (remote_engine_id, command, 0),
        )

    def _request_ownership_cancel(self, req_id: ReqId, reason: str) -> None:
        """Ask P for a proof that no WRITE can still target this generation."""

        if req_id in self._ownership_cancel_pending:
            return
        generation = self._ownership_generations.get(req_id)
        remote_engine_id = self._ownership_remote_engines.get(req_id)
        if not generation or not remote_engine_id:
            logger.error(
                "Cannot cancel KV ownership for %s: generation=%r remote=%r",
                req_id,
                generation,
                remote_engine_id,
            )
            return
        self._ownership_cancel_pending.add(req_id)
        self._queue_peer_command(
            remote_engine_id,
            (
                OWNERSHIP_CANCEL_MSG,
                (req_id, generation, self.engine_id, reason),
            ),
        )

    def _expire_recv_ownership(self, now: float) -> None:
        for req_id, expires in list(self._ownership_deadlines.items()):
            if now < expires or req_id not in self._live_recv_reqs:
                continue
            logger.warning(
                "NIXL receive lease expired for request %s; requesting "
                "producer terminal proof before releasing destination blocks",
                req_id,
            )
            self._request_ownership_cancel(req_id, "receive lease expired")

    def _drain_safe_releases(self, meta: NixlConnectorMetadata) -> None:
        while True:
            try:
                req_id, generation, reason = self._ownership_safe_releases.get_nowait()
            except queue.Empty:
                break
            if self._ownership_generations.get(req_id) != generation:
                logger.debug(
                    "Ignoring stale ownership proof for %s generation %s",
                    req_id,
                    generation,
                )
                continue
            self._ownership_states[req_id] = RequestOwnershipState.TERMINAL
            self._ownership_deadlines.pop(req_id, None)
            meta.reqs_to_release[req_id] = (generation, reason)

    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        """Clip every group's block list down to what P actually has to send.

        See ``transfer_slice`` for the three clipped cases.  Groups that D
        rebuilds locally come back empty; any other group stays non-empty,
        which is what downstream reads as "this group is covered".
        """
        if len(block_ids) == 0:
            # No blocks to clip, eg a full prefix cache hit.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.transfer_slices), (
            "Number of KV cache groups must match"
        )
        return tuple(
            blocks if (clip := self.transfer_slices[i]) is None else blocks[clip]
            for i, blocks in enumerate(block_ids)
        )

    def _remote_block_starts(self, blocks: "KVCacheBlocks") -> list[int]:
        """Per group, index in the prompt's block list of the first block D asks for.

        D's list is neither a prefix nor a suffix of P's: the head is a prefix
        hit, and under SWA bounded replay the scheduler aligns the external
        token count down to a block multiple and replays the tail itself.  So
        P cannot align on either end of its own list and needs the offset.
        -1 marks the groups whose lists are clipped (sliding window, Mamba) or
        empty; those keep the old tail/head alignment.
        """
        starts: list[int] = []
        for i, group in enumerate(blocks.blocks):
            if self.transfer_slices[i] is not None:
                starts.append(-1)
                continue
            starts.append(
                next(
                    (
                        j
                        for j, block in enumerate(group)
                        if block.block_hash is None and not block.is_null
                    ),
                    len(group),
                )
            )
        return starts

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
        for global_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            nixl_meta: NixlAgentMetadata = decoder.decode(
                rank_metadata.agent_metadata_bytes
            )
            # Our workers register scheduler blocks, so a kernel block lives
            # inside a registered block and block ids travel unmodified.
            if nixl_meta.physical_blocks_per_logical_kv_block != 1:
                raise ValueError(
                    f"rank {global_rank} reports "
                    f"{nixl_meta.physical_blocks_per_logical_kv_block} physical "
                    "blocks per logical KV block; NIXL registers scheduler "
                    "blocks and only supports 1"
                )
            if nixl_meta.physical_block_size != physical_block_size:
                raise ValueError(
                    f"rank {global_rank} block size "
                    f"{nixl_meta.physical_block_size} != {physical_block_size}; "
                    "all ranks of an engine must agree"
                )
            encoded_data[global_rank] = rank_metadata.agent_metadata_bytes
            logger.info(
                "global rank %d: encoded NixlHandshakePayload size: %s bytes",
                global_rank,
                str(len(encoded_data[global_rank])),
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
                generation = self._request_generation(
                    request.request_id, params
                )
                self._ownership_generations[request.request_id] = generation
                self._reqs_to_send[request.request_id] = (
                    len(token_ids),
                    layerwise,
                    params.get("remote_engine_id"),
                    generation,
                )
                # The producer computes KV locally. The async-load flag is only
                # for consumers waiting on remote KV and would make vLLM assert
                # that the producer has external cached tokens.
                return 0, False
            return 0, False

    def _decode_peer_infos(
        self, remote_engine_id: str, encoded_peer_infos: bytes
    ) -> dict[str, Any]:
        """Decode and validate metadata without leaking protocol errors."""
        try:
            peer_infos = msgspec.msgpack.decode(encoded_peer_infos)
            if not isinstance(peer_infos, dict):
                raise TypeError("peer metadata is not a mapping")

            required = (
                "physical_block_size",
                "side_channel_addr",
                "pp_size",
                "tp_size",
                "pp_split",
            )
            missing = [key for key in required if key not in peer_infos]
            if missing:
                raise KeyError(f"missing fields: {', '.join(missing)}")

            for key in ("physical_block_size", "pp_size", "tp_size"):
                value = peer_infos[key]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise TypeError(f"{key} must be a positive integer")

            dp_rank = peer_infos.get("dp_rank", 0)
            if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
                raise TypeError("dp_rank must be a non-negative integer")

            pp_split = peer_infos["pp_split"]
            if not isinstance(pp_split, list) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in pp_split
            ):
                raise TypeError("pp_split must be a list of integers")

            remote_addr = peer_infos["side_channel_addr"]
            if not isinstance(remote_addr, str) or not remote_addr:
                raise TypeError("side_channel_addr must be a non-empty string")
            host, sep, port = remote_addr.rpartition(":")
            if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
                raise ValueError("side_channel_addr must be host:port")
            if remote_addr == f"{self.side_channel_host}:{self.side_channel_port}":
                raise ValueError("remote side channel resolves to the local scheduler")

            remote_block_size = peer_infos["physical_block_size"]
            if not self._has_multi_size and remote_block_size != self.block_size:
                raise ValueError(
                    f"block size {remote_block_size} != local block size "
                    f"{self.block_size}"
                )
        except (msgspec.DecodeError, KeyError, TypeError, ValueError) as error:
            raise NodeError(
                f"invalid metadata for remote engine {remote_engine_id}: {error}"
            ) from error
        return peer_infos

    def _get_or_connect_peer(self, remote_engine_id: str) -> PeerMeta:
        """Resolve one remote engine and enqueue CONNECT before block commands."""

        if peer_meta := self.peer_node_metas.get(remote_engine_id):
            return peer_meta

        with zmq_ctx(zmq.REQ, _PROXY_ADDR) as sock:
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDTIMEO, 5000)
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            encoded_peer_infos = zmq_comm2proxy(
                sock,
                msgspec.msgpack.encode((GET_NODE_META_MSG, remote_engine_id)),
                need_recv=True,
                logger=logger,
            )
        peer_infos = self._decode_peer_infos(remote_engine_id, encoded_peer_infos)

        remote_addr = peer_infos["side_channel_addr"]
        peer_socket = None
        try:
            peer_socket = make_zmq_socket(
                path=make_zmq_path("tcp", remote_addr),
                ctx=self.ctx,
                socket_type=zmq.REQ,
                bind=False,
            )
            peer_socket.setsockopt(zmq.LINGER, 0)
            peer_socket.setsockopt(zmq.SNDTIMEO, 3000)
            peer_socket.setsockopt(zmq.RCVTIMEO, 3000)
        except (RuntimeError, ValueError, zmq.ZMQError) as error:
            if peer_socket is not None:
                peer_socket.close(linger=0)
            raise NodeError(
                f"failed to create side channel for remote engine {remote_engine_id}"
            ) from error

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
            pp_size=peer_infos["pp_size"],
            tp_size=peer_infos["tp_size"],
            pp_split=peer_infos["pp_split"],
            dp_rank=peer_infos.get("dp_rank", 0),
            socket=peer_socket,
        )
        self.peer_node_metas[remote_engine_id] = peer_meta
        try:
            self._async_loop.call_soon_threadsafe(
                self.send_to_peer_infos.put_nowait,
                (remote_engine_id, ("connect", node_meta)),
            )
        except RuntimeError as error:
            self.peer_node_metas.pop(remote_engine_id, None)
            peer_socket.close(linger=0)
            raise NodeError(
                f"failed to enqueue side-channel connect for {remote_engine_id}"
            ) from error
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
                generation = self._request_generation(
                    request.request_id, params
                )
                try:
                    if not remote_engine_id:
                        raise NodeError("request is missing remote_engine_id")
                    self._get_or_connect_peer(remote_engine_id)
                except NodeError as error:
                    logger.warning(
                        "Remote prefill setup failed for request %s (engine=%s): %s; "
                        "failing only this KV receive",
                        request.request_id,
                        remote_engine_id,
                        error,
                    )
                    params["do_remote_prefill"] = False
                    self._reqs_not_send_to_worker.add(request.request_id)
                    self._scheduler_failed_recvs.add(request.request_id)
                    return
                self._ownership_generations[request.request_id] = generation
                self._ownership_remote_engines[request.request_id] = str(
                    remote_engine_id
                )
                self._ownership_states[request.request_id] = (
                    RequestOwnershipState.PREPARED
                )
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
                    block_starts = self._remote_block_starts(blocks)
                    self._reqs_to_recv[request.request_id] = (
                        request,
                        block_ids_send_to_remote,
                    )
                    self._ownership_deadlines[request.request_id] = (
                        time.perf_counter() + self._kv_lease_duration
                    )
                    # Replay groups are deliberately empty; a request with no
                    # destination at all would leave P nothing to write.
                    assert any(
                        len(block_id) for block_id in block_ids_send_to_remote
                    ), "D allocated external tokens but no KV group has blocks"
                else:
                    self._reqs_not_send_to_worker.add(request.request_id)

                logger.debug(
                    "Sending transfer request_id %s intent %s with block ids %s",
                    request.request_id,
                    intent.value,
                    block_ids_send_to_remote,
                )
                self._queue_peer_command(
                    str(remote_engine_id),
                    (
                        "block_ids",
                        (
                            request.request_id,
                            generation,
                            intent.value,
                            block_ids_send_to_remote,
                            self.engine_id,
                            block_starts,
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

        (
            prefill_tokens,
            layer_wise,
            remote_engine_id,
            generation,
        ) = self._reqs_to_send[request_id]

        if num_computed_tokens + num_scheduled_tokens < prefill_tokens:
            return

        send_dst = meta.reqs_to_send if not layer_wise else meta.reqs_to_send_layerwise
        block_ids = self.get_sw_clipped_blocks(self._reqs_blocks[request_id])

        logger.debug("block ids send to local worker is: %s", block_ids)
        # Replay groups are deliberately empty; a request with no source at all
        # would mean P holds no KV for it.
        assert any(len(block_id) for block_id in block_ids), (
            "no KV group has blocks to send"
        )
        # P only need local block and remote_engine_id
        send_dst[request_id] = ReqMeta(
            block_ids,
            None,
            remote_engine_id,
            False,
            prefill_tokens,
            generation,
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
            self._expire_recv_ownership(time.perf_counter())
            # Loop through _reqs_to_recv and convert to ReqMeta.
            for req_id, (req, block_ids) in self._reqs_to_recv.items():
                assert req.kv_transfer_params is not None
                meta._add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    engine_id=req.kv_transfer_params.get("remote_engine_id"),
                    generation=self._ownership_generations.get(req_id, ""),
                )
            # Clear need recv
            self._reqs_to_recv.clear()
            meta.reqs_to_abort = self._reqs_not_processed_recv
            self._reqs_not_processed_recv = set()
            self._drain_safe_releases(meta)
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
        """Release scheduler-side ownership after worker WRITE completion.

        Also drops completions for sends this engine is not waiting on.  The
        worker reports one no matter how a send ends, including a lease that
        expires long after the request left the scheduler, and the scheduler
        then does ``assert req_id in self.requests`` and kills the engine.  A
        send is in ``_pending_worker_sends`` from the moment it is handed to the
        worker, so anything outside that set is a completion whose blocks are
        already freed and must not be reported again.
        """
        # Scheduler-originated failures are valid only while this scheduler
        # still owns the receive. A cancel may remove the request before the
        # next connector output; never surface such a stale completion.
        failed = self._scheduler_failed_recvs & self._live_recv_reqs
        stale_failed = self._scheduler_failed_recvs - failed
        if stale_failed:
            logger.debug(
                "Dropping stale scheduler receive failures: %s",
                sorted(stale_failed),
            )
        self._scheduler_failed_recvs.clear()
        if failed:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = failed.copy()
            else:
                connector_output.finished_recving.update(failed)
            if connector_output.failed_recving is None:
                connector_output.failed_recving = failed
            else:
                connector_output.failed_recving.update(failed)

        for req_id in connector_output.finished_recving or ():
            self._stop_recv_tracking(req_id)
        for req_id in connector_output.failed_recving or ():
            self._stop_recv_tracking(req_id)

        finished = connector_output.finished_sending
        if not finished:
            return
        # Multiproc connector output reports a request only after every local
        # TP/PP worker reached a native terminal state. Publish that proof
        # before filtering scheduler-stale completions.
        for req_id in set(finished):
            generation = self._ownership_generations.get(req_id)
            if generation:
                self._ownership_states[req_id] = RequestOwnershipState.TERMINAL
                with suppress(RuntimeError):
                    self._async_loop.call_soon_threadsafe(
                        self._ownership_terminal_events.put_nowait,
                        (
                            req_id,
                            generation,
                            OWNERSHIP_RESULT_TERMINAL,
                        ),
                    )
        stale = finished - self._pending_worker_sends
        if stale:
            # A leak would show up here as a request that is never freed, so
            # this is loud rather than silent.
            logger.warning(
                "Ignoring send completion for %d request(s) this engine no "
                "longer owns: %s",
                len(stale),
                sorted(stale),
            )
            finished -= stale
        self._pending_worker_sends -= finished

    def has_pending_push_work(self) -> bool:
        """Keep the engine loop alive while push work can still make progress."""
        return bool(
            self._reqs_to_send
            or self._reqs_blocks
            or self._reqs_to_send_time_record
            or self._pending_worker_sends
            or self._live_recv_reqs
            or self._ownership_cancel_pending
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
            self._stop_recv_tracking(request.request_id)
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
                    self._stop_recv_tracking(request.request_id)
                    return False, {}
                generation = self._request_generation(
                    request.request_id, params
                )
                self._ownership_generations[request.request_id] = generation
                self._ownership_remote_engines[request.request_id] = str(
                    remote_engine_id
                )
                self._request_ownership_cancel(
                    request.request_id, "request aborted before allocation"
                )
                params["do_remote_prefill"] = False
                # No destination block was disclosed, so local release is
                # already safe; the best-effort cancel only frees P sooner.
                self._stop_recv_tracking(request.request_id)
                return False, None

            # Dropped on every finish, not only on abort: a fully locally cached
            # request that finishes normally would otherwise stay here forever.
            fully_local = request.request_id in self._reqs_not_send_to_worker
            self._reqs_not_send_to_worker.discard(request.request_id)
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
                if fully_local:
                    # this req is full cache by local, when aborted, return False
                    self._stop_recv_tracking(request.request_id)
                    return False, {}
                else:
                    # Keep D blocks until P proves every native WRITE terminal.
                    self._request_ownership_cancel(
                        request.request_id, "consumer request aborted"
                    )
                    self._reqs_not_processed_recv.add(request.request_id)
                    return True, {}

            self._stop_recv_tracking(request.request_id)
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
        shutdown_event = asyncio.Event()
        self._shutdown_async_event = shutdown_event
        if self._stop_event.is_set():
            shutdown_event.set()

        async def tasks():
            background_tasks = {
                asyncio.create_task(self._proxy_connector(metadata)),
                asyncio.create_task(self._worker_data_forward()),
                asyncio.create_task(self._run_send_infos()),
                asyncio.create_task(shutdown_event.wait()),
            }
            done, pending = await asyncio.wait(
                background_tasks,
                return_when=asyncio.FIRST_COMPLETED,
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
        finally:
            self._shutdown_async_event = None

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
        """Forward peer control messages and return a real teardown ACK."""
        # a socket publish infos to workers, tcp://ip:port
        scheduler_addr = self.scheduler_addr
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
            recv_from.setsockopt(zmq.RCVTIMEO, 100)
            expected_workers = self.node_info["tp_size"] * self.node_info["pp_size"]
            pending: dict[int, dict[str, Any]] = {}
            completed: dict[int, tuple[str, bytes]] = {}
            ownership_waiters: dict[tuple[ReqId, str], set[bytes]] = {}
            ownership_tombstones = RequestTombstoneStore(
                self._ownership_tombstone_ttl,
                self._ownership_tombstone_limit,
            )

            async def reply(identity: bytes, payload: bytes) -> None:
                await recv_from.send_multipart((identity, b"", payload))

            def remember_completed(
                teardown_id: int, engine_id: str, result: bytes
            ) -> None:
                completed[teardown_id] = (engine_id, result)
                if len(completed) > 4096:
                    completed.pop(next(iter(completed)))

            async def expire_teardowns() -> None:
                now = time.monotonic()
                for teardown_id, state in list(pending.items()):
                    if now < state["deadline"]:
                        continue
                    logger.error(
                        "NIXL teardown %s timed out for %s: %d/%d worker ACKs",
                        teardown_id,
                        state["engine_id"],
                        len(state["acks"]),
                        expected_workers,
                    )
                    for waiter in state["waiters"]:
                        await reply(waiter, TEARDOWN_RESULT_FAILED)
                    remember_completed(
                        teardown_id,
                        state["engine_id"],
                        TEARDOWN_RESULT_FAILED,
                    )
                    del pending[teardown_id]

            async def complete_ownership_events() -> None:
                while True:
                    try:
                        req_id, generation, result = (
                            self._ownership_terminal_events.get_nowait()
                        )
                    except asyncio.QueueEmpty:
                        break
                    state = (
                        RequestOwnershipState.CANCELED
                        if result == OWNERSHIP_RESULT_CANCELED
                        else RequestOwnershipState.TERMINAL
                    )
                    try:
                        ownership_tombstones.record(req_id, generation, state)
                    except RuntimeError as error:
                        # Fail closed: D receives no proof and keeps its blocks.
                        logger.critical(
                            "Unable to retain ownership tombstone for %s/%s: %s",
                            req_id,
                            generation,
                            error,
                        )
                        continue
                    encoded = msgspec.msgpack.encode((state.value, generation))
                    for waiter in ownership_waiters.pop(
                        (req_id, generation), set()
                    ):
                        with suppress(zmq.ZMQError):
                            await reply(waiter, encoded)
                    if self._ownership_generations.get(req_id) == generation:
                        self._ownership_generations.pop(req_id, None)
                        self._ownership_remote_engines.pop(req_id, None)
                        self._ownership_states.pop(req_id, None)

            while True:
                try:
                    await complete_ownership_events()
                    identity, _, data = await recv_from.recv_multipart()
                    message = msgspec.msgpack.decode(data)
                    # E25 uses this existing P/D endpoint. State messages are
                    # handled locally and never published as worker commands.
                    from vllm_iluvatar.v1.worker.prefill_state.source_state import (
                        GET_STATE,
                        serve_state,
                    )

                    if (
                        isinstance(message, (tuple, list))
                        and message
                        and message[0] == GET_STATE
                    ):
                        state_reply, state_payload = serve_state(
                            getattr(self, "_p21_wire", None)
                            if not self.is_consumer
                            else None,
                            self.engine_id,
                            message,
                        )
                        await recv_from.send_multipart(
                            (
                                identity,
                                b"",
                                msgspec.msgpack.encode(state_reply),
                                state_payload,
                            )
                        )
                        continue
                    op_type, payload = message

                    if op_type == TEARDOWN_COMPLETE_MSG:
                        (
                            remote_engine_id,
                            generation,
                            teardown_id,
                            rank,
                            status,
                        ) = payload
                        # Worker REQ sockets only need delivery acknowledgement.
                        await reply(identity, b"")
                        state = pending.get(int(teardown_id))
                        if (
                            state is None
                            or state["engine_id"] != remote_engine_id
                            or not 0 <= int(rank) < expected_workers
                        ):
                            continue
                        state["acks"][int(rank)] = str(status)
                        logger.info(
                            "NIXL teardown ACK id=%s engine=%s generation=%s "
                            "rank=%s status=%s (%d/%d)",
                            teardown_id,
                            remote_engine_id,
                            generation,
                            rank,
                            status,
                            len(state["acks"]),
                            expected_workers,
                        )
                        if len(state["acks"]) >= expected_workers:
                            result = (
                                TEARDOWN_RESULT_OK
                                if all(
                                    item == "closed"
                                    for item in state["acks"].values()
                                )
                                else TEARDOWN_RESULT_FAILED
                            )
                            for waiter in state["waiters"]:
                                await reply(waiter, result)
                            remember_completed(
                                int(teardown_id), remote_engine_id, result
                            )
                            del pending[int(teardown_id)]
                        continue

                    if op_type == "block_ids":
                        if len(payload) < 5:
                            raise ValueError("block_ids ownership generation missing")
                        req_id = str(payload[0])
                        generation = str(payload[1])
                        remote_engine_id = str(payload[4])
                        if terminal := ownership_tombstones.get(
                            req_id, generation
                        ):
                            await reply(
                                identity,
                                msgspec.msgpack.encode(
                                    (terminal.value, generation)
                                ),
                            )
                            continue
                        current = self._ownership_generations.get(req_id)
                        if current is not None and current != generation:
                            # This scheduler never accepted the stale generation;
                            # reject it without disturbing the active one.
                            ownership_tombstones.record(
                                req_id,
                                generation,
                                RequestOwnershipState.CANCELED,
                            )
                            await reply(
                                identity,
                                msgspec.msgpack.encode(
                                    (
                                        OWNERSHIP_RESULT_CANCELED,
                                        generation,
                                    )
                                ),
                            )
                            continue
                        self._ownership_generations[req_id] = generation
                        self._ownership_remote_engines[req_id] = remote_engine_id
                        self._ownership_states[req_id] = (
                            RequestOwnershipState.ACCEPTED
                        )
                        # XPUB preserves per-subscriber order. Reply only after
                        # PREPARED is queued to every connected local worker.
                        await publish_to_connector.send(data)
                        await reply(
                            identity,
                            msgspec.msgpack.encode(
                                (
                                    OWNERSHIP_RESULT_ACCEPTED,
                                    generation,
                                )
                            ),
                        )
                        continue

                    if op_type == OWNERSHIP_CANCEL_MSG:
                        req_id, generation, remote_engine_id, _reason = payload
                        req_id = str(req_id)
                        generation = str(generation)
                        remote_engine_id = str(remote_engine_id)
                        if terminal := ownership_tombstones.get(
                            req_id, generation
                        ):
                            await reply(
                                identity,
                                msgspec.msgpack.encode(
                                    (terminal.value, generation)
                                ),
                            )
                            continue
                        current = self._ownership_generations.get(req_id)
                        if current is not None and current != generation:
                            ownership_tombstones.record(
                                req_id,
                                generation,
                                RequestOwnershipState.CANCELED,
                            )
                            await reply(
                                identity,
                                msgspec.msgpack.encode(
                                    (
                                        OWNERSHIP_RESULT_CANCELED,
                                        generation,
                                    )
                                ),
                            )
                            continue
                        self._ownership_generations[req_id] = generation
                        self._ownership_remote_engines[req_id] = remote_engine_id
                        ownership_waiters.setdefault(
                            (req_id, generation), set()
                        ).add(identity)
                        # Do not reply until Multiproc reports all TP/PP workers
                        # terminal. Duplicate CANCEL is safe and helps a worker
                        # that missed an earlier publication.
                        await publish_to_connector.send(data)
                        continue

                    if op_type == "disconnect":
                        if isinstance(payload, (list, tuple)):
                            remote_engine_id = str(payload[0])
                            teardown_id = int(payload[1])
                        else:
                            remote_engine_id = str(payload)
                            teardown_id = 0
                        self.send_to_peer_infos.put_nowait((remote_engine_id, None))
                        if teardown_id in completed:
                            completed_engine, result = completed[teardown_id]
                            await reply(
                                identity,
                                result
                                if completed_engine == remote_engine_id
                                else TEARDOWN_RESULT_FAILED,
                            )
                            continue
                        if teardown_id in pending:
                            if (
                                pending[teardown_id]["engine_id"]
                                != remote_engine_id
                            ):
                                await reply(
                                    identity, TEARDOWN_RESULT_FAILED
                                )
                            else:
                                pending[teardown_id]["waiters"].append(
                                    identity
                                )
                            continue
                        pending[teardown_id] = {
                            "engine_id": remote_engine_id,
                            "waiters": [identity],
                            "acks": {},
                            "deadline": time.monotonic()
                            + self._teardown_timeout,
                        }
                        # Do not reply yet. The Router sees success only after
                        # every local TP/PP worker closed the native endpoint.
                        await publish_to_connector.send(data)
                        logger.info(
                            "NIXL scheduler started teardown for %s; "
                            "id=%s waiting for %d workers",
                            remote_engine_id,
                            teardown_id,
                            expected_workers,
                        )
                        continue

                    await reply(identity, b"")
                    await publish_to_connector.send(data)
                    logger.info(
                        "NIXL scheduler forwarded %d bytes to workers (op=%s)",
                        len(data),
                        op_type,
                    )
                except zmq.Again:
                    if self._stop_event.is_set():
                        break
                    await complete_ownership_events()
                    await expire_teardowns()
                except Exception as e:
                    logger.warning("NIXL scheduler forward failed: %r", e)
                await expire_teardowns()

    async def _run_send_infos(self):
        async def retry_later(
            remote_engine_id: str,
            send_infos: tuple[str, Any],
            attempt: int,
        ) -> None:
            op_type, payload = send_infos
            if op_type not in ("block_ids", OWNERSHIP_CANCEL_MSG):
                return
            req_id, generation = str(payload[0]), str(payload[1])
            if self._ownership_generations.get(req_id) != generation:
                return
            if (
                op_type == "block_ids"
                and req_id in self._ownership_cancel_pending
            ):
                return
            if (
                op_type == OWNERSHIP_CANCEL_MSG
                and req_id not in self._ownership_cancel_pending
            ):
                return
            await asyncio.sleep(min(0.2 * (2 ** min(attempt, 4)), 3.0))
            await self.send_to_peer_infos.put(
                (remote_engine_id, send_infos, attempt + 1)
            )

        def reset_peer_socket(peer_meta: PeerMeta) -> None:
            with suppress(BaseException):
                peer_meta.socket.close(linger=0)
            peer_meta.socket = make_zmq_socket(
                path=make_zmq_path("tcp", peer_meta.addr),
                ctx=self.ctx,
                socket_type=zmq.REQ,
                bind=False,
            )
            peer_meta.socket.setsockopt(zmq.LINGER, 0)
            peer_meta.socket.setsockopt(zmq.SNDTIMEO, 3000)
            peer_meta.socket.setsockopt(zmq.RCVTIMEO, 3000)

        while not self._stop_event.is_set():
            item = await self.send_to_peer_infos.get()
            remote_engine_id, send_infos = item[:2]
            attempt = int(item[2]) if len(item) > 2 else 0
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
            if peer_meta is None:
                asyncio.create_task(
                    retry_later(remote_engine_id, send_infos, attempt)
                )
                continue
            try:
                await peer_meta.socket.send(msgspec.msgpack.encode(send_infos))
                response = await peer_meta.socket.recv()
                op_type, payload = send_infos
                if op_type in ("block_ids", OWNERSHIP_CANCEL_MSG):
                    req_id, generation = str(payload[0]), str(payload[1])
                    result, reply_generation = msgspec.msgpack.decode(response)
                    if str(reply_generation) != generation:
                        raise ValueError(
                            "ownership ACK generation mismatch: "
                            f"{reply_generation!r} != {generation!r}"
                        )
                    if result == OWNERSHIP_RESULT_ACCEPTED:
                        self._ownership_states[req_id] = (
                            RequestOwnershipState.ACCEPTED
                        )
                    elif result in (
                        OWNERSHIP_RESULT_CANCELED,
                        OWNERSHIP_RESULT_TERMINAL,
                    ):
                        self._ownership_cancel_pending.discard(req_id)
                        self._ownership_states[req_id] = (
                            RequestOwnershipState.TERMINAL
                        )
                        self._ownership_safe_releases.put(
                            (
                                req_id,
                                generation,
                                f"producer ownership {result}",
                            )
                        )
                    else:
                        raise ValueError(
                            f"unexpected ownership ACK result {result!r}"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A missing ACK is an uncertain commit: the peer may already
                # have forwarded block_ids to its workers. Keep ownership and
                # retry the idempotent generation-scoped command.
                with suppress(BaseException):
                    reset_peer_socket(peer_meta)
                logger.warning(
                    "NIXL scheduler peer request failed for %s "
                    "(attempt=%d): %r",
                    remote_engine_id,
                    attempt + 1,
                    error,
                )
                asyncio.create_task(
                    retry_later(remote_engine_id, send_infos, attempt)
                )
