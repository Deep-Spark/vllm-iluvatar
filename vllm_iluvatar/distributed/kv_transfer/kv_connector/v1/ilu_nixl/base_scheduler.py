# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side PP extensions for the Ilu NIXL connector."""

import threading
import time
from typing import Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

from .metadata import HeartbeatInfo, IluNixlConnectorMetadata

logger = init_logger("vllm." + __name__)


class IluNixlBaseConnectorScheduler(NixlBaseConnectorScheduler):
    """Base scheduler that propagates PP topology through connector metadata."""

    def set_xfer_handshake_metadata(
        self,
        metadata: dict[int | tuple[int, int], KVConnectorHandshakeMetadata],
    ) -> None:
        """Serve handshake metadata by global rank.

        IluNixl currently keeps its vLLM 0.23-era global-rank handshake
        protocol while reusing vLLM 0.24's NIXL internals. The connector
        facade converts (pp_rank, tp_rank) to global rank before calling here.
        """
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        for rank_key, rank_metadata in metadata.items():
            if isinstance(rank_key, tuple):
                pp_rank, tp_rank = rank_key
                global_rank = pp_rank * tp_size + tp_rank
            else:
                global_rank = rank_key
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "IluNixlConnectorScheduler expects NixlHandshakePayload "
                    "for handshake metadata."
                )
            encoded_data[global_rank] = encoder.encode(rank_metadata)
            logger.debug(
                "Global rank %d: encoded NixlHandshakePayload size: %s bytes",
                global_rank,
                str(len(encoded_data[global_rank])),
            )

        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                    tp_size,
                ),
                daemon=True,
                name="ilu_nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[int, Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        tp_size: int,
    ):
        """Background thread serving IluNixl global-rank handshakes."""
        path = make_zmq_path("tcp", host, port)
        logger.info(
            "Starting IluNixl global-rank handshake listener on path: %s, "
            "available ranks=%s",
            path,
            sorted(encoded_data.keys()),
        )
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue

                try:
                    decoded_msg = msgspec.msgpack.decode(msg)
                    if len(decoded_msg) == 2:
                        msg, target_rank = decoded_msg
                    elif len(decoded_msg) == 3:
                        msg, target_pp_rank, target_tp_rank = decoded_msg
                        target_rank = target_pp_rank * tp_size + target_tp_rank
                        logger.warning(
                            "IluNixl listener received PP-aware handshake "
                            "(pp=%s, tp=%s); converted to global rank %s",
                            target_pp_rank,
                            target_tp_rank,
                            target_rank,
                        )
                    else:
                        logger.error(
                            "IluNixl listener: invalid handshake message %r",
                            decoded_msg,
                        )
                        continue
                except Exception:
                    logger.exception("IluNixl listener failed to decode message")
                    continue
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                    continue
                if not isinstance(target_rank, int):
                    logger.error(
                        "IluNixl listener: invalid target_rank %r, available=%s",
                        target_rank,
                        sorted(encoded_data.keys()),
                    )
                    continue
                if target_rank not in encoded_data:
                    logger.error(
                        "IluNixl listener: no metadata for global rank %s, "
                        "available=%s",
                        target_rank,
                        sorted(encoded_data.keys()),
                    )
                    continue

                sock.send_multipart((identity, b"", encoded_data[target_rank]))

    def on_new_request(self, request) -> None:
        # Keep this method in sync with vLLM
        # vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        # at community commit 63e161f29. Ilu delta: carry pp_size/pp_split in
        # heartbeat metadata for heterogeneous PP handshakes.
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return

        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        pp_size = params.get("pp_size", 1)
        pp_split = params.get("pp_split", "")
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return

        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                pp_size=pp_size,
                pp_split=pp_split,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def build_connector_meta(self, scheduler_output):
        # Keep this method in sync with vLLM
        # vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        # at community commit 63e161f29. Ilu delta: use IluNixlConnectorMetadata
        # so ReqMeta carries remote PP topology to the worker.
        meta = IluNixlConnectorMetadata()

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta
