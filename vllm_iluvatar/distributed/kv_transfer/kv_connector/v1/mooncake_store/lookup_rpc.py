# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ LookupKey RPC server and client between Scheduler and Worker rank 0."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import zmq
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash

from .layout import BlobBlockHashes
from .protocol import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    MooncakeLookupResult,
    decode_lookup_response,
    encode_lookup_response,
    get_zmq_rpc_path_lookup,
)

if TYPE_CHECKING:
    from .worker import IluMooncakeStoreWorker

logger = init_logger(__name__)

DEFAULT_LOOKUP_TIMEOUT_MS = 5000


class LookupKeyServer:
    """ZMQ REP server on worker rank 0 for prefix-cache lookup & store reset."""

    def __init__(
        self,
        store_worker: IluMooncakeStoreWorker,
        vllm_config: VllmConfig,
    ) -> None:
        self.store_worker = store_worker
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            try:
                os.unlink(self._ipc_path)
            except OSError:
                pass
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )
        self.running = True
        self._closed = False
        self.thread = threading.Thread(
            target=self._serve_loop,
            name="MooncakeLookupServer",
            daemon=True,
        )
        self.thread.start()

    def _serve_loop(self) -> None:
        poller = zmq.Poller()  # type: ignore[attr-defined]
        poller.register(self.socket, zmq.POLLIN)  # type: ignore[attr-defined]
        try:
            while self.running:
                if not poller.poll(timeout=100):
                    continue
                try:
                    frames = self.socket.recv_multipart(copy=False)
                except zmq.ZMQError:  # type: ignore[attr-defined]
                    if self.running:
                        logger.exception("LookupKeyServer failed to receive frame")
                    break

                response = RESP_ERR
                try:
                    msg_type = bytes(frames[0])
                    if msg_type == LOOKUP_MSG:
                        num_tokens = int.from_bytes(frames[1], byteorder="big")
                        hash_len = int.from_bytes(frames[2], byteorder="big")
                        blob = frames[3].buffer
                        block_hashes = BlobBlockHashes(blob, hash_len)
                        result = self.store_worker.lookup(num_tokens, block_hashes)
                        response = encode_lookup_response(result)
                    elif msg_type == RESET_MSG:
                        ok = self.store_worker.reset_store()
                        response = RESP_OK if ok else RESP_ERR
                    else:
                        logger.warning(
                            "LookupKeyServer received unknown msg_type: %r",
                            msg_type,
                        )
                        response = RESP_ERR
                except Exception as exc:
                    logger.error("LookupKeyServer request error: %s", exc)
                    if msg_type == LOOKUP_MSG:
                        response = encode_lookup_response(MooncakeLookupResult(0))
                    else:
                        response = RESP_ERR

                try:
                    self.socket.send(response)
                except zmq.ZMQError:  # type: ignore[attr-defined]
                    if self.running:
                        logger.exception("LookupKeyServer failed to send reply")
                    break
        finally:
            try:
                self.socket.close(linger=0)
            finally:
                self.ctx.term()
                if os.path.exists(self._ipc_path):
                    try:
                        os.unlink(self._ipc_path)
                    except OSError:
                        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)


class LookupKeyClient:
    """ZMQ REQ client in Scheduler for prefix-cache lookup & store reset."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._timeout_ms = int(
            extra.get("lookup_timeout_ms", DEFAULT_LOOKUP_TIMEOUT_MS)
        )
        self._socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket = self._create_socket()
        self._closed = False
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="MooncakeLookupClient",
        )
        self.futures: dict[str, Future[MooncakeLookupResult]] = {}

    def _create_socket(self) -> zmq.Socket:
        sock = make_zmq_socket(
            self.ctx,
            self._socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        if self._timeout_ms > 0:
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)  # type: ignore[attr-defined]
        return sock

    def _reconnect_socket(self) -> None:
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        self.socket = self._create_socket()

    def _lookup_Blocking(
        self,
        num_tokens: int,
        block_hashes: list[BlockHash],
    ) -> MooncakeLookupResult:
        if self._closed:
            return MooncakeLookupResult(0)
        hash_len = len(block_hashes[0]) if block_hashes else 0
        frames = (
            LOOKUP_MSG,
            num_tokens.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
        )
        try:
            self.socket.send_multipart(frames, copy=False)
            resp = self.socket.recv()
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            if not self._closed:
                logger.warning("Mooncake lookup RPC error (%s); reconnecting", exc)
                self._reconnect_socket()
            return MooncakeLookupResult(0)
        return decode_lookup_response(resp)

    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> MooncakeLookupResult | None:
        if self._closed:
            return MooncakeLookupResult(0)
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(
                self._lookup_Blocking,
                num_tokens,
                list(block_hashes),
            )
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as exc:
            logger.error("Mooncake lookup failed for %s: %s", req_id, exc)
            return MooncakeLookupResult(0)
        finally:
            self.futures.pop(req_id, None)

    def discard(self, req_id: str) -> None:
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset_blocking(self) -> bool:
        if self._closed:
            return False
        try:
            self.socket.send(RESET_MSG)
            resp = self.socket.recv()
            return bytes(resp) == RESP_OK
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            if not self._closed:
                logger.warning("Mooncake reset RPC error (%s); reconnecting", exc)
                self._reconnect_socket()
            return False

    def reset(self) -> bool:
        if self._closed:
            return False
        return self.executor.submit(self._reset_blocking).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fut in self.futures.values():
            fut.cancel()
        self.futures.clear()

        def _cleanup_sock() -> None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        try:
            self.executor.submit(_cleanup_sock).result(timeout=1.0)
        except Exception:
            _cleanup_sock()
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            try:
                self.ctx.term()
            except Exception:
                pass
