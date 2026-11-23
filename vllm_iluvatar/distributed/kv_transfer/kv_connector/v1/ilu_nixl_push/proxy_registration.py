# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node registration with the ixLLM PD proxy for the Ilu NIXL push connector.

The upstream 0.31.0 push connector learns the peer's coordinates from the
client (``kv_transfer_params``) and never talks to a proxy. The ixLLM
``proxy_nixl_push`` service, however, is a *registry*: every P/D engine opens a
keepalive TCP connection to ``proxy_port + 1`` and publishes its node info, and
the proxy then routes requests and hands each side the peer's coordinates.

This mixin restores that control plane for the push connector:

* ``PUT_NODE_META_MSG`` keepalive  -> makes the engine visible to the proxy
  (without it the proxy has no prefill/decode node and rejects every request).
* ``GET_NODE_META_MSG`` lookup     -> fills the peer topology
  (``tp_size`` / ``pp_size`` / ``pp_split``) that ``proxy_nixl_push`` does not
  forward to the decoder, but which the push scheduler and the PP-aware
  handshake require.

Both are pure control-plane operations; the KV path stays native NIXL side
channel, so an unreachable proxy only degrades routing, never a transfer.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

PUT_NODE_META_MSG = b"put_node_meta_msg"
GET_NODE_META_MSG = b"get_node_meta_msg"
INFO_DONE = b"info_done\0"

# The proxy drops a node as soon as its keepalive connection closes, so a
# proxy that is not up yet must be retried, never given up on.
KEEPALIVE_RETRY_INTERVAL_S = 5.0
# Blocking ZMQ round trip from the scheduler thread: keep it short.
NODE_META_QUERY_TIMEOUT_MS = 1500


def proxy_addr() -> tuple[str, int] | None:
    """``(host, port)`` of the proxy control plane, or None when unused."""
    raw = (os.environ.get("VLLM_ILU_NIXL_PROXY_ADDR") or "").strip()
    if not raw:
        raw = (os.environ.get("IXLLM_ADDR_PD") or "").strip()
    raw = raw.removeprefix("tcp://")
    if not raw:
        return None
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        logger.warning("Ignoring malformed proxy address %r", raw)
        return None
    return host, int(port)


def _server_addr() -> str:
    """This engine's own OpenAI HTTP endpoint, as the proxy will call it."""
    return (os.environ.get("VLLM_ILU_NIXL_SERVER_ADDR") or "").strip()


class ProxyRegistrationMixin:
    """Publish this engine to the ixLLM PD proxy and look peers up there."""

    _proxy_thread: threading.Thread | None = None
    _proxy_stop: threading.Event | None = None

    # --- lifecycle ----------------------------------------------------- #

    def _init_proxy_registration(self) -> None:
        self._proxy_thread = None
        self._proxy_stop = threading.Event()
        self._proxy_loop: asyncio.AbstractEventLoop | None = None
        self._proxy_writer: asyncio.StreamWriter | None = None
        self._proxy_server_addr: str = ""
        self._proxy_meta_cache: dict[str, dict[str, Any]] = {}
        self._disconnected_peer_engines: queue.SimpleQueue[str] = (
            queue.SimpleQueue()
        )

    def _stop_proxy_registration(self) -> None:
        stop = getattr(self, "_proxy_stop", None)
        if stop is not None:
            stop.set()
        loop = getattr(self, "_proxy_loop", None)
        writer = getattr(self, "_proxy_writer", None)
        if loop is not None and writer is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(writer.close)
        thread = getattr(self, "_proxy_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=KEEPALIVE_RETRY_INTERVAL_S + 2.0)
        self._proxy_thread = None

    def start_proxy_registration(self) -> None:
        """Start the keepalive thread once this engine's topology is final."""
        addr = proxy_addr()
        if addr is None:
            logger.info(
                "VLLM_ILU_NIXL_PROXY_ADDR unset: push connector runs without a "
                "PD proxy registration"
            )
            return
        srv_addr = _server_addr()
        if not srv_addr:
            logger.warning(
                "VLLM_ILU_NIXL_SERVER_ADDR unset: the proxy cannot reach this "
                "engine over HTTP, so requests will not be routed here"
            )
        self._proxy_server_addr = srv_addr
        thread = getattr(self, "_proxy_thread", None)
        if thread is not None and thread.is_alive():
            return
        stop = getattr(self, "_proxy_stop", None)
        if stop is None or stop.is_set():
            self._proxy_stop = threading.Event()
        self._proxy_thread = threading.Thread(
            target=self._proxy_keepalive_main,
            args=(addr, self._proxy_stop),
            daemon=True,
            name="ilu-push-proxy-keepalive",
        )
        self._proxy_thread.start()

    # --- node info ----------------------------------------------------- #

    def _proxy_pp_split(self) -> str:
        """Layer partition of this engine, in the form the peer handshake wants."""
        split = getattr(self, "_pp_split", "") or os.environ.get(
            "VLLM_PP_LAYER_PARTITION", ""
        )
        if split:
            return str(split)
        pp_size = int(
            getattr(self.vllm_config.parallel_config, "pipeline_parallel_size", 1)
        )
        if pp_size <= 1:
            return ""
        try:
            from vllm.distributed.utils import get_pp_indices

            model_config = self.vllm_config.model_config
            hf_text_config = getattr(model_config, "hf_text_config", None)
            total = int(
                getattr(hf_text_config, "num_hidden_layers", 0)
                or model_config.hf_config.num_hidden_layers
            )
            return ",".join(
                str(end - start)
                for start, end in (
                    get_pp_indices(total, rank, pp_size) for rank in range(pp_size)
                )
            )
        except Exception:
            logger.debug("Could not infer the PP layer split", exc_info=True)
            return ""

    def _proxy_node_info(self) -> dict[str, Any]:
        parallel = self.vllm_config.parallel_config
        transfer_cfg = self.vllm_config.kv_transfer_config
        block_size = int(
            getattr(self, "registered_block_size", getattr(self, "block_size", 0)) or 0
        )
        return {
            "server_addr": getattr(self, "_proxy_server_addr", "") or _server_addr(),
            "side_channel_addr": (
                f"{self.side_channel_host}:{self.side_channel_port}"
            ),
            "pp_size": int(getattr(parallel, "pipeline_parallel_size", 1)),
            "tp_size": int(getattr(parallel, "tensor_parallel_size", 1)),
            "engine_id": self.engine_id,
            "pp_split": self._proxy_pp_split(),
            "dp_rank": int(getattr(parallel, "data_parallel_rank", 0) or 0),
            "kv_role": getattr(transfer_cfg, "kv_role", None),
            "physical_block_size": block_size,
            # Required by vllm-router ``--opt-nixl-proxy`` registration check.
            "teardown_protocol_version": 1,
        }

    # --- keepalive ----------------------------------------------------- #

    def _proxy_keepalive_main(
        self, addr: tuple[str, int], stop: threading.Event
    ) -> None:
        try:
            asyncio.run(self._proxy_keepalive_loop(addr, stop))
        except Exception:
            logger.exception("PD proxy keepalive thread exited with an error")

    async def _proxy_keepalive_loop(
        self, addr: tuple[str, int], stop: threading.Event
    ) -> None:
        self._proxy_loop = asyncio.get_running_loop()
        host, port = addr
        keepalive_port = port + 1  # the proxy serves node info on port + 1
        failures = 0
        while not stop.is_set():
            connected_at = 0.0
            try:
                reader, writer = await asyncio.open_connection(host, keepalive_port)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                log = logger.warning if failures == 1 else logger.debug
                log(
                    "Cannot connect to PD proxy at %s:%d (%s); retrying every %.0fs",
                    host,
                    keepalive_port,
                    exc,
                    KEEPALIVE_RETRY_INTERVAL_S,
                )
                await self._proxy_sleep(stop, KEEPALIVE_RETRY_INTERVAL_S)
                continue
            try:
                self._proxy_writer = writer
                node_info = self._proxy_node_info()
                writer.write(
                    msgspec.msgpack.encode((PUT_NODE_META_MSG, node_info, {}))
                )
                writer.write(INFO_DONE)
                await writer.drain()
                connected_at = time.monotonic()
                if failures:
                    logger.info(
                        "Registered with PD proxy %s:%d after %d retry attempt(s): "
                        "engine=%s role=%s tp=%d pp=%d",
                        host,
                        keepalive_port,
                        failures,
                        node_info["engine_id"],
                        node_info["kv_role"],
                        node_info["tp_size"],
                        node_info["pp_size"],
                    )
                    failures = 0
                # The proxy holds the node while this read blocks; it returns
                # only when the proxy closes the connection.
                await reader.readexactly(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop.is_set():
                    break
                if time.monotonic() - connected_at >= 1.0:
                    logger.warning(
                        "PD proxy %s:%d closed the keepalive connection (%s); "
                        "reconnecting",
                        host,
                        keepalive_port,
                        exc,
                    )
                else:
                    logger.debug(
                        "PD proxy %s:%d closed the keepalive connection (%s)",
                        host,
                        keepalive_port,
                        exc,
                    )
            finally:
                self._proxy_writer = None
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            await self._proxy_sleep(stop, KEEPALIVE_RETRY_INTERVAL_S)

    @staticmethod
    async def _proxy_sleep(stop: threading.Event, seconds: float) -> None:
        """Sleep that wakes up immediately on shutdown."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while not stop.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))

    # --- peer lookup --------------------------------------------------- #

    def _query_proxy_node_meta(self, engine_id: str) -> dict[str, Any] | None:
        """``GET_NODE_META_MSG`` for a peer engine, cached for the process."""
        cached = getattr(self, "_proxy_meta_cache", {}).get(engine_id)
        if cached is not None:
            return cached
        addr = proxy_addr()
        if addr is None or not engine_id:
            return None
        host, port = addr
        try:
            with zmq_ctx(zmq.REQ, f"tcp://{host}:{port}") as sock:
                sock.setsockopt(zmq.SNDTIMEO, NODE_META_QUERY_TIMEOUT_MS)
                sock.setsockopt(zmq.RCVTIMEO, NODE_META_QUERY_TIMEOUT_MS)
                sock.send(msgspec.msgpack.encode((GET_NODE_META_MSG, engine_id)))
                frames = sock.recv_multipart()
                status = frames[0] if frames else b""
                if status != b"success":
                    logger.warning(
                        "PD proxy has no node info for engine %s (%s); the peer "
                        "topology must come from kv_transfer_params",
                        engine_id,
                        status,
                    )
                    return None
                raw_meta = frames[1] if len(frames) > 1 else sock.recv()
                meta = msgspec.msgpack.decode(raw_meta)
        except Exception as exc:
            logger.warning(
                "Failed to query PD proxy %s:%d for engine %s: %s",
                host,
                port,
                engine_id,
                exc,
            )
            return None
        if not isinstance(meta, dict):
            return None
        self._proxy_meta_cache[engine_id] = meta
        return meta

    def _fill_remote_topology_from_proxy(self, params: dict[str, Any]) -> None:
        """Complete a decoder request with P's topology that the proxy omits.

        ``proxy_nixl_push`` forwards ``remote_host``/``remote_port`` but not the
        producer's ``tp_size``/``pp_size``/``pp_split``. The push scheduler
        indexes ``tp_size`` directly, and the PP-aware handshake needs
        ``pp_split``, so fill them from the proxy registry before the request
        reaches the scheduler hooks.
        """
        if params.get("tp_size") and params.get("pp_split"):
            return
        remote_engine_id = params.get("remote_engine_id")
        if not remote_engine_id:
            return
        meta = self._query_proxy_node_meta(remote_engine_id)
        if meta is None:
            return

        if not params.get("tp_size"):
            params["tp_size"] = int(meta.get("tp_size") or 1)
        if not params.get("pp_size"):
            params["pp_size"] = int(meta.get("pp_size") or 1)
        if not params.get("pp_split"):
            params["pp_split"] = str(meta.get("pp_split") or "")

        side_channel = str(meta.get("side_channel_addr") or "")
        if side_channel:
            host, _, port = side_channel.rpartition(":")
            if not params.get("remote_host") and host:
                params["remote_host"] = host
            if not params.get("remote_port") and port.isdigit():
                params["remote_port"] = int(port)

    # --- hooks used by the scheduler ----------------------------------- #

    def on_proxy_new_request(self, request: Any) -> None:
        """Decoder-side: ensure the remote producer topology is complete."""
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return
        if not params.get("do_remote_prefill"):
            return
        self._fill_remote_topology_from_proxy(params)
        # ``vllm-router --opt-nixl-proxy`` serializes ``remote_port`` as a JSON
        # string, while upstream ``HeartbeatInfo`` / ``RemoteMeta`` validate
        # ``port: int`` under msgspec. Normalize integer fields in place.
        for int_key in ("remote_port", "tp_size", "pp_size", "dcp_size"):
            val = params.get(int_key)
            if isinstance(val, str) and val.isdigit():
                params[int_key] = int(val)
        # Upstream tracks producer-side leases by ``remote_request_id`` and
        # disables heartbeats when it is absent. In push mode both legs share
        # one request id, so it can always be derived.
        if not params.get("remote_request_id"):
            params["remote_request_id"] = request.request_id
