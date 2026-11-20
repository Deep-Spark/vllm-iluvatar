# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ context manager tolerant of DEALER sockets.

``vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils.zmq_ctx`` rejects
every socket type except ROUTER/REQ, but the single-RTT handshake fast path
pipelines ``GET_META_BATCH_MSG`` over a DEALER. Without this shim the handshake
raises ``ValueError: Unexpected socket type: 5`` on vLLM versions whose helper
has not been widened yet, which then surfaces as a bogus block-level transfer
failure and kills the engine.

Shaped like the vLLM helper, one extra socket type.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import zmq
from vllm.utils.network_utils import make_zmq_socket

_ALLOWED_SOCKET_TYPES = (zmq.ROUTER, zmq.REQ, zmq.DEALER)


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket, allowing DEALER alongside ROUTER/REQ."""
    if socket_type not in _ALLOWED_SOCKET_TYPES:
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
