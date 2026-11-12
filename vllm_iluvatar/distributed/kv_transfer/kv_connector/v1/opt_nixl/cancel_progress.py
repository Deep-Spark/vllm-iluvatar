# SPDX-License-Identifier: Apache-2.0
"""Keep producer cancellation proofs progressing after inference becomes idle."""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import logging
import threading

logger = logging.getLogger(__name__)
_MARK = "_ilu_nixl_cancel_wakeup"


class ProducerCancelProgress:
    """Track only cancellations waiting for the existing all-worker proof.

    The actor owns begin/complete. Engine initialization binds a nonblocking
    queue notification, and the engine thread reads pending. Nothing here
    releases KV blocks or supplies a transfer completion.
    """

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}
        self._wake: Callable[[], None] | None = None
        self._notified = False
        self._lock = threading.RLock()

    def bind(self, wake: Callable[[], None]) -> None:
        with self._lock:
            self._wake = wake
            self._notified = False
            self._notify_locked()

    def begin(self, request_id: str, generation: str) -> None:
        with self._lock:
            self._pending[request_id] = generation
            self._notify_locked()

    def complete(self, request_id: str, generation: str) -> None:
        # A late terminal event must not clear a newer generation's poll gate.
        with self._lock:
            if self._pending.get(request_id) == generation:
                del self._pending[request_id]
            if not self._pending:
                self._notified = False

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def _notify_locked(self) -> None:
        if not self._pending or self._notified or self._wake is None:
            return
        try:
            # The bound callback only calls Queue.put_nowait, never GPU work.
            # Keep it under the lock so an empty->new batch cannot lose its wake.
            self._wake()
        except Exception:
            # Preserve the poll gate and retry on a later CANCEL/bind. A failed
            # notification is not evidence that any worker has finished.
            logger.exception("Could not wake the engine to drain NIXL cancellation")
        else:
            self._notified = True


def bind_engine_cancel_wakeup(engine, wake_type) -> None:
    """Attach the existing engine WAKEUP queue to producer NIXL children."""
    connector = getattr(engine.scheduler, "connector", None)
    work = [connector] if connector is not None else []
    seen = set()
    while work:
        child = work.pop()
        if id(child) in seen:
            continue
        seen.add(id(child))
        work.extend(getattr(child, "sub_connectors", ()) or ())
        scheduler = getattr(child, "connector_scheduler", None)
        bind = getattr(scheduler, "bind_engine_wakeup", None)
        if callable(bind) and not scheduler.is_consumer:
            # Capture the queue, not the engine/context, to avoid extending the
            # engine lifetime through the connector's callback.
            input_queue = engine.input_queue
            bind(lambda q=input_queue: q.put_nowait((wake_type, None)))


def install_engine_cancel_wakeup(engine_cls, wake_type) -> None:
    if getattr(engine_cls._process_input_queue, _MARK, False):
        return
    original = engine_cls._process_input_queue

    @wraps(original)
    def process_input(self, *args, **kwargs):
        # General plugins can load inside the already-running constructor.
        # Bind before the first input wait instead of wrapping that constructor.
        if not getattr(self, _MARK, False):
            bind_engine_cancel_wakeup(self, wake_type)
            setattr(self, _MARK, True)
        return original(self, *args, **kwargs)

    setattr(process_input, _MARK, True)
    engine_cls._process_input_queue = process_input
