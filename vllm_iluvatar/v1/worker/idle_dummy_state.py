# SPDX-License-Identifier: Apache-2.0
"""Thread-local state for runtime DP idle dummy-batch short-circuiting."""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

_TLS = threading.local()


class AllRanksIdleDummyShortCircuit(Exception):
    """Internal control-flow signal raised when all DP ranks run idle dummy."""


@contextmanager
def idle_dummy_batch_scope() -> Generator[None, None, None]:
    """Mark the current thread as executing a runtime DP ``execute_dummy_batch``."""
    prev_in_idle = getattr(_TLS, "in_idle_dummy", False)
    prev_short = getattr(_TLS, "short_circuited", False)
    _TLS.in_idle_dummy = True
    _TLS.short_circuited = False
    try:
        yield
    finally:
        _TLS.in_idle_dummy = prev_in_idle
        _TLS.short_circuited = prev_short


def is_idle_dummy_batch() -> bool:
    """Return True iff the current call originates from ``execute_dummy_batch``."""
    return bool(getattr(_TLS, "in_idle_dummy", False))


def mark_idle_dummy_short_circuited() -> None:
    """Record that the DP sync short-circuited an all-idle dummy step."""
    if getattr(_TLS, "in_idle_dummy", False):
        _TLS.short_circuited = True


def was_idle_dummy_short_circuited() -> bool:
    """Return True if the active ``idle_dummy_batch_scope`` short-circuited."""
    return bool(getattr(_TLS, "short_circuited", False))
