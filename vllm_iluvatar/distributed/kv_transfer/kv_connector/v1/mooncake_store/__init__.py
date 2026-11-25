# SPDX-License-Identifier: Apache-2.0
"""Iluvatar Mooncake Store V1 KV connector package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import IluMooncakeStoreConfig
    from .connector import (
        IluMooncakeStoreConnector,
        MooncakeStoreConnector,
        MooncakeStoreKVEvents,
    )
    from .scheduler import IluMooncakeStoreScheduler
    from .worker import IluMooncakeStoreWorker, MooncakeStoreWorker

__all__ = [
    "IluMooncakeStoreConfig",
    "IluMooncakeStoreConnector",
    "IluMooncakeStoreScheduler",
    "IluMooncakeStoreWorker",
    "MooncakeStoreConnector",
    "MooncakeStoreKVEvents",
    "MooncakeStoreWorker",
]


def __getattr__(name: str):
    if name == "IluMooncakeStoreConfig":
        from .config import IluMooncakeStoreConfig

        return IluMooncakeStoreConfig
    if name in (
        "IluMooncakeStoreConnector",
        "MooncakeStoreConnector",
        "MooncakeStoreKVEvents",
    ):
        from . import connector as _conn

        return getattr(_conn, name)
    if name == "IluMooncakeStoreScheduler":
        from .scheduler import IluMooncakeStoreScheduler

        return IluMooncakeStoreScheduler
    if name in ("IluMooncakeStoreWorker", "MooncakeStoreWorker"):
        from . import worker as _wrk

        return getattr(_wrk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
