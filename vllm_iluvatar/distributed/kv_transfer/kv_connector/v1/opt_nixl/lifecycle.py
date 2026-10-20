"""Remote-engine lifecycle state shared by the worker's uvloop actor.

The objects in this module contain no NIXL or CUDA calls.  They only describe
ownership and completion data so the worker can keep all transport mutations on
one thread while the engine main thread consumes immutable result batches.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import torch

ReqId = str
WorkerRole = Literal["producer", "consumer"]


class NixlDrainIncomplete(RuntimeError):
    """Native ownership remains live; retain buffers and allow shutdown retry."""


class RemoteSessionState(Enum):
    CONNECTING = auto()
    ACTIVE = auto()
    DRAINING = auto()
    STUCK = auto()
    CLOSED = auto()


class RequestOwnershipState(str, Enum):
    """Cross-process ownership of one D-side destination generation."""

    PREPARED = "prepared"
    ACCEPTED = "accepted"
    WRITING = "writing"
    TERMINAL = "terminal"
    CANCELED = "canceled"

    @property
    def is_safe_to_release(self) -> bool:
        return self in (
            RequestOwnershipState.TERMINAL,
            RequestOwnershipState.CANCELED,
        )


@dataclass(frozen=True)
class RequestOwnershipTombstone:
    state: RequestOwnershipState
    created_at: float


class RequestTombstoneStore:
    """Bounded replay fence for terminal request generations.

    Expiry is intentionally much longer than the control-message retry window.
    If the bound is reached before any tombstone is old enough to prune, fail
    closed instead of forgetting a generation that could still be replayed.
    """

    def __init__(self, ttl: float, limit: int):
        if ttl <= 0 or limit <= 0:
            raise ValueError("request tombstone ttl and limit must be positive")
        self.ttl = ttl
        self.limit = limit
        self._items: OrderedDict[
            tuple[ReqId, str], RequestOwnershipTombstone
        ] = OrderedDict()

    def _prune(self, now: float) -> None:
        while self._items:
            key, tombstone = next(iter(self._items.items()))
            if now - tombstone.created_at < self.ttl:
                break
            del self._items[key]

    def get(
        self, req_id: ReqId, generation: str, now: float | None = None
    ) -> RequestOwnershipState | None:
        now = time.monotonic() if now is None else now
        self._prune(now)
        tombstone = self._items.get((req_id, generation))
        return None if tombstone is None else tombstone.state

    def record(
        self,
        req_id: ReqId,
        generation: str,
        state: RequestOwnershipState,
        now: float | None = None,
    ) -> RequestOwnershipState:
        if not state.is_safe_to_release:
            raise ValueError(f"non-terminal tombstone state: {state.value}")
        now = time.monotonic() if now is None else now
        self._prune(now)
        key = (req_id, generation)
        if existing := self._items.get(key):
            return existing.state
        if len(self._items) >= self.limit:
            raise RuntimeError(
                "request ownership tombstone capacity exhausted; refusing "
                "to forget a possibly replayable generation"
            )
        self._items[key] = RequestOwnershipTombstone(state, now)
        return state

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class RemoteEngineSession:
    """Uvloop-owned lifecycle of one remote engine generation."""

    engine_id: str
    generation: int
    role: WorkerRole
    state: RemoteSessionState = RemoteSessionState.CONNECTING
    teardown_id: int | None = None
    disconnected_at: float | None = None
    drain_deadline: float | None = None
    pending_send: set[ReqId] = field(default_factory=set)
    inflight_send: set[ReqId] = field(default_factory=set)
    pending_recv: set[ReqId] = field(default_factory=set)
    stuck_reported: bool = False
    reported_teardown_id: int | None = None

    def activate(self) -> None:
        if self.state not in (
            RemoteSessionState.CONNECTING,
            RemoteSessionState.ACTIVE,
        ):
            raise RuntimeError(
                f"cannot activate {self.engine_id} from state {self.state.name}"
            )
        self.state = RemoteSessionState.ACTIVE

    def start_draining(self, now: float, timeout: float) -> None:
        if self.state in (RemoteSessionState.CLOSED, RemoteSessionState.STUCK):
            return
        self.state = RemoteSessionState.DRAINING
        self.disconnected_at = now
        self.drain_deadline = now + timeout

    @property
    def accepts_work(self) -> bool:
        return self.state in (
            RemoteSessionState.CONNECTING,
            RemoteSessionState.ACTIVE,
        )

    @property
    def request_count(self) -> int:
        return len(self.pending_send) + len(self.inflight_send) + len(self.pending_recv)

    @property
    def can_release(self) -> bool:
        return (
            self.state
            in (
                RemoteSessionState.DRAINING,
                RemoteSessionState.STUCK,
            )
            and self.request_count == 0
        )


@dataclass
class CompletionBatch:
    """One uvloop-to-main-thread completion message."""

    done_sending: set[ReqId] = field(default_factory=set)
    done_recving: set[ReqId] = field(default_factory=set)
    # Main thread performs CUDA layout/quantization post-processing with ReqMeta.
    recv_metadata: dict[ReqId, Any] = field(default_factory=dict)
    invalid_block_ids: set[int] = field(default_factory=set)
    failures: dict[ReqId, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.done_sending
            or self.done_recving
            or self.recv_metadata
            or self.invalid_block_ids
            or self.failures
        )

    def merge(self, other: CompletionBatch) -> None:
        self.done_sending.update(other.done_sending)
        self.done_recving.update(other.done_recving)
        self.recv_metadata.update(other.recv_metadata)
        self.invalid_block_ids.update(other.invalid_block_ids)
        self.failures.update(other.failures)


@dataclass
class SendRequestState:
    """Uvloop-owned join state for one producer request.

    Local layer readiness, remote destination blocks, and session activation
    arrive independently.  None of those events is consumed until all required
    inputs are present.
    """

    req_id: ReqId
    metadata: Any | None = None
    session: RemoteEngineSession | None = None
    deadline: float | None = None
    ready_layers: list[tuple[int, Any]] = field(default_factory=list)
    transfer_intent: str | None = None
    remote_block_ids: Any | None = None
    remote_block_starts: list[int] | None = None
    started: bool = False
    generation: str = ""
    ownership_state: RequestOwnershipState = RequestOwnershipState.PREPARED
    cancel_requested: bool = False
    command_fingerprint: bytes | None = None

    def install_metadata(self, metadata: Any, deadline: float | None) -> None:
        """Merge scheduler metadata without dropping earlier join inputs."""

        self.metadata = metadata
        if deadline is not None:
            self.deadline = deadline

    def is_ready(self) -> bool:
        """Return whether all independent producer prerequisites have joined."""

        return (
            self.metadata is not None
            and self.session is not None
            and self.session.state == RemoteSessionState.ACTIVE
            and self.transfer_intent is not None
            and self.has_remote_blocks
            and self.has_local_kv
            and self.ownership_state
            in (
                RequestOwnershipState.ACCEPTED,
                RequestOwnershipState.WRITING,
            )
            and not self.cancel_requested
        )

    @property
    def has_local_kv(self) -> bool:
        return bool(self.ready_layers)

    @property
    def has_remote_blocks(self) -> bool:
        return self.remote_block_ids is not None


@dataclass(frozen=True)
class LayerReadyBatch:
    """Main-thread layer-ready event consumed by the uvloop actor."""

    requests: dict[ReqId, Any]
    layer_index: int | str
    cuda_event: Any
    slot_mapping: torch.Tensor | None = None
