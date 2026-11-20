# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote engine session lifecycle with DRAINING protection for in-flight READs."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class RemoteSessionState(enum.Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    STUCK = "stuck"
    CLOSED = "closed"


@dataclass
class RemoteEngineSession:
    """Tracks NIXL handles and active RDMA READs for one remote engine generation."""

    engine_id: str
    generation: int
    host: str = ""
    port: int = 0
    state: RemoteSessionState = RemoteSessionState.ACTIVE
    remote_agents: dict[int, str] = field(default_factory=dict)
    dlist_handles: dict[int, int] = field(default_factory=dict)
    inflight_handles: set[int] = field(default_factory=set)
    drain_deadline: float | None = None
    stuck_logged: bool = False

    def add_inflight(self, handle: int) -> None:
        self.inflight_handles.add(handle)

    def remove_inflight(self, handle: int) -> None:
        self.inflight_handles.discard(handle)

    @property
    def has_inflight(self) -> bool:
        return bool(self.inflight_handles)

    def start_draining(self, now: float, grace_timeout: float) -> None:
        if self.state in (RemoteSessionState.CLOSED, RemoteSessionState.STUCK):
            return
        self.state = RemoteSessionState.DRAINING
        if self.drain_deadline is None:
            self.drain_deadline = now + max(0.0, grace_timeout)

    def check_drain_progress(self, now: float) -> tuple[bool, bool]:
        """Return ``(can_release_now, entered_stuck_state)``."""
        if self.state == RemoteSessionState.CLOSED:
            return False, False
        if not self.inflight_handles:
            return True, False
        if (
            self.state == RemoteSessionState.DRAINING
            and self.drain_deadline is not None
            and now >= self.drain_deadline
        ):
            self.state = RemoteSessionState.STUCK
            newly_stuck = not self.stuck_logged
            self.stuck_logged = True
            return False, newly_stuck
        return False, False
