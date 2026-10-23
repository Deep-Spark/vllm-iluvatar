# SPDX-License-Identifier: Apache-2.0
"""Iluvatar wrappers for ``vllm.v1.engine.core``.

ILU wrappers for DP prefill synchronization and replacement-request handling.

Target   : vllm.v1.engine.core.DPEngineCoreProc.{
           _should_throttle_prefills,_process_engine_step,_handle_client_request}
Baseline : main_91eacae36dd
Why      : Closed-loop traffic can deliver replacement prefills to DP ranks at
           different decode steps. Letting the first ready rank immediately
           prefill splits a global EP step into real-prefill and dummy/decode
           work, adding a second prefill-sized step to TTFT.
           - VLLM_ILU_DP_ENABLE_SYNC_PREFILLS enables rank-synchronized
           prefills.
           - VLLM_ILU_DP_SYNC_READY_REPLAY includes completed remote-KV SWA
             replays in the pre-scheduling readiness vote.
           - VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS bounds how many model steps a
           partial prefill may wait before fail-open.
           - VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS bounds the post-completion wait
           for a replacement request before the rank enters a dummy batch.
How      : Before scheduling, all DP EngineCore ranks synchronize two CPU
           counters (prefill-ready and decode-busy) over a cloned c10d group,
           isolated from vLLM's DP lifecycle collectives. New and chunked
           prefills are deferred for a bounded number of steps, then fail open
           to prevent starvation. After a rank becomes empty, a bounded queue
           wait covers the short output/input replacement round trip.
           This module does not copy either community method. It wraps the live
           methods with two deltas: a pre-scheduling prefill-sync decision and a
           post-execution replacement wait. All other behavior delegates to the
           captured community implementation.
Scope    : Hybrid low-latency decode with plugin global round-robin enabled.
Remove   : When upstream offers rank-synchronized prefill scheduling and a
           bounded post-completion input-queue wait.

This module also contains the PP + speculative-decode wrappers corresponding
to the ``main_6e448d0`` baseline.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    import torch
    from torch.distributed import Store


@dataclass(slots=True)
class DPPrefillSyncState:
    enabled: bool
    max_replacement_wait_s: float
    max_defer_steps: int
    sync_group: dist.ProcessGroup | None
    sync_ready_replay: bool = False
    rearm_after_admission: bool = False
    coalesce_steps: int = 2
    partial_steps: int = 0
    force_throttle: bool = False
    sync_tensor: torch.Tensor | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            import torch

            counters = 3 if self.rearm_after_admission else 2
            self.sync_tensor = torch.empty(counters, dtype=torch.int32, device="cpu")
        else:
            self.sync_tensor = None

    def update(
        self,
        ready_ranks: int,
        decode_ranks: int,
        dp_size: int,
        *,
        continuing_ranks: int = 0,
        admission_allowed: bool = True,
    ) -> bool:
        # An already-running chunked prefill must keep advancing. This vote is
        # global: otherwise ranks could enter different model/EP schedules.
        if self.rearm_after_admission and continuing_ranks:
            self.partial_steps = 0
            self.force_throttle = False
            return False
        partial_prefill = 0 < ready_ranks < dp_size
        if not partial_prefill:
            self.partial_steps = 0
            self.force_throttle = False
            return False

        self.partial_steps += 1
        defer_limit = self.max_defer_steps
        if decode_ranks == 0:
            defer_limit = min(defer_limit, self.coalesce_steps)
        self.force_throttle = self.partial_steps <= defer_limit
        if self.rearm_after_admission and not self.force_throttle and admission_allowed:
            # Start a new bounded coalescing window after this admission slot.
            # A community cadence veto must not consume the slot: otherwise
            # the two periods can unnecessarily multiply admission latency.
            self.partial_steps = 0
        return self.force_throttle


def _clone_dp_process_group(
    group: dist.ProcessGroup,
    store: Store,
) -> dist.ProcessGroup:
    import torch
    from vllm.distributed.utils import init_gloo_process_group

    timeout = group._get_backend(torch.device("cpu")).options._timeout
    return init_gloo_process_group(
        c10d.PrefixStore("ilu_dp_prefill_sync", store),
        group.rank(),
        group.size(),
        timeout,
    )


def _local_prefill_state(
    scheduler,
    sync_ready_replay: bool = False,
) -> tuple[bool, bool]:
    # A completed async P/D receive remains in ``skipped_waiting`` until
    # Scheduler.schedule() promotes it. Count it before scheduling so the DP
    # gate can coalesce 128-token SWA replays across ranks. After promotion, a
    # throttled replay stays in that queue as WAITING with replay_start set.
    replay_ready = False
    if sync_ready_replay:
        finished_recvs = getattr(scheduler, "finished_recving_kv_req_ids", ())
        replay_ready = any(
            (
                getattr(request, "status", None)
                == RequestStatus.WAITING_FOR_REMOTE_KVS
                and getattr(request, "request_id", None) in finished_recvs
            )
            or (
                getattr(request, "status", None)
                in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
                and getattr(request, "replay_start", 0) > 0
            )
            for request in scheduler.skipped_waiting
        )
    prefill_ready = bool(scheduler.waiting) or replay_ready or any(
        request.is_prefill_chunk for request in scheduler.running
    )
    running_decode = any(
        not request.is_prefill_chunk for request in scheduler.running
    )
    return prefill_ready, running_decode


def _make_should_throttle_prefills(orig):
    # [ILU override] Wrapper, not a full-method copy. Delta: OR the
    # rank-synchronized force-defer decision with the community policy.
    def _should_throttle_prefills(self) -> bool:
        return self._ilu_dp_prefill_sync_state.force_throttle or orig(self)

    return _should_throttle_prefills


def _make_handle_client_request(orig):
    # [ILU override] A synchronized prefill wave must activate every DP rank.
    # Ignore the community target-engine exclusion because ADD and START_DP_WAVE
    # arrive over independent queues and can otherwise leave the target idle.
    def _handle_client_request(self, request_type, request) -> None:
        from vllm.v1.engine import EngineCoreRequestType

        if request_type == EngineCoreRequestType.START_DP_WAVE:
            prefill_sync_state = getattr(
                self,
                "_ilu_dp_prefill_sync_state",
                None,
            )
            if prefill_sync_state is None:
                import vllm.envs as envs

                sync_prefills = envs.VLLM_ILU_DP_ENABLE_SYNC_PREFILLS
            else:
                sync_prefills = prefill_sync_state.enabled

            if sync_prefills:
                new_wave, _exclude_engine = request
                request = (new_wave, None)

        return orig(self, request_type, request)

    return _handle_client_request


def _make_process_engine_step(orig, *, community_should_throttle=None):
    # [ILU override] Wrapper, not a full-method copy. Deltas:
    # 1) synchronize prefills before delegating to community scheduling;
    # 2) bounded-wait for a replacement after community execution empties
    #    the local scheduler.
    def _process_engine_step(self) -> bool:
        if not hasattr(self, "_ilu_dp_prefill_sync_state"):
            import vllm.envs as envs

            self._ilu_dp_prefill_sync_state = DPPrefillSyncState(
                enabled=envs.VLLM_ILU_DP_ENABLE_SYNC_PREFILLS,
                max_replacement_wait_s=(
                    envs.VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS / 1000
                ),
                max_defer_steps=envs.VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS,
                sync_group=_clone_dp_process_group(self.dp_group, self.dp_store),
                sync_ready_replay=envs.VLLM_ILU_DP_SYNC_READY_REPLAY,
                rearm_after_admission=envs.VLLM_ILU_DP_PREFILL_REARM,
            )
        prefill_sync_state = self._ilu_dp_prefill_sync_state
        waiting = False
        running_decode = False
        sync_active = prefill_sync_state.enabled and (
            self.engines_running or self.scheduler.has_unfinished_requests()
        )
        if sync_active:
            waiting, running_decode = _local_prefill_state(
                self.scheduler,
                prefill_sync_state.sync_ready_replay,
            )
            state = prefill_sync_state.sync_tensor
            assert state is not None
            state[0] = int(waiting)
            state[1] = int(running_decode)
            if prefill_sync_state.rearm_after_admission:
                state[2] = int(
                    any(request.is_prefill_chunk for request in self.scheduler.running)
                )
            dist.all_reduce(
                state,
                op=dist.ReduceOp.SUM,
                group=prefill_sync_state.sync_group,
            )
            values = state.tolist()
            ready_ranks, decode_ranks = values[:2]
            continuing_ranks = 0
            admission_allowed = True
            if prefill_sync_state.rearm_after_admission:
                if community_should_throttle is None:
                    raise RuntimeError(
                        "Rearmed DP admission requires the community policy"
                    )
                continuing_ranks = int(values[2])
                admission_allowed = not community_should_throttle(self)
            prefill_sync_state.update(
                ready_ranks,
                decode_ranks,
                self.dp_size,
                continuing_ranks=continuing_ranks,
                admission_allowed=admission_allowed,
            )
            # Every rank that participates in prefill synchronization must
            # execute exactly one real or dummy worker step. Otherwise worker
            # and EngineCore collectives can advance in different orders.
            self.engines_running = True
            if (
                prefill_sync_state.force_throttle
                and waiting
                and not running_decode
            ):
                return False

        executed = orig(self)
        if (
            executed
            and prefill_sync_state.max_replacement_wait_s > 0
            and not self.scheduler.has_unfinished_requests()
        ):
            deadline = time.monotonic() + prefill_sync_state.max_replacement_wait_s
            while not self.scheduler.has_unfinished_requests():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self.input_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                self._handle_client_request(*request)
            self._process_input_queue()

        return executed

    return _process_engine_step


def make_init(original_init: Callable[..., None]) -> Callable[..., None]:
    """Set the inline-draft transport mode after community initialization."""

    @wraps(original_init)
    def _ilu_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # [ILU override] EngineCore.__init__ - delta: PP batch-queue outputs
        # carry drafts inline rather than through the delayed worker RPC.
        self.draft_in_model_output = (
            self.batch_queue is not None and self.use_spec_decode
        )

    return _ilu_init


def make_post_step(original_post_step: Callable[..., None]) -> Callable[..., None]:
    """Skip the stale draft-token RPC when outputs already carry drafts."""

    @wraps(original_post_step)
    def _ilu_post_step(self: Any, model_executed: bool) -> None:
        # [ILU override] EngineCore.post_step - delta: Scheduler receives
        # draft tokens from ModelRunnerOutput in update_from_output.
        if getattr(
            self,
            "draft_in_model_output",
            self.batch_queue is not None and self.use_spec_decode,
        ):
            return
        original_post_step(self, model_executed)

    return _ilu_post_step
