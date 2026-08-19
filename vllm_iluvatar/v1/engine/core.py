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

if TYPE_CHECKING:
    import torch
    from torch.distributed import Store


@dataclass(slots=True)
class DPPrefillSyncState:
    enabled: bool
    max_replacement_wait_s: float
    max_defer_steps: int
    sync_group: dist.ProcessGroup | None
    coalesce_steps: int = 2
    partial_steps: int = 0
    force_throttle: bool = False
    sync_tensor: torch.Tensor | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            import torch

            self.sync_tensor = torch.empty(2, dtype=torch.int32, device="cpu")
        else:
            self.sync_tensor = None

    def update(self, ready_ranks: int, decode_ranks: int, dp_size: int) -> bool:
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


def _local_prefill_state(scheduler) -> tuple[bool, bool]:
    prefill_ready = bool(scheduler.waiting) or any(
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


def _make_process_engine_step(orig):
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
            )
        prefill_sync_state = self._ilu_dp_prefill_sync_state
        waiting = False
        running_decode = False
        sync_active = prefill_sync_state.enabled and (
            self.engines_running or self.scheduler.has_unfinished_requests()
        )
        if sync_active:
            import torch

            waiting, running_decode = _local_prefill_state(self.scheduler)
            state = prefill_sync_state.sync_tensor
            assert state is not None
            state[0] = int(waiting)
            state[1] = int(running_decode)
            dist.all_reduce(
                state,
                op=dist.ReduceOp.SUM,
                group=prefill_sync_state.sync_group,
            )
            ready_ranks, decode_ranks = (int(value) for value in state.tolist())

            prefill_sync_state.update(ready_ranks, decode_ranks, self.dp_size)
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
