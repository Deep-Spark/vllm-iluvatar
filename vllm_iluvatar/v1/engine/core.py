# SPDX-License-Identifier: Apache-2.0
"""Iluvatar wrappers for ``vllm.v1.engine.core``.

ILU wrappers for DP prefill synchronization and replacement-request handling.

Target   : vllm.v1.engine.core.EngineCore.get_kv_cache_group_metadata;
           vllm.v1.engine.core.DPEngineCoreProc.{
           _should_throttle_prefills,_process_engine_step,_handle_client_request}
Baseline : v0.31.0 (db9527a468)
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
           HMA models also need schema version, model identity, layout, and
           schema hash; community metadata is only group_idx / kind /
           block_size / sliding_window.
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
           ``get_kv_cache_group_metadata`` is likewise a wrapper: HMA models
           emit the dynamo_nixl schema rows; every other model delegates to
           the community implementation.
Scope    : Hybrid low-latency decode with plugin global round-robin enabled.
Remove   : When upstream offers rank-synchronized prefill scheduling and a
           bounded post-completion input-queue wait, and when upstream
           EngineCore metadata already carries the HMA schema.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    import torch
    from torch.distributed import Store


def _make_pp_prefill_coalesce(orig, max_wait_s: float):
    """Admit arrivals before short initial PP batches within one wait budget.

    Opt-in for synchronous PP with DP=1. The upstream scheduler and executor
    still own every batch, including empty batches. Waiting stops after the
    first worker result and is never applied to decode or KV-transfer paths.

    Target: EngineCore.step_with_batch_queue at vLLM 341bf7d4.
    Why: early short batches can fill the PP queue before arrivals are admitted.
    How: share one bounded arrival wait across the initial pipeline fill.
    Scope: opt-in synchronous PP, DP=1, non-resumable generation only.
    Remove when upstream supports bounded admission during initial PP fill.
    """

    @wraps(orig)
    def step_with_batch_queue(self):
        config = self.vllm_config
        if (
            max_wait_s <= 0
            or config.parallel_config.data_parallel_size != 1
            or config.scheduler_config.async_scheduling
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or self.is_pooling_model
            or not hasattr(self, "input_queue")
        ):
            return orig(self)

        batches = self.batch_queue
        if not batches:
            self._ilu_pp_initial_fill = True
            self._ilu_pp_coalesce_remaining_s = max_wait_s
            self._ilu_pp_coalesce_last = None

        running = self.scheduler.running
        budget = self.scheduler.max_num_scheduled_tokens
        if (
            getattr(self, "_ilu_pp_initial_fill", False)
            and self._ilu_pp_coalesce_remaining_s > 0
            and batches
            and 2 <= len(batches) < self.batch_queue_size
            and running
            and not self.scheduler.waiting
            and all(
                request.num_tokens == request.num_prompt_tokens
                and not request.resumable
                for request in running
            )
        ):
            remaining = sum(
                max(0, request.num_tokens - request.num_computed_tokens)
                for request in running
            )
            state = tuple(
                (request.request_id, request.num_computed_tokens) for request in running
            )
            if (
                remaining < budget
                and getattr(self, "_ilu_pp_coalesce_last", None) != state
            ):
                self._ilu_pp_coalesce_last = state
                started = time.monotonic()
                try:
                    item = self.input_queue.get(
                        timeout=self._ilu_pp_coalesce_remaining_s
                    )
                except queue.Empty:
                    pass
                else:
                    self._handle_client_request(*item)
                    self._process_input_queue()
                finally:
                    self._ilu_pp_coalesce_remaining_s = max(
                        0.0,
                        self._ilu_pp_coalesce_remaining_s
                        - (time.monotonic() - started),
                    )

        result = orig(self)
        if result[0] is not None:
            self._ilu_pp_initial_fill = False
        return result

    return step_with_batch_queue


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


def _iter_pre_schedule_waiting(scheduler):
    """Walk the queues that still hold a remote-KV / replay request.

    Community 0.31 Scheduler has no ``skipped_waiting``. Skipped and
    WAITING_FOR_REMOTE_KVS holders go back to ``waiting`` /
    ``kv_holding_waiting``. A leftover ``skipped_waiting`` attribute is
    walked only when a scheduler still exposes it.
    """
    skipped = getattr(scheduler, "skipped_waiting", None)
    if skipped is not None:
        yield from skipped
        return
    yield from getattr(scheduler, "waiting", ())
    yield from getattr(scheduler, "kv_holding_waiting", ())


def _local_prefill_state(
    scheduler,
    sync_ready_replay: bool = False,
) -> tuple[bool, bool]:
    running_decode = any(not request.is_prefill_chunk for request in scheduler.running)
    running_prefill = any(request.is_prefill_chunk for request in scheduler.running)
    if not sync_ready_replay:
        return bool(scheduler.waiting) or running_prefill, running_decode

    # Transport admission is not replay readiness. In particular a fresh PD
    # request can sit in WAITING while allocate_slots fails, before its receive
    # has even started. Do not let that request keep the global gate ready.
    finished_recvs = getattr(scheduler, "finished_recving_kv_req_ids", ())

    def compute_ready(request) -> bool:
        status = getattr(request, "status", None)
        if status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            return getattr(request, "request_id", None) in finished_recvs
        if status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            return False
        if getattr(request, "num_stale_output_tokens", 0) > 0 and not getattr(
            request, "drop_stale_output", False
        ):
            return False
        params = getattr(request, "kv_transfer_params", None) or {}
        if (
            params.get("do_remote_prefill")
            and getattr(request, "num_computed_tokens", 0) == 0
        ):
            # Scheduler.schedule still runs on this rank and can allocate/start
            # the receive during a coalescing window (load_kv_async bypass).
            return False
        return True

    active = len(scheduler.running) + getattr(
        scheduler, "num_waiting_for_streaming_input", 0
    )
    max_num_active_reqs = getattr(scheduler, "max_num_active_reqs", None)
    has_slot = max_num_active_reqs is None or active < max_num_active_reqs
    queued_ready = has_slot and any(
        compute_ready(r) for r in _iter_pre_schedule_waiting(scheduler)
    )
    return running_prefill or queued_ready, running_decode


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
    @wraps(orig)
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
        self.scheduler._ilu_ready_replay_throttle = False
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
                continuing_ranks = int(values[2])
                if community_should_throttle is not None:
                    admission_allowed = not community_should_throttle(self)
            prefill_sync_state.update(
                ready_ranks,
                decode_ranks,
                self.dp_size,
                continuing_ranks=continuing_ranks,
                admission_allowed=admission_allowed,
            )
            # Preserve the globally agreed replay gate despite local capacity heuristics.
            self.scheduler._ilu_ready_replay_throttle = (
                prefill_sync_state.sync_ready_replay
                and prefill_sync_state.force_throttle
            )
            # Every rank that participates in prefill synchronization must
            # execute exactly one real or dummy worker step. Otherwise worker
            # and EngineCore collectives can advance in different orders.
            self.engines_running = True
            if prefill_sync_state.force_throttle and waiting and not running_decode:
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


def _make_get_kv_cache_group_metadata(orig):
    # [ILU override] Wrapper, not a full-method copy. Delta: HMA models publish
    # the dynamo_nixl schema rows; every other model keeps community metadata.
    def get_kv_cache_group_metadata(self) -> list[dict]:
        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.dynamo_nixl.schema import (
            build_engine_group_metadata,
            is_hma_model_config,
        )

        if not is_hma_model_config(self.vllm_config):
            return orig(self)
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []
        return build_engine_group_metadata(self.vllm_config, kv_cache_config)

    return get_kv_cache_group_metadata
