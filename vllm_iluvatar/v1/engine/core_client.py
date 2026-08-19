# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/engine/core_client.py

ILU override implementation for
DPLBAsyncMPClient.get_core_engine_for_request.  This module holds ONLY the
override implementation; the wiring (register/_apply, fingerprint + signature
guards, binding) lives in
``ilu_patch.general_patches.patch_core_client`` and is auto-discovered during
``register_general``.

Target   : vllm.v1.engine.core_client.DPLBAsyncMPClient.get_core_engine_for_request
Baseline : main_6e448d0
Why      : The community LB relies on coordinator-reported (waiting, running)
           counts that lag ~100 ms.  Exact-inflight removes that lag for one
           client; global round-robin gives stable request distribution across
           multiple API server processes.
How      : full-method copy with small env-gated deltas (see ``ILU
           MODIFICATION`` markers).
           Class attribute ``_ilu_use_exact_inflight`` is read from
           ``VLLM_ILU_DP_LB_EXACT_INFLIGHT`` (registered in ``envs.py``).
Scope    : DPLBAsyncMPClient on Iluvatar when data-parallelism is enabled.
Remove   : when upstream adopts a pluggable LB strategy or the exact-inflight
           algorithm.

Round-robin validation:
  - Unit microbench compared the original per-request open/lock/update/close
    counter with the current cached fd + mmap + flock counter; cached state
    avoids repeated file setup while preserving cross-API-server coordination.
  - Model perf checks showed RR mainly helps multi-API-server DP+EP traffic by
    avoiding stale least-loaded counts and distributing requests more evenly.

Community globals (envs, sys, get_late_interaction_engine_index) are bound onto
this module by ``patch_core_client._apply()`` at register-time (kept lazy so
importing this module never pulls in vllm).
"""

from __future__ import annotations

import contextlib
import fcntl
import mmap
import os
import struct

# Community globals bound by patch_core_client._apply().
envs = None
sys = None
get_late_interaction_engine_index = None

_DP_RR_MMAP_SIZE = 8
_DP_RR_COUNTER = struct.Struct("Q")
_DP_RR_MMAP_CACHE: dict[str, tuple[int, mmap.mmap]] = {}


def _dp_round_robin_lb_enabled() -> bool:
    return envs.VLLM_ILU_DP_ROUND_ROBIN_LB


def _get_global_rr_mmap_state(
    rr_file: str,
    fallback_index: int,
) -> tuple[int, mmap.mmap]:
    state = _DP_RR_MMAP_CACHE.get(rr_file)
    if state is not None:
        return state

    fd = os.open(rr_file, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.fstat(fd).st_size < _DP_RR_MMAP_SIZE:
            os.ftruncate(fd, _DP_RR_MMAP_SIZE)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, _DP_RR_COUNTER.pack(fallback_index))
        fcntl.flock(fd, fcntl.LOCK_UN)
        mm = mmap.mmap(fd, _DP_RR_MMAP_SIZE, access=mmap.ACCESS_WRITE)
    except Exception:
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise

    state = (fd, mm)
    _DP_RR_MMAP_CACHE[rr_file] = state
    return state


def _next_global_rr_engine_index(
    num_engines: int,
    fallback_index: int,
) -> tuple[int, int]:
    rr_file = envs.VLLM_ILU_DP_ROUND_ROBIN_LB_FILE
    if not rr_file:
        rr_file = f"/tmp/vllm-dp-rr-{os.getppid()}-{num_engines}.state"

    fd = None
    try:
        fd, mm = _get_global_rr_mmap_state(rr_file, fallback_index)
        fcntl.flock(fd, fcntl.LOCK_EX)
        next_index = _DP_RR_COUNTER.unpack_from(mm, 0)[0]
        eng_index = next_index % num_engines
        _DP_RR_COUNTER.pack_into(mm, 0, (next_index + 1) % num_engines)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return eng_index, (eng_index + 1) % num_engines
    except Exception:
        if fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(fd, fcntl.LOCK_UN)
        eng_index = fallback_index % num_engines
        return eng_index, (eng_index + 1) % num_engines


# [ILU override] get_core_engine_for_request — delta: exact-inflight LB
# branch (env-gated).  Full-method copy of the baseline; only the lines
# inside the MODIFICATION blocks differ from community.
def _ilu_get_core_engine_for_request(self, request):
    if (eng_index := request.data_parallel_rank) is None and (
        eng_index := get_late_interaction_engine_index(
            request.pooling_params, len(self.core_engines)
        )
    ) is None:
        num_engines = len(self.core_engines)
        # >>> ILU MODIFICATION — global RR / exact-inflight LB (env-gated) >>>
        if _dp_round_robin_lb_enabled():
            eng_index, self.eng_start_index = _next_global_rr_engine_index(
                num_engines,
                self.eng_start_index,
            )
        elif self._ilu_use_exact_inflight:
            eng_index = min(
                range(num_engines),
                key=lambda i: self.engine_inflight[self.core_engines[i]],
            )
        # <<< ILU MODIFICATION <<<
        else:
            current_counts = self.lb_engines
            min_score: float = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                idx = (self.eng_start_index + i) % num_engines
                waiting, running, kv_cache_usage = current_counts[idx]
                inflight = self.engine_inflight[self.core_engines[idx]]
                score: float = max(self.client_count * inflight, waiting + running)
                if waiting:
                    score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
                if score < min_score:
                    min_score = score
                    eng_index = idx
            current_counts[eng_index][0] += self.client_count
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

    chosen_engine = self.core_engines[eng_index]
    self.reqs_in_flight[request.request_id] = chosen_engine
    self.engine_inflight[chosen_engine] += 1
    return chosen_engine
