# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/engine/core_client.py

ILU override implementation for
DPLBAsyncMPClient.get_core_engine_for_request.  This module holds ONLY the
override implementation; the wiring (register/_apply, fingerprint + signature
guards, binding) lives in
``ilu_patch.general_patches.patch_core_client`` and is auto-discovered during
``register_general``.

Target   : vllm.v1.engine.core_client.DPLBAsyncMPClient.get_core_engine_for_request
Baseline : main_88e5e2c57be8
Why      : The community LB relies on coordinator-reported (waiting, running)
           counts that lag ~100 ms.  With a single API server the client
           already holds an exact, real-time record of every in-flight request.
           Counting that directly eliminates the lag and gives optimal
           least-loaded routing.
How      : full-method copy with one delta (see ``ILU MODIFICATION`` markers).
           Class attribute ``_ilu_use_exact_inflight`` is read from
           ``VLLM_ILU_DP_LB_EXACT_INFLIGHT`` (registered in ``envs.py``).
Scope    : DPLBAsyncMPClient on Iluvatar when data-parallelism is enabled.
Remove   : when upstream adopts a pluggable LB strategy or the exact-inflight
           algorithm.

Community globals (sys, get_late_interaction_engine_index) are bound onto this
module by ``patch_core_client._apply()`` at register-time (kept lazy so
importing this module never pulls in vllm).
"""

from __future__ import annotations

# Community globals bound by patch_core_client._apply().
sys = None
get_late_interaction_engine_index = None


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
        # >>> ILU MODIFICATION — exact-inflight LB (env-gated) >>>
        if self._ilu_use_exact_inflight:
            counts = [0] * num_engines
            index_of = {
                eng: i for i, eng in enumerate(self.core_engines)
            }
            for eng in self.reqs_in_flight.values():
                i = index_of.get(eng)
                if i is not None:
                    counts[i] += 1
            eng_index = min(range(num_engines), key=lambda i: counts[i])
        # <<< ILU MODIFICATION <<<
        else:
            current_counts = self.lb_engines
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            current_counts[eng_index][0] += self.client_count
            self.eng_start_index = (
                self.eng_start_index + 1
            ) % num_engines

    chosen_engine = self.core_engines[eng_index]
    self.reqs_in_flight[request.request_id] = chosen_engine
    return chosen_engine
