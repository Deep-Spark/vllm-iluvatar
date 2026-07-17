# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar DP load-balancer overrides.

Corresponds to: vllm/v1/engine/core_client.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.v1.engine.core_client.DPLBAsyncMPClient.get_core_engine_for_request
Baseline : main_88e5e2c57be8
Why      : The community LB relies on coordinator-reported (waiting, running)
           counts that lag ~100 ms.  With a single API server
           (``client_count == 1``) the client already holds an exact, real-time
           record of every in-flight request (``reqs_in_flight``).  Counting
           that directly eliminates the lag and gives optimal least-loaded
           routing.  (The ``eng_start_index`` rotation for tie-breaking has
           been adopted upstream and is no longer an ILU-only delta.)
How      : full-method copy with one delta (see ``ILU MODIFICATION`` markers),
           guarded by a source fingerprint against the baseline snapshot
           inlined below.  Class attribute ``_ilu_use_exact_inflight`` is read
           from ``VLLM_ILU_DP_LB_EXACT_INFLIGHT`` (registered in ``envs.py``).
Scope    : DPLBAsyncMPClient on Iluvatar when data-parallelism is enabled.
Remove   : when upstream adopts a pluggable LB strategy or the exact-inflight
           algorithm.
Fingerprint: get_core_engine_for_request community source is hashed at
           register-time against the GET_CORE_ENGINE_FOR_REQUEST baseline
           below; gated by VLLM_ILU_PATCH_CHECK_FINGERPRINT (default on) so
           startup fails fast if upstream changed the body.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of
# DPLBAsyncMPClient.get_core_engine_for_request (4-space method indent),
# used as the drift anchor by the fingerprint guard.
#
# Baseline: main_88e5e2c57be8
GET_CORE_ENGINE_FOR_REQUEST = '''\
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count
            # Rotate the scan start so that ties (equal scores, e.g. right
            # after a coordinator stats reset when engines look equally loaded)
            # don't systematically favor the same engine. This removes the
            # fixed tie-break bias without affecting load-aware decisions when
            # scores actually differ.
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
'''


def register() -> None:
    """Register Iluvatar DPLBAsyncMPClient LB override."""
    p.register_once(
        __name__,
        _apply,
        detail="DPLBAsyncMPClient.get_core_engine_for_request override",
    )


def _apply() -> None:
    import sys as _sys
    import types

    from vllm import envs
    from vllm.v1.engine.core_client import DPLBAsyncMPClient
    from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

    import vllm_iluvatar.v1.engine.core_client as impl

    # Drift guard for the full-method copy (VLLM_ILU_PATCH_CHECK_FINGERPRINT).
    p.check_fingerprint(
        "DPLBAsyncMPClient.get_core_engine_for_request",
        inspect.getsource(DPLBAsyncMPClient.get_core_engine_for_request),
        GET_CORE_ENGINE_FOR_REQUEST,
        resync_hint=(
            "Re-sync _ilu_get_core_engine_for_request in "
            "vllm_iluvatar/v1/engine/core_client.py and the "
            "GET_CORE_ENGINE_FOR_REQUEST baseline in "
            "ilu_patch/general_patches/patch_core_client.py to the new "
            "community source of DPLBAsyncMPClient.get_core_engine_for_request."
        ),
    )

    # Strict signature validation (impl function uses prefixed name;
    # wrap with SimpleNamespace to match community method name).
    _ilu_ns = types.SimpleNamespace(
        get_core_engine_for_request=impl._ilu_get_core_engine_for_request,
    )
    p.check_signatures(
        DPLBAsyncMPClient, _ilu_ns,
        ["get_core_engine_for_request"],
        strict=True,
    )

    # Bind community globals onto the impl module.
    impl.sys = _sys
    impl.get_late_interaction_engine_index = get_late_interaction_engine_index

    # Set env-gated class attribute.
    use_exact = bool(getattr(envs, "VLLM_ILU_DP_LB_EXACT_INFLIGHT", False))
    DPLBAsyncMPClient._ilu_use_exact_inflight = use_exact  # type: ignore[attr-defined]

    DPLBAsyncMPClient.get_core_engine_for_request = (  # type: ignore[assignment]
        impl._ilu_get_core_engine_for_request
    )
