# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar DPLBAsyncMPClient LB patch."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_core_client


def _make_request(request_id, data_parallel_rank=None, pooling_params=None):
    return SimpleNamespace(
        request_id=request_id,
        data_parallel_rank=data_parallel_rank,
        pooling_params=pooling_params,
    )


def _setup_modules(monkeypatch, use_exact_inflight: bool):
    """Inject mock vllm modules and apply the patch."""

    # Mock vllm.envs
    envs_module = ModuleType("vllm.envs")
    envs_module.VLLM_ILU_DP_LB_EXACT_INFLIGHT = use_exact_inflight
    envs_module.VLLM_ILU_DP_ROUND_ROBIN_LB = False
    envs_module.VLLM_ILU_DP_ROUND_ROBIN_LB_FILE = ""

    # Mock get_late_interaction_engine_index (always returns None)
    pool_module = ModuleType("vllm.v1.pool.late_interaction")
    pool_module.get_late_interaction_engine_index = lambda params, n: None

    class FakeDPLBAsyncMPClient:
        def __init__(self, engines, client_count=1, client_index=0):
            self.core_engines = engines
            self.client_count = client_count
            self.eng_start_index = (len(engines) * client_index) // client_count
            self.reqs_in_flight: dict[str, bytes] = {}
            self.engine_inflight = {engine: 0 for engine in engines}
            self.lb_engines = [[0, 0, 0.0] for _ in engines]

        def get_core_engine_for_request(self, request):
            raise NotImplementedError("should be patched")

    core_client_module = ModuleType("vllm.v1.engine.core_client")
    core_client_module.DPLBAsyncMPClient = FakeDPLBAsyncMPClient

    # Disable fingerprint check — the fake class source won't match the
    # real upstream baseline snapshot.
    monkeypatch.setenv("VLLM_ILU_PATCH_CHECK_FINGERPRINT", "0")

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", ModuleType("vllm.v1.engine"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", core_client_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.pool", ModuleType("vllm.v1.pool"))
    monkeypatch.setitem(sys.modules, "vllm.v1.pool.late_interaction", pool_module)

    patch_core_client._apply()
    return FakeDPLBAsyncMPClient


class TestExactInflightLB:
    """Tests for the exact-inflight LB strategy."""

    def test_routes_to_least_loaded_engine(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=True)
        engines = [b"\x00\x00", b"\x01\x00", b"\x02\x00"]
        client = Cls(engines)

        # Simulate engine 0 has 2 in-flight, engine 1 has 1, engine 2 has 0.
        client.reqs_in_flight = {"a": engines[0], "b": engines[0], "c": engines[1]}
        client.engine_inflight = {engines[0]: 2, engines[1]: 1, engines[2]: 0}

        chosen = client.get_core_engine_for_request(_make_request("d"))
        assert chosen == engines[2], "should route to engine with 0 in-flight"

    def test_ties_broken_to_lower_index(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=True)
        engines = [b"\x00\x00", b"\x01\x00"]
        client = Cls(engines)

        # Both engines empty — should pick index 0.
        chosen = client.get_core_engine_for_request(_make_request("a"))
        assert chosen == engines[0]

    def test_tracks_chosen_engine_in_reqs_in_flight(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=True)
        engines = [b"\x00\x00", b"\x01\x00"]
        client = Cls(engines)

        client.get_core_engine_for_request(_make_request("req1"))
        assert "req1" in client.reqs_in_flight


class TestCoordinatorLBWithTieBreaking:
    """Tests for the coordinator-based LB path with tie-breaking rotation."""

    def test_picks_lowest_score_engine(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=False)
        engines = [b"\x00\x00", b"\x01\x00"]
        client = Cls(engines)
        # Engine 0: waiting=5, running=1, kv=0.0 -> score=6
        # Engine 1: waiting=1, running=2, kv=0.0 -> score=3
        client.lb_engines = [[5, 1, 0.0], [1, 2, 0.0]]

        chosen = client.get_core_engine_for_request(_make_request("a"))
        assert chosen == engines[1]

    def test_tie_breaking_rotates_start_index(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=False)
        engines = [b"\x00\x00", b"\x01\x00", b"\x02\x00"]
        client = Cls(engines)
        client.lb_engines = [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]]
        start = client.eng_start_index

        client.get_core_engine_for_request(_make_request("a"))
        assert client.eng_start_index == (start + 1) % 3

        # Reset counts for next round.
        client.lb_engines = [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]]
        client.get_core_engine_for_request(_make_request("b"))
        assert client.eng_start_index == (start + 2) % 3

    def test_increments_local_waiting_count(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=False)
        engines = [b"\x00\x00", b"\x01\x00"]
        client = Cls(engines, client_count=2)
        client.lb_engines = [[0, 0, 0.0], [0, 0, 0.0]]

        client.get_core_engine_for_request(_make_request("a"))
        # Chosen engine's waiting count should be incremented by client_count.
        chosen_idx = list(client.reqs_in_flight.values())[0]
        idx = engines.index(chosen_idx)
        assert client.lb_engines[idx][0] == 2


class TestDataParallelRankBypass:
    """When request.data_parallel_rank is set, LB is bypassed."""

    def test_uses_explicit_rank(self, monkeypatch):
        Cls = _setup_modules(monkeypatch, use_exact_inflight=True)
        engines = [b"\x00\x00", b"\x01\x00", b"\x02\x00"]
        client = Cls(engines)

        chosen = client.get_core_engine_for_request(
            _make_request("x", data_parallel_rank=2)
        )
        assert chosen == engines[2]
