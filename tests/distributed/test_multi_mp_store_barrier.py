# SPDX-License-Identifier: Apache-2.0
"""CPU actual-Multi tests for the pinned non-lazy MP producer combination.

Child completion arrivals and LM's terminal-history contract are controlled.
The separate phase-order reproduction also exercises the real metadata bind,
rank aggregator, Nixl scheduler filtering and Scheduler free consumer.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("test_multi_completion_sources.py")
spec = importlib.util.spec_from_file_location("mp_barrier_cpu_helpers", SOURCE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class MPStoreBarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.MultiCompletionTests.setUpClass()

    def setUp(self):
        self.case = base.MultiCompletionTests(
            "test_legal_lmcache_completion_both_orders_reaches_actual_parent_free"
        )
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def pair(self, reverse=False, lazy=False, can_store=True):
        nx_original = self.case.child()
        nx_original.reports.clear()
        nx = type("OptNixlConnector", (SimpleNamespace,), {})(**vars(nx_original))
        lm_original = self.case.child()
        lm_original.reports.clear()
        lm = type("LMCacheMPConnector", (SimpleNamespace,), {})(**vars(lm_original))
        lm._can_store = can_store
        lm.lazy_offload = lazy
        lm.worker_adapter = SimpleNamespace(
            lazy_offload=lazy,
            _ilu_enable_terminal_transfer_fence=lambda *, generation_aware: generation_aware,
        )
        nx.connector_scheduler = SimpleNamespace(is_consumer=False)
        nx.connector_worker = SimpleNamespace(is_consumer=False)
        worker = self.case.multi([lm, nx] if reverse else [nx, lm])
        worker._ilu_terminal_fence_profile = True
        return worker, nx, lm

    def poll(self, worker, finished=()):
        value = worker.get_transfer_results(set(finished))
        value.kv_connector_worker_meta = worker.build_connector_worker_meta()
        return value

    def send(self, child, rid="R"):
        child.reports.append(base.Transfer({rid}))

    def assert_drained(self, worker):
        self.assertFalse(worker._extra_async_saves)
        self.assertFalse(worker._ilu_multi_worker_send_sources)

    def test_explicit_profile_requires_request_level_failure_reporting(self):
        _, _, child = self.pair()
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
        )
        env = {'LMCACHE_ILUVATAR_TERMINAL_TRANSFERS': '1',
               'LMCACHE_MP_TRANSFER_MODE': 'lmcache_driven',
               'VLLM_ILUVATAR_TERMINAL_KV_OWNERSHIP': '1'}
        with patch.dict(os.environ, env):
            self.assertFalse(self.case.module._terminal_fence_profile(child, config))
            child._report_request_errors = True
            self.assertTrue(self.case.module._terminal_fence_profile(child, config))

    def test_only_explicit_eligible_profile_rejects_old_lp_worker_or_scheduler(self):
        _, nixl, child = self.pair()
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
        )

        class Multi:
            def __init__(self, connectors):
                self._connectors = connectors
                self._vllm_config = config

            def get_transfer_results(self, _ids):
                raise AssertionError('Only construction is exercised')

            build_connector_worker_meta = get_transfer_results
            update_connector_output = get_transfer_results

        self.case.module.install_multi_completion_sources(Multi)
        env = {'LMCACHE_ILUVATAR_TERMINAL_TRANSFERS': '1',
               'LMCACHE_MP_TRANSFER_MODE': 'lmcache_driven',
               'VLLM_ILUVATAR_TERMINAL_KV_OWNERSHIP': '1'}
        child.worker_adapter = SimpleNamespace()
        with patch.dict(os.environ, env):
            self.assertFalse(hasattr(Multi([nixl, child]), '_ilu_terminal_fence_profile'))
            child._report_request_errors = True
            with self.assertRaisesRegex(RuntimeError, 'requires the matching LMCache plugin'):
                Multi([nixl, child])
            child.scheduler_adapter = SimpleNamespace()
            with self.assertRaisesRegex(RuntimeError, 'requires the matching LMCache plugin'):
                Multi([nixl, child])
            child._ilu_enable_terminal_session_fence = lambda *, generation_aware: None
            self.assertTrue(Multi([nixl, child])._ilu_terminal_fence_profile)
        with patch.dict(os.environ, {'LMCACHE_ILUVATAR_TERMINAL_TRANSFERS': '0'}):
            self.assertFalse(hasattr(Multi([nixl, child]), '_ilu_terminal_fence_profile'))

    def test_early_nixl_waits_for_late_extra_and_actual_lm_source(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                worker, nx, lm = self.pair(reverse)
                self.send(nx)
                self.assertFalse(self.poll(worker).finished_sending)
                worker._extra_async_saves["R"] = 1  # real bind covered separately
                self.assertFalse(self.poll(worker, {"R"}).finished_sending)
                self.send(lm)
                result = self.poll(worker)
                self.assertEqual(result.finished_sending, {"R"})
                self.assertEqual(result.kv_connector_worker_meta.send_sources, {"R": {0, 1}})
                self.assert_drained(worker)

    def test_lm_first_keeps_extra_and_waits_for_nixl(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                worker, nx, lm = self.pair(reverse)
                worker._extra_async_saves["R"] = 1
                self.send(lm)
                self.assertFalse(self.poll(worker, {"R"}).finished_sending)
                self.assertEqual(worker._extra_async_saves, {"R": 1})
                worker._extra_async_saves["R"] = 1  # replay while LM is held
                self.send(nx)
                self.assertEqual(self.poll(worker).finished_sending, {"R"})
                self.assert_drained(worker)

    def test_only_lm_async_still_requires_lm_source(self):
        worker, nx, lm = self.pair()
        self.send(nx)
        self.assertFalse(self.poll(worker).finished_sending)
        self.send(lm)
        self.assertEqual(self.poll(worker, {"R"}).finished_sending, {"R"})
        self.assert_drained(worker)

    def test_duplicate_child_does_not_consume_distinct_responsibility(self):
        worker, nx, lm = self.pair()
        worker._extra_async_saves["R"] = 1
        for _ in range(3):
            self.send(nx)
            self.assertFalse(self.poll(worker).finished_sending)
            self.assertEqual(worker._extra_async_saves, {"R": 1})
        self.send(lm)
        self.assertEqual(self.poll(worker).finished_sending, {"R"})
        self.assert_drained(worker)

    def test_replayed_metadata_and_late_nixl_do_not_reopen_completion(self):
        worker, nx, lm = self.pair()
        worker._extra_async_saves["R"] = 1
        self.send(nx)
        self.send(lm)
        self.assertEqual(self.poll(worker).finished_sending, {"R"})
        worker._extra_async_saves["R"] = 1
        self.assertFalse(self.poll(worker).finished_sending)
        self.assert_drained(worker)
        self.send(nx)
        self.assertFalse(self.poll(worker).finished_sending)
        self.assert_drained(worker)

    def test_late_send_suppression_preserves_receives_errors_and_metadata(self):
        worker, nx, lm = self.pair()
        self.send(lm)
        self.assertEqual(self.poll(worker, {"R"}).finished_sending, {"R"})
        marker = object()
        nx.build_connector_worker_meta = lambda: marker
        value = base.Transfer({"R"}, {"recv"})
        value.failed_recving = {"recv"}
        nx.reports.append(value)
        result = self.poll(worker)
        self.assertFalse(result.finished_sending)
        self.assertEqual(result.finished_recving, {"recv"})
        self.assertEqual(result.failed_recving, {"recv"})
        self.assertIs(result.kv_connector_worker_meta.metadata[0], marker)
        self.assertEqual(result.kv_connector_worker_meta.recv_sources, {"recv": {0}})
        self.assert_drained(worker)

    def test_valid_send_reaches_original_nixl_filter_and_scheduler_free(self):
        worker, nx, lm = self.pair()
        worker._extra_async_saves["R"] = 1
        self.send(nx)
        self.assertFalse(self.poll(worker).finished_sending)
        self.send(lm)
        result = self.poll(worker, {"R"})
        sx = self.case.child(nixl=True, pending={"R"}, awaiting={"R"})
        scheduler = self.case.multi([sx, self.case.child()])
        parent, freed = self.case.parent(scheduler, ["R"])
        parent._update_from_kv_xfer_finished(result)
        self.assertEqual(freed, ["R"])
        self.assertFalse(parent.requests)
        self.assertEqual(sx.seen[-1][0], {"R"})
        self.assertFalse(sx.connector_scheduler._pending_worker_sends)

    def test_lazy_consumer_and_non_mp_keep_original_policy(self):
        for lazy, can_store, rename in [(True, True, False), (False, False, False),
                                         (False, True, True)]:
            with self.subTest(lazy=lazy, can_store=can_store, rename=rename):
                worker, nx, lm = self.pair(lazy=lazy, can_store=can_store)
                if rename:
                    worker._connectors[1] = SimpleNamespace(**vars(lm))
                self.send(nx)
                self.assertEqual(self.poll(worker).finished_sending, {"R"})

    def test_unsupported_recognized_mp_contract_fails_explicitly(self):
        worker, _, lm = self.pair()
        del lm.worker_adapter.lazy_offload
        with self.assertRaisesRegex(RuntimeError, "completion contract"):
            self.poll(worker)

    def test_invalid_two_child_extra_is_rejected(self):
        for extra in (0, -1, 2, True):
            with self.subTest(extra=extra):
                worker, nx, _ = self.pair()
                worker._extra_async_saves["R"] = extra
                self.send(nx)
                with self.assertRaisesRegex(RuntimeError, "async-save count"):
                    self.poll(worker)

    def test_many_completed_ids_leave_no_new_pending_state(self):
        worker, nx, lm = self.pair()
        for i in range(1000):
            rid = f"R{i}"
            self.send(nx, rid)
            self.assertFalse(self.poll(worker).finished_sending)
            worker._extra_async_saves[rid] = 1
            self.send(lm, rid)
            self.assertEqual(self.poll(worker, {rid}).finished_sending, {rid})
            self.send(nx, rid)
            self.assertFalse(self.poll(worker).finished_sending)
            self.assert_drained(worker)
        self.assertFalse(hasattr(lm.worker_adapter, "_returned_finished"))

    def test_actual_mp_no_store_terminal_history_contract(self):
        source = os.environ.get("LMCACHE_ADAPTER_SOURCE")
        if not source:
            self.skipTest("Set LMCACHE_ADAPTER_SOURCE to qualify the pinned MP source")
        scope = {}
        methods = base.definitions(Path(source), "LMCacheMPWorkerAdapter",
                                   ["_process_finished_stores"], scope)
        adapter = SimpleNamespace(finished_stores=set(), previously_finished=set(),
                                  store_futures={}, _returned_finished=set(),
                                  _update_and_get_finished_store=lambda: set())
        process = MethodType(methods["_process_finished_stores"], adapter)
        self.assertEqual(process(set(), set()), set())
        self.assertEqual(process(set(), {"R"}), {"R"})
        self.assertEqual(adapter._returned_finished, {"R"})
        self.assertEqual(process(set(), {"R"}), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
