# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU tests for concurrent scheduling; no GPU imports or model weights."""

import importlib.util
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "smoke_schedule_runner", Path(__file__).with_name("runner.py")
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SchedulerTests(unittest.TestCase):
    def args(self, parallel=2):
        return SimpleNamespace(
            devices=",".join(map(str, range(16))), max_parallel_models=parallel
        )

    def models(self, sizes):
        return [dict(id="model-" + str(i), logical_gpus=n) for i, n in enumerate(sizes)]

    def result(self, model, **extra):
        return dict(id=model["id"], status="PASS", correct_answers=20, **extra)

    def test_two_eight_gpu_models_overlap_with_disjoint_devices(self):
        barrier = threading.Barrier(2)
        allocations = {}

        def run(model, cases, args, control):
            allocations[model["id"]] = set(args.devices.split(","))
            barrier.wait(timeout=3)
            return self.result(model)

        results = smoke.schedule_models(self.models([8, 8]), [], self.args(), run=run)
        a, b = list(allocations.values())
        self.assertEqual(len(results), 2)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(b), 8)
        self.assertFalse(a & b)
        self.assertEqual(a | b, set(map(str, range(16))))

    def test_sixteen_gpu_model_is_exclusive_and_all_models_run_once(self):
        live = set()
        mutex = threading.Lock()
        seen = []

        def run(model, cases, args, control):
            assigned = set(args.devices.split(","))
            with mutex:
                self.assertFalse(live & assigned)
                if model["logical_gpus"] == 16:
                    self.assertFalse(live)
                live.update(assigned)
                self.assertLessEqual(len(live), 16)
                seen.append(model["id"])
            time.sleep(0.02)
            with mutex:
                live.difference_update(assigned)
            return self.result(model)

        selected = self.models([8, 16, 8, 16, 8, 8])
        results = smoke.schedule_models(selected, [], self.args(), run=run)
        self.assertCountEqual(seen, [m["id"] for m in selected])
        self.assertEqual(len(results), len(selected))

    def test_three_four_gpu_models_start_together(self):
        barrier = threading.Barrier(3)
        allocations = {}

        def run(model, cases, args, control):
            allocations[model["id"]] = set(args.devices.split(","))
            barrier.wait(timeout=3)
            return self.result(model)

        results = smoke.schedule_models(
            self.models([4, 4, 4]), [], self.args(3), run=run
        )
        groups = list(allocations.values())
        self.assertEqual(len(results), 3)
        self.assertEqual([len(g) for g in groups], [4, 4, 4])
        self.assertEqual(len(set.union(*groups)), 12)

    def test_mixed_sizes_respect_capacity_and_large_model_is_exclusive(self):
        barrier = threading.Barrier(3)
        live = set()
        mutex = threading.Lock()
        selected = self.models([4, 4, 8, 16])

        def run(model, cases, args, control):
            assigned = set(args.devices.split(","))
            with mutex:
                self.assertFalse(live & assigned)
                if model["logical_gpus"] == 16:
                    self.assertFalse(live)
                live.update(assigned)
                self.assertLessEqual(len(live), 16)
            if model["logical_gpus"] != 16:
                barrier.wait(timeout=3)
            with mutex:
                live.difference_update(assigned)
            return self.result(model)

        results = smoke.schedule_models(selected, [], self.args(3), run=run)
        self.assertCountEqual([r["id"] for r in results], [m["id"] for m in selected])

    def test_three_way_cleanup_failure_cancels_both_peers(self):
        barrier = threading.Barrier(3)
        started = []

        def run(model, cases, args, control):
            started.append(model["id"])
            barrier.wait(timeout=3)
            if model["id"] == "model-0":
                return self.result(model, cleanup_failed=True)
            self.assertTrue(control.cancelled.wait(timeout=3))
            return self.result(model, interrupted=True)

        results = smoke.schedule_models(
            self.models([4, 4, 4, 4]), [], self.args(3), run=run
        )
        self.assertCountEqual(started, ["model-0", "model-1", "model-2"])
        self.assertEqual(len(results), 3)

    def test_serial_override_preserves_suite_order(self):
        seen = []

        def run(model, cases, args, control):
            seen.append(model["id"])
            return self.result(model)

        selected = self.models([8, 16, 8])
        smoke.schedule_models(selected, [], self.args(1), run=run)
        self.assertEqual(seen, [m["id"] for m in selected])

    def test_cleanup_failure_cancels_peer_and_never_starts_pending(self):
        barrier = threading.Barrier(2)
        started = []

        def run(model, cases, args, control):
            started.append(model["id"])
            barrier.wait(timeout=3)
            if model["id"] == "model-0":
                return self.result(model, cleanup_failed=True)
            self.assertTrue(control.cancelled.wait(timeout=3))
            return self.result(model, interrupted=True)

        results = smoke.schedule_models(
            self.models([8, 8, 8]), [], self.args(), run=run
        )
        self.assertCountEqual(started, ["model-0", "model-1"])
        self.assertEqual(len(results), 2)

    def test_worker_exception_cancels_peer(self):
        barrier = threading.Barrier(2)

        def run(model, cases, args, control):
            barrier.wait(timeout=3)
            if model["id"] == "model-0":
                raise RuntimeError("injected worker failure")
            self.assertTrue(control.cancelled.wait(timeout=3))
            return self.result(model, interrupted=True)

        with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
            smoke.schedule_models(self.models([8, 8, 8]), [], self.args(), run=run)

    def test_invalid_capacity_is_rejected_before_launch(self):
        run = Mock()
        for selected, args in [
            (self.models([32]), self.args()),
            (self.models([8]), self.args(4)),
        ]:
            with self.assertRaises(ValueError):
                smoke.schedule_models(selected, [], args, run=run)
        run.assert_not_called()

    def test_cancel_terminates_only_registered_cpu_process_groups(self):
        control = smoke.RunControl()
        owned = []
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
        )
        try:
            first = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                start_new_session=True,
            )
            owned.append(first)
            control.register(first)
            control.cancel()
            self.assertEqual(first.wait(timeout=3), -signal.SIGTERM)
            second = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                start_new_session=True,
            )
            owned.append(second)
            control.register(second)
            self.assertEqual(second.wait(timeout=3), -signal.SIGTERM)
            self.assertIsNone(unrelated.poll())
        finally:
            for proc in owned + [unrelated]:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)

    def test_concurrent_ports_are_unique(self):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: smoke.choose_ports(), range(16)))
        ports = [port for pair in values for port in pair]
        self.assertEqual(len(ports), len(set(ports)))


if __name__ == "__main__":
    unittest.main()
