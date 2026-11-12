"""CPU tests for late producer cancellations and the existing engine wake queue."""
import gc
import importlib.util
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
import weakref

PATH = Path(__file__).resolve().parents[2] / "vllm_iluvatar/distributed/kv_transfer/kv_connector/v1/opt_nixl/cancel_progress.py"
SPEC = importlib.util.spec_from_file_location("cancel_progress_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Progress = MODULE.ProducerCancelProgress


class CancelProgressTests(unittest.TestCase):
    def test_pending_before_bind_notifies_once_and_duplicate_coalesces(self):
        progress, signals = Progress(), queue.Queue()
        progress.begin("R", "G")
        self.assertTrue(progress.has_pending())
        progress.bind(lambda: signals.put_nowait("wake"))
        for _ in range(100):
            progress.begin("R", "G")
        self.assertEqual(signals.qsize(), 1)
        progress.complete("R", "G")
        self.assertFalse(progress.has_pending())
        progress.begin("R", "G2")
        self.assertEqual(signals.qsize(), 2)

    def test_old_proof_cannot_close_new_generation(self):
        progress = Progress()
        progress.begin("R", "new")
        progress.complete("R", "old")
        self.assertTrue(progress.has_pending())
        progress.complete("R", "new")
        self.assertFalse(progress.has_pending())

    def test_failed_wake_keeps_ownership_and_retries(self):
        progress, calls = Progress(), []
        def wake():
            calls.append(1)
            if len(calls) == 1:
                raise queue.Full("CPU queue failure")
        progress.bind(wake)
        with self.assertLogs(MODULE.logger, "ERROR"):
            progress.begin("R", "G")
        self.assertTrue(progress.has_pending())
        progress.begin("R", "G")
        progress.begin("R", "G")
        self.assertEqual(len(calls), 2)

    def test_last_proof_closes_batch_and_new_batch_wakes(self):
        progress, signals = Progress(), []
        progress.bind(lambda: signals.append(1))
        progress.begin("A", "GA")
        progress.begin("B", "GB")
        progress.complete("A", "GA")
        self.assertTrue(progress.has_pending())
        self.assertEqual(len(signals), 1)
        progress.complete("B", "GB")
        self.assertFalse(progress.has_pending())
        progress.begin("C", "GC")
        self.assertEqual(len(signals), 2)

    def make_engine(self, progress, *, consumer=False):
        scheduler = SimpleNamespace(is_consumer=consumer, bind_engine_wakeup=progress.bind)
        child = SimpleNamespace(connector_scheduler=scheduler)
        root = SimpleNamespace(sub_connectors=[child, child, SimpleNamespace()])
        class Engine:
            def _process_input_queue(self):
                self.original_calls += 1
                self.entered.set()
                return self.input_queue.get(timeout=2)
        engine = Engine()
        engine.scheduler = SimpleNamespace(connector=root)
        engine.input_queue = queue.Queue()
        engine.entered = threading.Event()
        engine.original_calls = 0
        MODULE.install_engine_cancel_wakeup(Engine, "WAKEUP")
        installed = Engine._process_input_queue
        MODULE.install_engine_cancel_wakeup(Engine, "WAKEUP")
        self.assertIs(installed, Engine._process_input_queue)
        return engine

    def test_real_blocking_queue_wakes_without_new_business_request(self):
        progress = Progress()
        engine = self.make_engine(progress)
        returned = []
        thread = threading.Thread(target=lambda: returned.append(engine._process_input_queue()))
        thread.start()
        self.assertTrue(engine.entered.wait(1))
        progress.begin("R", "G")
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(returned, [("WAKEUP", None)])
        self.assertEqual(engine.original_calls, 1)
        self.assertTrue(progress.has_pending(), "A wake is not a completion proof")

    def test_constructor_time_pending_is_bound_before_first_wait(self):
        progress = Progress()
        progress.begin("R", "G")
        engine = self.make_engine(progress)
        self.assertEqual(engine._process_input_queue(), ("WAKEUP", None))
        self.assertTrue(progress.has_pending())

    def test_decode_child_is_not_bound(self):
        progress = Progress()
        engine = self.make_engine(progress, consumer=True)
        engine.input_queue.put_nowait("sentinel")
        self.assertEqual(engine._process_input_queue(), "sentinel")
        progress.begin("R", "G")
        self.assertTrue(engine.input_queue.empty())

    def test_callback_does_not_retain_engine(self):
        progress = Progress()
        engine = self.make_engine(progress)
        engine.input_queue.put_nowait("sentinel")
        engine._process_input_queue()
        ref = weakref.ref(engine)
        del engine
        gc.collect()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main(verbosity=2)
