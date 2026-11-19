# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Exercise orderly process-tree shutdown without a GPU or vLLM import."""

import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "iluvatar_smoke_shutdown", Path(__file__).with_name("runner.py")
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

WORKER = r"""
import pathlib, sys, time
root = pathlib.Path(sys.argv[1])
(root / 'worker-ready').touch()
while not (root / 'finish-worker').exists():
    time.sleep(0.01)
time.sleep(0.3)
(root / 'worker-cleaned').touch()
"""
PARENT = r"""
import pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
worker = subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root)])
def stop(signum, frame):
    (root / 'finish-worker').touch()
    if sys.argv[3] == 'wait':
        worker.wait(timeout=10)
    (root / 'parent-cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while not (root / 'worker-ready').exists():
    time.sleep(0.01)
(root / 'ready').touch()
while True:
    time.sleep(1)
"""


class ShutdownTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux process-group cleanup")
    def test_workers_finish_cleanup_even_if_parent_exits_first(self):
        for mode, cancel in (
            ("wait", False),
            ("exit-first", False),
            ("wait", True),
            ("exit-first", True),
        ):
            with (
                self.subTest(mode=mode, cancel=cancel),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                proc = subprocess.Popen(
                    [sys.executable, "-c", PARENT, str(root), WORKER, mode],
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 15
                    while not (root / "ready").exists():
                        self.assertIsNone(proc.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    if cancel:
                        control = smoke.RunControl()
                        control.register(proc)
                        control.cancel()
                    smoke.stop_process(proc)
                    self.assertEqual(proc.returncode, 0)
                    self.assertTrue((root / "parent-cleaned").exists())
                    self.assertTrue((root / "worker-cleaned").exists())
                    self.assertFalse(smoke.process_group_members(proc.pid))
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)

    @unittest.skipUnless(sys.platform == "linux", "Linux process-group cleanup")
    def test_sigkill_is_cleanup_failure_even_after_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,signal,sys,time;"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    "pathlib.Path(sys.argv[1]).touch();time.sleep(60)",
                    str(ready),
                ],
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while not ready.exists():
                    self.assertIsNone(proc.poll())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                wait = smoke.wait_process_group
                with (
                    mock.patch.object(
                        smoke,
                        "wait_process_group",
                        side_effect=lambda process, timeout: wait(process, 0.1),
                    ),
                    self.assertRaisesRegex(RuntimeError, "SIGKILL"),
                ):
                    smoke.stop_process(proc)
                self.assertEqual(proc.returncode, -signal.SIGKILL)
                self.assertFalse(smoke.process_group_members(proc.pid))
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
