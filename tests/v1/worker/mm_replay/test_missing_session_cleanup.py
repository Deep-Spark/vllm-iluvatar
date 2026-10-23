"""An image request's error cleanup must survive a missing local session."""

import threading
from collections import deque
from types import SimpleNamespace

from vllm_iluvatar.v1.worker.mm_replay.archive import SourceArchive
from vllm_iluvatar.v1.worker.mm_replay.receiver import RangeReceiver
from vllm_iluvatar.v1.worker.mm_replay.scheduler import ReplayScheduler


def test_missing_directory_cannot_turn_request_finish_into_engine_error(tmp_path):
    scheduler = ReplayScheduler.__new__(ReplayScheduler)
    scheduler.root = tmp_path / "already-removed"
    scheduler.session = "session"
    scheduler.engine_id = "decode"
    scheduler.archive = None
    scheduler.records = {"image": {"generation": "g"}}
    scheduler.pending = {}
    scheduler.admitted = set()
    scheduler.errors = {"image": "image replay failed"}
    scheduler.counts = dict(completed=0)
    receiver = RangeReceiver.__new__(RangeReceiver)
    receiver.root = scheduler.root / "decode"
    receiver.lock = threading.Lock()
    receiver.requests = {"image": {"pending": deque([(0, 64)])}}
    receiver.retiring = {}
    receiver.release_queue = deque()
    receiver.wakeup = threading.Event()
    scheduler.receiver = receiver

    scheduler.finished(SimpleNamespace(request_id="image"))

    assert not scheduler.records and not scheduler.errors
    assert not receiver.requests
    assert scheduler.counts["completed"] == 1
    assert len(receiver.release_queue) == 1
    assert receiver.wakeup.is_set()
    # Do not silently recreate an empty cache and pretend the data survived.
    assert not scheduler.root.exists()


def test_source_diagnostics_failure_does_not_kill_lease_thread(tmp_path, monkeypatch):
    source = SourceArchive(
        tmp_path / "source", "P", "run", budget_bytes=4096, hidden_size=8
    )
    scheduler = ReplayScheduler.__new__(ReplayScheduler)
    scheduler.root = tmp_path
    scheduler.archive = source
    scheduler._stop = threading.Event()
    scheduler._maintenance_period = 0.01
    scheduler.admitted = set()
    scheduler.records = {}
    retried = threading.Event()
    calls = 0

    def unavailable():
        nonlocal calls
        calls += 1
        if calls >= 2:
            retried.set()
        raise FileNotFoundError("source/archive.lock disappeared")

    monkeypatch.setattr(source, "stats", unavailable)
    thread = threading.Thread(target=scheduler._maintain)
    thread.start()
    try:
        assert retried.wait(1), "maintenance exited after its first I/O failure"
        assert thread.is_alive()
    finally:
        scheduler._stop.set()
        thread.join(2)
    assert not thread.is_alive()
