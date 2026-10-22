"""P keeps images alive while a long prefill precedes all decoder heartbeats."""

import threading
import time

from vllm_iluvatar.v1.worker.mm_replay.archive import SourceArchive
from vllm_iluvatar.v1.worker.mm_replay.scheduler import ReplayScheduler


def test_live_producer_owns_lease_until_prefill_completion(tmp_path):
    archive = SourceArchive(
        tmp_path / "source",
        "P",
        "run",
        budget_bytes=4096,
        hidden_size=8,
        lease_seconds=0.15,
    )
    archive.admit(
        "long",
        "gen",
        "a" * 64,
        1048576,
        [dict(key="image", start=1048000, rows=20, input=b"pixels", fields=[])],
    )
    scheduler = ReplayScheduler.__new__(ReplayScheduler)
    scheduler.root = tmp_path
    scheduler.archive = archive
    scheduler._stop = threading.Event()
    scheduler._maintenance_period = 0.025
    scheduler.admitted = {"long"}
    scheduler.records = {"long": dict(generation="gen", input_digest="a" * 64)}
    thread = threading.Thread(target=scheduler._maintain)
    thread.start()
    try:
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            time.sleep(0.025)
            assert archive.stats()["requests"] == 1
        # Normal P completion hands ownership to the native D lease; with no
        # D heartbeat, the ordinary timeout must still expire it.
        scheduler.admitted.clear()
        scheduler.records.clear()
        deadline = time.monotonic() + 1
        while archive.stats()["requests"]:
            assert time.monotonic() < deadline
            time.sleep(0.025)
    finally:
        scheduler._stop.set()
        thread.join(3)
    assert not thread.is_alive()


def test_stale_maintenance_cannot_renew_or_resurrect_new_generation(tmp_path):
    archive = SourceArchive(
        tmp_path / "source",
        "P",
        "run",
        budget_bytes=4096,
        hidden_size=8,
        lease_seconds=0.12,
    )
    archive.admit(
        "same-id",
        "new",
        "b" * 64,
        129,
        [dict(key="image", start=3, rows=20, input=b"pixels", fields=[])],
    )
    scheduler = ReplayScheduler.__new__(ReplayScheduler)
    scheduler.root = tmp_path
    scheduler.archive = archive
    scheduler._stop = threading.Event()
    scheduler._maintenance_period = 0.02
    scheduler.admitted = {"same-id"}
    scheduler.records = {"same-id": dict(generation="old", input_digest="a" * 64)}
    thread = threading.Thread(target=scheduler._maintain)
    thread.start()
    try:
        time.sleep(0.25)
        assert thread.is_alive()
        assert archive.stats()["requests"] == 0
    finally:
        scheduler._stop.set()
        thread.join(3)
