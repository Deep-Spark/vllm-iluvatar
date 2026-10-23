"""Session cleanup must never remove unknown or live sibling directories."""

import json
import os

from vllm_iluvatar.v1.worker.mm_replay.lifecycle import claim_session, owner_identity


def test_reclaim_only_explicit_dead_owners(tmp_path):
    current = tmp_path / ("a" * 64)
    current.mkdir()
    unknown = tmp_path / ("b" * 64)
    unknown.mkdir()
    (unknown / "keep").write_text("not our session")
    live = tmp_path / ("c" * 64)
    live.mkdir()
    (live / "owner.json").write_text(json.dumps(owner_identity()))
    dead = tmp_path / ("d" * 64)
    dead.mkdir()
    owner = owner_identity()
    owner["start_ticks"] -= 1
    (dead / "owner.json").write_text(json.dumps(owner))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "owner.json").write_text(json.dumps(owner))
    claim_session(current)
    assert (
        current.exists() and unknown.exists() and live.exists() and unrelated.exists()
    )
    assert not dead.exists()


def test_foreign_pid_namespace_cannot_be_probed_with_local_proc(tmp_path):
    current = tmp_path / ("a" * 64)
    current.mkdir()
    foreign = tmp_path / ("b" * 64)
    foreign.mkdir()
    owner = owner_identity()
    # In --ipc=host deployments a foreign container can use the same PID for
    # a different process. A local start-time mismatch proves nothing there.
    owner["pid_namespace"] = "pid:[foreign-container]"
    owner["start_ticks"] -= 1
    (foreign / "owner.json").write_text(json.dumps(owner))
    claim_session(current)
    assert foreign.is_dir()


def test_legacy_owner_without_namespace_is_preserved(tmp_path):
    current = tmp_path / ("a" * 64)
    current.mkdir()
    legacy = tmp_path / ("b" * 64)
    legacy.mkdir()
    owner = owner_identity()
    owner["kind"] = "vllm_iluvatar_mm_replay_v1"
    owner.pop("pid_namespace", None)
    owner["start_ticks"] -= 1
    (legacy / "owner.json").write_text(json.dumps(owner))
    claim_session(current)
    assert legacy.is_dir()


def test_ancestor_proc_mount_cannot_prove_owner_death(tmp_path, monkeypatch):
    current = tmp_path / ("a" * 64)
    current.mkdir()
    sibling = tmp_path / ("b" * 64)
    sibling.mkdir()
    owner = owner_identity()
    owner["start_ticks"] -= 1
    (sibling / "owner.json").write_text(json.dumps(owner))
    readlink = os.readlink
    monkeypatch.setattr(
        os,
        "readlink",
        lambda p: "foreign-proc-pid" if p == "/proc/self" else readlink(p),
    )
    claim_session(current)
    assert sibling.is_dir()


def test_non_object_owner_marker_is_unknown_not_a_startup_error(tmp_path):
    current = tmp_path / ("a" * 64)
    current.mkdir()
    for index, value in enumerate([None, [], 3, "old-format"]):
        sibling = tmp_path / (str(index) * 64)
        sibling.mkdir()
        (sibling / "owner.json").write_text(json.dumps(value))
    claim_session(current)
    assert len(list(tmp_path.iterdir())) == 5
