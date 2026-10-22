"""Session cleanup must never remove unknown or live sibling directories."""

import json

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
