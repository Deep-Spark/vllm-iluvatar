# SPDX-License-Identifier: Apache-2.0
"""CPU tests for connector-validated even remote-prefix replay alignment."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import (
    patch_prefix_replay_alignment as patch,
)
from vllm_iluvatar.ilu_patch.general_patches.patch_prefix_replay_alignment import (
    _HELPER,
    _NEW_CALL,
    _OLD_BLOCK,
)


def _request(**params):
    return SimpleNamespace(num_prompt_tokens=130, kv_transfer_params=params or None)


def _scheduler(**overrides):
    values = {
        "prefix_replay_tokens": 128,
        "block_size": 64,
        "connector": SimpleNamespace(prefix_replay_alignment=2),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_patch_schedule_source_replaces_block_rounding():
    source = "def schedule(self):\n" + _OLD_BLOCK + "                    pass\n"
    patched = patch.patch_schedule_source(source)
    assert _OLD_BLOCK not in patched
    assert _NEW_CALL in patched
    assert "ext_tokens % self.block_size" not in patched
    assert patched.endswith("                    pass\n")
    assert patch.patch_schedule_source(patched) == patched


def test_patch_schedule_source_leaves_unrelated_snippets():
    source = "def schedule(self):\n    if num_new_tokens == 0:\n        return\n"
    assert patch.patch_schedule_source(source) == source


def test_no_replay_window_is_unchanged():
    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(prefix_replay_tokens=0),
        _request(do_remote_prefill=True, transfer_id="t"),
        130,
        True,
        0,
    )
    assert (ext, async_load) == (130, True)


def test_even_alignment_keeps_complete_remote_prompt():
    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(),
        _request(do_remote_prefill=True, remote_request_id="r"),
        130,
        True,
        0,
    )
    assert (ext, async_load) == (130, True)


def test_block_alignment_without_connector_validation():
    cases = [
        _scheduler(connector=SimpleNamespace()),
        _scheduler(connector=SimpleNamespace(prefix_replay_alignment=None)),
        _scheduler(connector=SimpleNamespace(prefix_replay_alignment=64)),
        _scheduler(prefix_replay_tokens=64),
        _scheduler(),
    ]
    requests = [
        _request(do_remote_prefill=True, transfer_id="t"),
        _request(do_remote_prefill=True, transfer_id="t"),
        _request(do_remote_prefill=True, transfer_id="t"),
        _request(do_remote_prefill=True, transfer_id="t"),
        _request(do_remote_prefill=True),
    ]
    locals_ = [0, 0, 0, 0, 0]
    for scheduler, request, local in zip(cases, requests, locals_):
        ext, async_load = patch.align_prefix_replay_ext_tokens(
            scheduler, request, 130, True, local
        )
        assert (ext, async_load) == (128, True)

    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(),
        _request(do_remote_prefill=True, transfer_id="t"),
        130,
        True,
        2,
    )
    assert (ext, async_load) == (128, True)

    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(),
        _request(do_remote_prefill=True, transfer_id="t"),
        120,
        True,
        0,
    )
    assert (ext, async_load) == (64, True)


def test_zero_tokens_after_rounding_clears_async_load():
    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(connector=SimpleNamespace()),
        _request(),
        32,
        True,
        0,
    )
    assert (ext, async_load) == (0, False)


def test_sync_hit_still_uses_block_alignment():
    ext, async_load = patch.align_prefix_replay_ext_tokens(
        _scheduler(),
        _request(do_remote_prefill=True, transfer_id="t"),
        130,
        False,
        0,
    )
    assert (ext, async_load) == (128, False)


def test_register_drives_the_schedule_rewrite():
    """Registering drives the shared rewrite, whatever the discovery order."""
    from vllm.v1.core.sched import scheduler as sched_mod

    patch.register()

    source = sched_mod.Scheduler.schedule._ilu_pp_spec_schedule_source
    assert _HELPER in source
    assert "ext_tokens -= ext_tokens % self.block_size" not in source
    assert getattr(sched_mod, _HELPER) is patch.align_prefix_replay_ext_tokens
