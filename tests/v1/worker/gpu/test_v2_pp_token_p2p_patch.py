# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the V2 PP sampled-token send/recv transport patch."""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_v2_pp_token_p2p as wiring
from vllm_iluvatar.v1.worker.gpu import pp_utils as pp_impl

_RECEIVE_BASELINE_SHA = (
    "e96d78229c4ca0e9bebd74b73570286702c66bef55432a94cec538406a40ed6a"
)
_BROADCAST_BASELINE_SHA = (
    "d8d5e43cd6d22eea6a629537e0780d22af537b6caadecb8cb85ce404e27448cd"
)
_BROADCAST_DRAFTS_BASELINE_SHA = (
    "e2cb75dabc5f87212ab1147549c8906e510354804245538dc0152ea45f93cda1"
)


@contextlib.contextmanager
def _patched(target: object, **attrs: Any):
    """Swap attributes on `target` for the duration of the block."""
    missing = object()
    saved = {name: getattr(target, name, missing) for name in attrs}
    for name, value in attrs.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(target, name)
            else:
                setattr(target, name, value)


@contextlib.contextmanager
def _fake_ixfd():
    """Install a stub ixformer.distributed that records send/recv calls."""
    calls: list[tuple[str, object, int, object]] = []
    mod = ModuleType("ixformer.distributed")
    mod.send = lambda t, dst, group: calls.append(("send", t, dst, group))
    mod.recv = lambda t, src, group: calls.append(("recv", t, src, group))
    # `import ixformer.distributed as ixfd` resolves through the package
    # attribute first, so the package has to be stubbed too when the real
    # ixformer is installed.
    pkg = ModuleType("ixformer")
    pkg.distributed = mod
    saved = {k: sys.modules.get(k) for k in ("ixformer", "ixformer.distributed")}
    sys.modules["ixformer"] = pkg
    sys.modules["ixformer.distributed"] = mod
    try:
        yield calls
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_baselines_match_declared_community_source():
    assert p.source_sha(wiring.RECEIVE_BASELINE) == _RECEIVE_BASELINE_SHA
    assert p.source_sha(wiring.BROADCAST_BASELINE) == _BROADCAST_BASELINE_SHA
    assert (
        p.source_sha(wiring.BROADCAST_DRAFTS_BASELINE)
        == _BROADCAST_DRAFTS_BASELINE_SHA
    )


def test_last_rank_sends_to_every_other_pp_rank():
    handler = SimpleNamespace(
        _ilu_token_p2p=("group", [3, 7, 11, 15]), is_last_rank=True, last_rank=15
    )
    tensor = object()
    with _fake_ixfd() as calls:
        pp_impl.ilu_token_broadcast(handler, tensor)
    assert calls == [
        ("send", tensor, 3, "group"),
        ("send", tensor, 7, "group"),
        ("send", tensor, 11, "group"),
    ]


def test_non_last_rank_receives_from_the_last_rank():
    handler = SimpleNamespace(
        _ilu_token_p2p=("group", [3, 7, 11, 15]), is_last_rank=False, last_rank=15
    )
    tensor = object()
    with _fake_ixfd() as calls:
        pp_impl.ilu_token_broadcast(handler, tensor)
    assert calls == [("recv", tensor, 15, "group")]


def test_falls_back_to_the_community_collective_without_an_ixfd_group():
    handler = SimpleNamespace(
        _ilu_token_p2p=None, is_last_rank=False, last_rank=15,
        broadcast_group="pp_broadcast",
    )
    seen: list[tuple[object, int, str]] = []
    dist = SimpleNamespace(
        broadcast=lambda t, src, group: seen.append((t, src, group))
    )
    tensor = object()
    with _patched(pp_impl.torch, distributed=dist):
        pp_impl.ilu_token_broadcast(handler, tensor)
    assert seen == [(tensor, 15, "pp_broadcast")]


def test_init_disables_the_transport_when_the_comm_is_missing():
    handler = SimpleNamespace()
    communicator = SimpleNamespace(ixfd_group=None, ranks=[0, 1])
    with _patched(
        pp_impl, get_pp_group=lambda: SimpleNamespace(
            device_communicator=communicator
        )
    ):
        pp_impl._init_token_p2p(handler)
    assert handler._ilu_token_p2p is None


def test_init_records_the_ixfd_group_and_ranks():
    handler = SimpleNamespace()
    communicator = SimpleNamespace(ixfd_group="g", ranks=[0, 4, 8, 12])
    with _patched(
        pp_impl, get_pp_group=lambda: SimpleNamespace(
            device_communicator=communicator
        )
    ):
        pp_impl._init_token_p2p(handler)
    assert handler._ilu_token_p2p == ("g", [0, 4, 8, 12])
