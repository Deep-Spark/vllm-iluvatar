# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``IluCommunicator.batch_isend_irecv``.

The community ``CudaCommunicator.batch_isend_irecv`` is pynccl-only and raises
``"No PyNCCL communicator found"`` on Iluvatar (pynccl disabled). The override
must route stateless standby groups through the TCPStore-backed
``StatelessProcessGroup`` (mirroring the plugin's ``broadcast`` override) and
normal groups through ixfd.

These tests construct the communicator via ``object.__new__`` (bypassing the
``__init__`` that needs a real device group) and assert dispatch only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.distributed.device_communicators import ilu_communicator


def _make_comm(**attrs):
    comm = object.__new__(ilu_communicator.IluCommunicator)
    for name, value in attrs.items():
        setattr(comm, name, value)
    return comm


def _op(is_send: bool, tensor, peer: int):
    return SimpleNamespace(
        op=torch.distributed.isend if is_send else torch.distributed.irecv,
        tensor=tensor,
        group_peer=peer,
    )


class _FakeStore:
    def __init__(self):
        self.sends = []
        self.recvs = []

    def send(self, tensor, dst):
        self.sends.append((tensor, dst))

    def recv(self, tensor, src):
        self.recvs.append((tensor, src))
        return tensor


def test_batch_isend_irecv_stateless_uses_tcp_store_group():
    store = _FakeStore()
    comm = _make_comm(
        is_stateless_group=True,
        tcp_store_group=store,
        use_vllm_comm=False,
        ixfd_group=None,
        ranks=[0, 1, 2, 3],
    )
    t_send = SimpleNamespace(data_ptr=lambda: 1)
    t_recv = SimpleNamespace(data_ptr=lambda: 2)

    comm.batch_isend_irecv([_op(True, t_send, 2), _op(False, t_recv, 1)])

    assert store.sends == [(t_send, 2)]
    assert store.recvs == [(t_recv, 1)]


def test_batch_isend_irecv_stateless_requires_tcp_store_group():
    comm = _make_comm(
        is_stateless_group=True,
        tcp_store_group=None,
        use_vllm_comm=False,
        ixfd_group=None,
        ranks=[0, 1, 2, 3],
    )
    with pytest.raises(AssertionError):
        comm.batch_isend_irecv([_op(True, SimpleNamespace(), 0)])


def test_batch_isend_irecv_ixfd_uses_global_ranks(monkeypatch):
    ixfd_calls = []

    class _FakeIxfd:
        @staticmethod
        def isend(tensor, dst, group):
            ixfd_calls.append(("isend", tensor, dst, group))

        @staticmethod
        def irecv(tensor, src, group):
            ixfd_calls.append(("irecv", tensor, src, group))

    monkeypatch.setattr(ilu_communicator, "ixfd", _FakeIxfd)

    comm = _make_comm(
        is_stateless_group=False,
        tcp_store_group=None,
        use_vllm_comm=True,
        ixfd_group="GROUP",
        ranks=[10, 11, 12, 13],
    )
    t_send = SimpleNamespace(data_ptr=lambda: 1)
    t_recv = SimpleNamespace(data_ptr=lambda: 2)

    comm.batch_isend_irecv([_op(True, t_send, 2), _op(False, t_recv, 1)])

    assert ixfd_calls == [
        ("isend", t_send, 12, "GROUP"),
        ("irecv", t_recv, 11, "GROUP"),
    ]


def test_batch_isend_irecv_no_ixfd_raises():
    comm = _make_comm(
        is_stateless_group=False,
        tcp_store_group=None,
        use_vllm_comm=False,
        ixfd_group=None,
        ranks=[10, 11, 12, 13],
    )
    with pytest.raises(ValueError, match="No ixfd communicator"):
        comm.batch_isend_irecv([_op(True, SimpleNamespace(), 0)])


def test_batch_isend_irecv_unsupported_op_raises():
    store = _FakeStore()
    comm = _make_comm(
        is_stateless_group=True,
        tcp_store_group=store,
        use_vllm_comm=False,
        ixfd_group=None,
        ranks=[0, 1, 2, 3],
    )
    bad_op = SimpleNamespace(op=None, tensor=SimpleNamespace(), group_peer=0)
    with pytest.raises(ValueError, match="Unsupported p2p op"):
        comm.batch_isend_irecv([bad_op])
