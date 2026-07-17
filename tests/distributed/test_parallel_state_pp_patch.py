# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar isend/irecv_tensor_dict PP patch.

Verifies that on ivcore30 the patch replaces isend→send and irecv→recv
(blocking), while on ivcore11 the community async path is preserved.
"""

from __future__ import annotations

import sys
from collections import namedtuple
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm_iluvatar.distributed import parallel_state as impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class FakeTensor:
    """Minimal stand-in for torch.Tensor used by isend/irecv bodies."""

    def __init__(self, numel_val: int = 4, is_cpu: bool = False):
        self._numel = numel_val
        self.is_cpu = is_cpu
        self.is_cuda = not is_cpu
        self.size = (numel_val,)
        self.dtype = "float32"
        self.device = "cpu" if is_cpu else "cuda:0"
        self.shape = (numel_val,)

    def numel(self):
        return self._numel

    def reshape(self, *args):
        return self

    def __getitem__(self, idx):
        return self

    def record_stream(self, stream):
        pass


class FakeGroupCoordinator:
    """Minimal GroupCoordinator stub that records which comm primitives
    (send/isend/recv/irecv) were called."""

    def __init__(self, world_size=2, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.ranks = list(range(world_size))
        self.device_group = "device_group"
        self.cpu_group = "cpu_group"
        self.use_cpu_custom_send_recv = False
        self.device_communicator = None
        self.calls: list[dict] = []

    def send_object(self, obj, dst):
        self.calls.append({"op": "send_object", "dst": dst})

    def recv_object(self, src):
        self.calls.append({"op": "recv_object", "src": src})
        return [("hidden", TensorMetadata("cuda:0", "float32", (4,)))]

    def _should_use_all_gather(self, key, numel, ag_group, ag_tensors):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    """Disable fingerprint check — fake classes won't match real baselines."""
    monkeypatch.setenv("VLLM_ILU_PATCH_CHECK_FINGERPRINT", "0")


@pytest.fixture()
def _set_ivcore30(monkeypatch):
    """Force is_ivcore30() to return True."""
    monkeypatch.setattr(impl, "_is_ivcore30_fn", lambda: True)


@pytest.fixture()
def _set_ivcore11(monkeypatch):
    """Force is_ivcore30() to return False."""
    monkeypatch.setattr(impl, "_is_ivcore30_fn", lambda: False)


@pytest.fixture()
def torch_mock():
    """Build a mock torch module with send/isend/recv/irecv tracking."""
    mock_torch = MagicMock()
    call_log: list[str] = []

    def _send(tensor, dst, group):
        call_log.append("send")

    def _isend(tensor, dst, group):
        call_log.append("isend")
        return MagicMock()  # fake handle

    def _recv(tensor, src, group):
        call_log.append("recv")

    def _irecv(tensor, src, group):
        call_log.append("irecv")
        return MagicMock()  # fake handle

    mock_torch.distributed.send = _send
    mock_torch.distributed.isend = _isend
    mock_torch.distributed.recv = _recv
    mock_torch.distributed.irecv = _irecv
    mock_torch.distributed.is_initialized.return_value = True
    mock_torch.cuda.current_stream.return_value = MagicMock()
    mock_torch.empty = lambda size, dtype, device: FakeTensor(numel_val=size[0])
    mock_torch.Tensor = FakeTensor

    mock_torch._call_log = call_log
    return mock_torch


@pytest.fixture(autouse=True)
def _bind_globals(torch_mock):
    """Bind the mock globals onto the impl module (mimics _apply)."""
    orig_torch = impl.torch
    orig_tm = impl.TensorMetadata
    orig_split = impl._split_tensor_dict
    orig_handle = impl.Handle
    orig_callable = impl.Callable

    impl.torch = torch_mock
    impl.TensorMetadata = TensorMetadata

    def _split(td):
        meta = []
        tensors = []
        for k, v in td.items():
            if isinstance(v, FakeTensor):
                meta.append((k, TensorMetadata(v.device, v.dtype, v.size)))
                tensors.append(v)
            else:
                meta.append((k, v))
        return meta, tensors

    impl._split_tensor_dict = _split

    class _Handle:
        def is_completed(self):
            return True
        def wait(self):
            pass

    impl.Handle = _Handle
    import collections.abc
    impl.Callable = collections.abc.Callable

    yield

    impl.torch = orig_torch
    impl.TensorMetadata = orig_tm
    impl._split_tensor_dict = orig_split
    impl.Handle = orig_handle
    impl.Callable = orig_callable


# ---------------------------------------------------------------------------
# Tests: isend_tensor_dict
# ---------------------------------------------------------------------------

class TestIsendTensorDict:

    def test_ivcore30_uses_blocking_send(self, _set_ivcore30, torch_mock):
        """On ivcore30, isend_tensor_dict must call torch.distributed.send."""
        gc = FakeGroupCoordinator(world_size=2, rank=0)
        td = {"hidden": FakeTensor(4)}

        orig_isend = lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError("original isend should not be called")
        )
        wrapped = impl._make_isend_tensor_dict(orig_isend)
        handles = wrapped(gc, td)

        assert torch_mock._call_log == ["send"], (
            f"Expected blocking send, got {torch_mock._call_log}"
        )
        assert handles == [], "ivcore30 should return empty handles"

    def test_ivcore11_uses_async_isend(self, _set_ivcore11, torch_mock):
        """On ivcore11, isend_tensor_dict must delegate to original."""
        gc = FakeGroupCoordinator(world_size=2, rank=0)
        td = {"hidden": FakeTensor(4)}

        delegate_called = []

        def orig_isend(self, tensor_dict, dst=None,
                       all_gather_group=None, all_gather_tensors=None):
            delegate_called.append(True)
            return ["fake_handle"]

        wrapped = impl._make_isend_tensor_dict(orig_isend)
        handles = wrapped(gc, td)

        assert delegate_called, "ivcore11 should delegate to community isend"
        assert handles == ["fake_handle"]
        assert torch_mock._call_log == [], (
            "ivcore11 should NOT call torch.distributed.send"
        )

    def test_ivcore30_empty_tensor_skipped(self, _set_ivcore30, torch_mock):
        """Zero-numel tensors should be skipped (no send call)."""
        gc = FakeGroupCoordinator(world_size=2, rank=0)
        td = {"empty": FakeTensor(numel_val=0)}

        wrapped = impl._make_isend_tensor_dict(lambda *a, **kw: [])
        handles = wrapped(gc, td)

        assert torch_mock._call_log == [], "empty tensor should not trigger send"
        assert handles == []


# ---------------------------------------------------------------------------
# Tests: irecv_tensor_dict
# ---------------------------------------------------------------------------

class TestIrecvTensorDict:

    def test_ivcore30_uses_blocking_recv(self, _set_ivcore30, torch_mock):
        """On ivcore30, irecv_tensor_dict must call torch.distributed.recv."""
        gc = FakeGroupCoordinator(world_size=2, rank=1)

        orig_irecv = lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError("original irecv should not be called")
        )
        wrapped = impl._make_irecv_tensor_dict(orig_irecv)
        tensor_dict, handles, postprocess = wrapped(gc)

        assert torch_mock._call_log == ["recv"], (
            f"Expected blocking recv, got {torch_mock._call_log}"
        )
        assert handles == [], "ivcore30 should return empty handles"
        assert "hidden" in tensor_dict

    def test_ivcore11_uses_async_irecv(self, _set_ivcore11, torch_mock):
        """On ivcore11, irecv_tensor_dict must delegate to original."""
        gc = FakeGroupCoordinator(world_size=2, rank=1)

        delegate_called = []

        def orig_irecv(self, src=None,
                       all_gather_group=None, all_gather_tensors=None):
            delegate_called.append(True)
            return {"hidden": FakeTensor(4)}, ["fake_handle"], []

        wrapped = impl._make_irecv_tensor_dict(orig_irecv)
        tensor_dict, handles, postprocess = wrapped(gc)

        assert delegate_called, "ivcore11 should delegate to community irecv"
        assert handles == ["fake_handle"]
        assert torch_mock._call_log == [], (
            "ivcore11 should NOT call torch.distributed.recv"
        )

    def test_ivcore30_world_size_1_returns_none(self, _set_ivcore30, torch_mock):
        """Single-GPU should short-circuit and return (None, [], [])."""
        gc = FakeGroupCoordinator(world_size=1, rank=0)

        wrapped = impl._make_irecv_tensor_dict(lambda *a, **kw: (None, [], []))
        tensor_dict, handles, postprocess = wrapped(gc)

        assert tensor_dict is None
        assert handles == []
        assert postprocess == []


# ---------------------------------------------------------------------------
# Tests: arch caching
# ---------------------------------------------------------------------------

class TestArchCaching:

    def test_is_ivcore30_caches_result(self, monkeypatch):
        """ilu_util.arch.is_ivcore30 should cache after first call."""
        import vllm_iluvatar.ilu_util.arch as arch_mod
        monkeypatch.setattr(arch_mod, "_resolved", False)
        monkeypatch.setattr(arch_mod, "_cuda_arch", None)

        ixformer_cuda = MagicMock()
        ixformer_cuda.cuda_arch = "ivcore30"
        ixformer_mod = ModuleType("ixformer")
        ixformer_mod.cuda = ixformer_cuda

        monkeypatch.setitem(sys.modules, "ixformer", ixformer_mod)

        assert arch_mod.is_ivcore30() is True
        assert arch_mod._resolved is True

        # Second call should use cache, not re-import.
        monkeypatch.delitem(sys.modules, "ixformer")
        assert arch_mod.is_ivcore30() is True

    def test_ivcore11_returns_false(self, monkeypatch):
        """ivcore11 arch should make is_ivcore30 return False."""
        import vllm_iluvatar.ilu_util.arch as arch_mod
        monkeypatch.setattr(arch_mod, "_resolved", False)
        monkeypatch.setattr(arch_mod, "_cuda_arch", None)

        ixformer_cuda = MagicMock()
        ixformer_cuda.cuda_arch = "ivcore11"
        ixformer_mod = ModuleType("ixformer")
        ixformer_mod.cuda = ixformer_cuda

        monkeypatch.setitem(sys.modules, "ixformer", ixformer_mod)

        assert arch_mod.is_ivcore30() is False
        assert arch_mod.cuda_arch() == "ivcore11"


# ---------------------------------------------------------------------------
# Tests: upper-layer compatibility
# ---------------------------------------------------------------------------

class TestUpperLayerCompat:
    """Verify that ivcore30 sync mode is compatible with the community
    send_tensor_dict / recv_tensor_dict / gpu_worker patterns."""

    def test_send_tensor_dict_compat(self, _set_ivcore30, torch_mock):
        """send_tensor_dict (isend + handle.wait loop) works with empty
        handles returned by the ivcore30 sync path."""
        gc = FakeGroupCoordinator(world_size=2, rank=0)
        td = {"hidden": FakeTensor(4)}

        wrapped_isend = impl._make_isend_tensor_dict(
            lambda *a, **kw: [MagicMock()]
        )
        gc.isend_tensor_dict = lambda *a, **kw: wrapped_isend(gc, *a, **kw)

        handles = gc.isend_tensor_dict(td)
        for h in handles:
            h.wait()  # empty list → no-op, should not raise

        assert torch_mock._call_log == ["send"]

    def test_pp_send_work_compat(self, _set_ivcore30, torch_mock):
        """gpu_worker stores isend handles in _pp_send_work; next iteration
        checks `if self._pp_send_work:`.  Empty list should be falsy."""
        gc = FakeGroupCoordinator(world_size=2, rank=0)
        td = {"hidden": FakeTensor(4)}

        wrapped_isend = impl._make_isend_tensor_dict(
            lambda *a, **kw: [MagicMock()]
        )
        pp_send_work = wrapped_isend(gc, td)

        assert not pp_send_work, (
            "empty handles list should be falsy for `if self._pp_send_work:`"
        )

    def test_recv_postprocess_still_runs(self, _set_ivcore30, torch_mock):
        """Even with empty handles, irecv postprocess callbacks must still
        be collected so that recv_tensor_dict / AsyncIntermediateTensors can
        invoke them."""
        gc = FakeGroupCoordinator(world_size=2, rank=1)

        # Override recv_object to return a tensor that triggers all_gather path
        def recv_object_ag(src):
            gc.calls.append({"op": "recv_object", "src": src})
            return [("hidden", TensorMetadata("cuda:0", "float32", (8,)))]

        gc.recv_object = recv_object_ag

        def should_ag(key, numel, ag_group, ag_tensors):
            return ag_group is not None

        gc._should_use_all_gather = should_ag

        mock_ag_group = MagicMock()
        mock_ag_group.world_size = 2
        mock_ag_group.rank_in_group = 0

        wrapped_irecv = impl._make_irecv_tensor_dict(
            lambda *a, **kw: ({}, [], [])
        )
        tensor_dict, handles, postprocess = wrapped_irecv(
            gc, all_gather_group=mock_ag_group
        )

        assert handles == [], "handles should be empty on ivcore30"
        assert len(postprocess) == 1, (
            "postprocess callback for all_gather should still be collected"
        )
