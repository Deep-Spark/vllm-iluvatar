# SPDX-License-Identifier: Apache-2.0
"""Platform, ordinary groups and stateless groups select the same communicator."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_iluvatar.backends import (
    communicator_class_path,
    resolve_device_communicator_cls,
)


# Compare both actual class paths, including the stateless coordinator's construction.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_each_entry_selects_the_same_class(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    platform_module = (
        "vllm_iluvatar.platforms.ilu_platform"
        if mode == "baseline"
        else "vllm_iluvatar.backends.ixaite.platform"
    )
    class_name = "IluPlatform" if mode == "baseline" else "IxaitePlatform"
    platform = getattr(importlib.import_module(platform_module), class_name)
    assert platform.get_device_communicator_cls() == communicator_class_path()
    selected = resolve_device_communicator_cls()
    assert f"{selected.__module__}.{selected.__name__}" == communicator_class_path()
    coordinator = importlib.import_module("vllm.distributed.stateless_coordinator")
    if mode == "ixaite":
        from vllm_iluvatar.backends.ixaite.register import _apply_stateless as apply
    else:
        from vllm_iluvatar.ilu_patch.general_patches.patch_stateless_coordinator import (
            _apply as apply,
        )
    monkeypatch.setattr(coordinator, "CudaCommunicator", object)
    apply()
    assert coordinator.CudaCommunicator is selected


# Both installers keep the original guard when the upstream symbol is absent.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_stateless_patch_skips_missing_communicator(monkeypatch, mode):
    coordinator = importlib.import_module("vllm.distributed.stateless_coordinator")
    monkeypatch.delattr(coordinator, "CudaCommunicator")
    if mode == "ixaite":
        from vllm_iluvatar.backends.ixaite.register import _apply_stateless as apply
    else:
        from vllm_iluvatar.ilu_patch.general_patches.patch_stateless_coordinator import (
            _apply as apply,
        )
    apply()
    assert not hasattr(coordinator, "CudaCommunicator")


# An unavailable library cannot silently construct a vendor communicator.
def test_unavailable_collectives_fail_before_construction(monkeypatch):
    from vllm_iluvatar.backends.ixaite import distributed, runtime

    monkeypatch.setattr(runtime, "load_ixaite_python_op", lambda *_: (None, "missing"))
    with pytest.raises(RuntimeError, match="missing"):
        distributed.IluIxaiteCommunicator._collective_backend()


# Bind the entire library, which variable-length operations must also use.
def test_collective_library_is_bound_once(monkeypatch):
    from vllm_iluvatar.backends.ixaite import distributed, runtime

    fake = ModuleType("ixaite.ops.distributed")
    ops = ModuleType("ixaite.ops")
    ixaite = ModuleType("ixaite")
    ixaite.ops = ops
    ops.distributed = fake
    monkeypatch.setitem(sys.modules, "ixaite", ixaite)
    monkeypatch.setitem(sys.modules, "ixaite.ops", ops)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(
        runtime, "load_ixaite_python_op", lambda *_: (object(), "ready")
    )
    assert distributed.IluIxaiteCommunicator._collective_backend() is fake


# A library's optional vendor delegation conflicts with strict process selection.
def test_vendor_large_all_reduce_is_rejected(monkeypatch):
    from vllm_iluvatar.backends.ixaite.distributed import IluIxaiteCommunicator

    monkeypatch.setenv("IXAITE_LARGE_AR_BACKEND", "ixformer")
    with pytest.raises(ValueError, match="IXAITE_LARGE_AR_BACKEND"):
        IluIxaiteCommunicator._collective_backend()


# EP implementations still requiring ixformer fail before initializing groups.
@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_high_throughput"])
def test_vendor_ep_is_rejected_before_groups(monkeypatch, backend):
    import vllm.config

    from vllm_iluvatar.backends.ixaite.distributed import IluIxaiteCommunicator

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(all2all_backend=backend)
        ),
    )
    with pytest.raises(ValueError, match="EP backend"):
        IluIxaiteCommunicator(None, unique_name="ep:0", use_all2all=True)
