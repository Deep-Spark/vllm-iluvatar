# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar PyNCCL wrapper patch."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_pynccl_wrapper


def _install_fake_pynccl_wrapper(monkeypatch):
    vllm = ModuleType("vllm")
    distributed = ModuleType("vllm.distributed")
    device_communicators = ModuleType("vllm.distributed.device_communicators")
    pynccl_wrapper = ModuleType(
        "vllm.distributed.device_communicators.pynccl_wrapper"
    )

    class NCCLLibrary:
        exported_functions = [
            SimpleNamespace(name="ncclGetVersion"),
            SimpleNamespace(name="ncclCommWindowRegister"),
            SimpleNamespace(name="ncclCommWindowDeregister"),
            SimpleNamespace(name="ncclAllReduce"),
        ]

    pynccl_wrapper.NCCLLibrary = NCCLLibrary
    vllm.distributed = distributed
    distributed.device_communicators = device_communicators
    device_communicators.pynccl_wrapper = pynccl_wrapper

    for module in (vllm, distributed, device_communicators, pynccl_wrapper):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    return NCCLLibrary


def test_pynccl_wrapper_patch_filters_only_symmetric_memory_symbols(
    monkeypatch,
):
    nccl_library = _install_fake_pynccl_wrapper(monkeypatch)

    patch_pynccl_wrapper._apply()

    names = [func.name for func in nccl_library.exported_functions]
    assert names == ["ncclGetVersion", "ncclAllReduce"]


def test_pynccl_wrapper_patch_forces_symmetric_memory_env_off(monkeypatch):
    _install_fake_pynccl_wrapper(monkeypatch)
    monkeypatch.setitem(os.environ, "VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setitem(os.environ, "VLLM_ALLREDUCE_USE_SYMM_MEM", "1")

    patch_pynccl_wrapper._apply()

    assert os.environ["VLLM_USE_NCCL_SYMM_MEM"] == "0"
    assert os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
