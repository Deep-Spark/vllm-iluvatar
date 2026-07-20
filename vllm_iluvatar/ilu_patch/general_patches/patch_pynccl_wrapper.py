# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar PyNCCL optional symmetric-memory symbols.

Corresponds to:
    vllm/distributed/device_communicators/pynccl_wrapper.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

This patch is a pure capability-table adjustment with no copied upstream method
body. Iluvatar NCCL builds may not expose the symmetric-memory window symbols,
so they are removed from ``NCCLLibrary.exported_functions`` before
``NCCLLibrary`` loads the shared library.

Target   : vllm.distributed.device_communicators.pynccl_wrapper.
           NCCLLibrary.exported_functions
Baseline : main_dcfebf93f4e
Why      : ``ncclCommWindowRegister`` / ``ncclCommWindowDeregister`` are only
           needed for NCCL symmetric memory and may be absent in vendor NCCL
           builds. Missing them should not disable the ordinary PyNCCL API.
How      : filter those optional symbols and force the corresponding symmetric
           memory env switches off for Iluvatar.
Scope    : PyNCCL symmetric-memory capability on Iluvatar only.
Remove   : when Iluvatar NCCL supports the window registration APIs.
"""

from __future__ import annotations

import importlib
import os

from vllm_iluvatar.ilu_patch import general_patch as p

_OPTIONAL_NCCL_SYMM_MEM_FUNCS = {
    "ncclCommWindowRegister",
    "ncclCommWindowDeregister",
}


def register() -> None:
    """Register Iluvatar PyNCCL symmetric-memory capability override."""
    p.register_once(
        __name__,
        _apply,
        detail="PyNCCL symmetric-memory symbol filter",
    )


def _apply() -> None:
    community = importlib.import_module(
        "vllm.distributed.device_communicators.pynccl_wrapper"
    )

    # [ILU override] NCCLLibrary.exported_functions - delta: do not require
    # NCCL symmetric-memory window APIs on Iluvatar vendor NCCL builds.
    community.NCCLLibrary.exported_functions = [
        func
        for func in community.NCCLLibrary.exported_functions
        if func.name not in _OPTIONAL_NCCL_SYMM_MEM_FUNCS
    ]

    # Keep later dynamic reads from vllm.envs on the non-symmetric-memory path.
    os.environ["VLLM_USE_NCCL_SYMM_MEM"] = "0"
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
