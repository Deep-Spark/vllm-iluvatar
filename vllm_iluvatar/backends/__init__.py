# SPDX-License-Identifier: Apache-2.0
"""The two process-wide backend choices; optional kernels are loaded lazily."""

from __future__ import annotations

import importlib
import os

_PROVIDER_ENV = "VLLM_ILU_KERNEL_PROVIDER"


# Validate launch controls before importing or registering either implementation.
def provider_mode() -> str:
    mode = os.environ.get(_PROVIDER_ENV, "baseline").strip().lower()
    if mode not in {"baseline", "ixaite"}:
        raise ValueError(f"{_PROVIDER_ENV} must be baseline|ixaite, got {mode!r}")
    if os.environ.get("VLLM_ILU_IXAITE_OP_ALLOWLIST", "").strip():
        raise ValueError("VLLM_ILU_IXAITE_OP_ALLOWLIST cannot mix the two backends")
    return mode


# Dotted paths keep platform discovery free of eager worker imports.
def worker_class_path() -> str:
    if provider_mode() == "ixaite":
        return "vllm_iluvatar.backends.ixaite.worker.IxaiteWorker"
    return "vllm_iluvatar.v1.worker.gpu_worker.IluWorker"


# Both ordinary and stateless groups must resolve the same communicator.
def communicator_class_path() -> str:
    if provider_mode() == "ixaite":
        return "vllm_iluvatar.backends.ixaite.distributed.IluIxaiteCommunicator"
    return (
        "vllm_iluvatar.distributed.device_communicators."
        "ilu_communicator.IluCommunicator"
    )


# Stateless groups need the actual class; importing it here keeps discovery lazy.
def resolve_device_communicator_cls() -> type:
    module_path, _, class_name = communicator_class_path().rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)
