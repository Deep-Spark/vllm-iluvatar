# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar RayExecutorV2 worker environment setup.

Corresponds to: vllm/v1/executor/ray_executor_v2.py

Target   : RayWorkerProc.initialize_worker
Baseline : main_6e448d0
Why      : some Iluvatar communication paths need the platform device-control
           environment variable narrowed before WorkerProc initializes NCCL.
How      : wrap the worker initialization method and add the env var from
           ``assigned_physical_gpu_ids``. The Ray executor orchestration remains
           community-owned.
Scope    : RayExecutorV2 worker process initialization on Iluvatar.
Safety   : wrapper only changes the env dict passed to the live community
           method; no copied method body or fingerprint snapshot is needed.
Remove   : when community RayExecutorV2 exposes a platform hook for deriving
           worker device-control env vars from assigned physical GPU IDs.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="RayExecutorV2 worker device-control env setup",
    )


def _apply() -> None:
    import vllm.v1.executor.ray_executor_v2 as _m

    from vllm_iluvatar.v1.executor.ray_executor_v2 import wrap_initialize_worker

    original = _m.RayWorkerProc.initialize_worker
    wrapped = wrap_initialize_worker(original)
    # [ILU override] RayWorkerProc.initialize_worker - delta: inject the
    # platform device-control env before delegating to the community method.
    if inspect.signature(original) != inspect.signature(wrapped):
        raise RuntimeError(
            "ILU patch: RayWorkerProc.initialize_worker signature mismatch: "
            f"community {inspect.signature(original)} != Iluvatar "
            f"{inspect.signature(wrapped)}"
        )
    _m.RayWorkerProc.initialize_worker = wrapped
