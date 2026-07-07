# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar RayExecutorV2 worker env patch."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_ray_executor_v2
from vllm_iluvatar.v1.executor.ray_executor_v2 import _device_control_env_vars


def test_device_control_env_vars_are_sorted():
    current_platform = SimpleNamespace(device_control_env_var="CUDA_VISIBLE_DEVICES")

    assert _device_control_env_vars([3, 1], current_platform) == {
        "CUDA_VISIBLE_DEVICES": "1,3"
    }


def test_ray_worker_initialize_worker_patch_adds_device_control_env(monkeypatch):
    calls = []

    class FakeRayWorkerProc:
        def initialize_worker(
            self,
            local_rank,
            env_vars,
            driver_env_vars=None,
            assigned_physical_gpu_ids=None,
        ):
            calls.append(
                {
                    "local_rank": local_rank,
                    "env_vars": env_vars,
                    "driver_env_vars": driver_env_vars,
                    "assigned_physical_gpu_ids": assigned_physical_gpu_ids,
                }
            )
            return "ok"

    ray_executor_module = ModuleType("vllm.v1.executor.ray_executor_v2")
    ray_executor_module.RayWorkerProc = FakeRayWorkerProc
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.executor", ModuleType("vllm.v1.executor"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.executor.ray_executor_v2",
        ray_executor_module,
    )

    platforms_module = ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(
        device_control_env_var="CUDA_VISIBLE_DEVICES"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    patch_ray_executor_v2._apply()

    original_env_vars = {"EXISTING": "1"}
    result = FakeRayWorkerProc().initialize_worker(
        0,
        original_env_vars,
        {"DRIVER": "1"},
        assigned_physical_gpu_ids=[3, 1],
    )

    assert result == "ok"
    assert original_env_vars == {"EXISTING": "1"}
    assert calls == [
        {
            "local_rank": 0,
            "env_vars": {
                "EXISTING": "1",
                "CUDA_VISIBLE_DEVICES": "1,3",
            },
            "driver_env_vars": {"DRIVER": "1"},
            "assigned_physical_gpu_ids": [3, 1],
        }
    ]
