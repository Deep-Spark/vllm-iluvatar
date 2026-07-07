# SPDX-License-Identifier: Apache-2.0
"""Iluvatar helpers for RayExecutorV2 worker initialization."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict, List, Optional


def _device_control_env_vars(
    assigned_physical_gpu_ids: Optional[List[int]],
    current_platform,
) -> Dict[str, str]:
    if assigned_physical_gpu_ids is None:
        return {}
    device_control_env_var = getattr(current_platform, "device_control_env_var", "")
    if not device_control_env_var:
        return {}
    return {
        device_control_env_var: ",".join(
            str(gpu_id) for gpu_id in sorted(assigned_physical_gpu_ids)
        )
    }


def wrap_initialize_worker(original_initialize_worker: Callable) -> Callable:
    @wraps(original_initialize_worker)
    def initialize_worker(
        self,
        local_rank: int,
        env_vars: Dict[str, str],
        driver_env_vars: Optional[Dict[str, str]] = None,
        assigned_physical_gpu_ids: Optional[List[int]] = None,
    ):
        from vllm.platforms import current_platform  # pyright: ignore[reportMissingImports]

        worker_env_vars = dict(env_vars)
        worker_env_vars.update(
            _device_control_env_vars(assigned_physical_gpu_ids, current_platform)
        )
        return original_initialize_worker(
            self,
            local_rank,
            worker_env_vars,
            driver_env_vars,
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
        )

    return initialize_worker
