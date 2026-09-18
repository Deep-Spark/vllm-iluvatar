# SPDX-License-Identifier: Apache-2.0
"""Wiring for the ``ray.init`` num_cpus cap.

Corresponds to: vllm/v1/engine/utils.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

This patch is a pure wrapper with no override implementation of its own, so no
upstream-mirrored implementation sibling is needed.

Target   : the ``ray.init`` call used by vllm.v1.engine.utils.CoreEngineActorManager
Baseline : main_a67969c5a0
Why      : containers without a cgroup CPU quota expose the host CPU count
          (e.g. 255) to ray; ``ray.init()`` defaults ``num_cpus`` to that, so the
          raylet pre-starts 255 python workers (``--num_prestart_python_workers
          == num_cpus``) which thrash/hang during startup and wedge the driver's
          ``CoreWorker`` connect on the raylet unix socket. Capping ``num_cpus``
          avoids the thrash.
How      : rebind ``ray.init`` to a wrapper that injects a capped ``num_cpus``
          (from ``VLLM_ILU_RAY_NUM_CPUS``, default 16) only for fresh local
          cluster starts (``address`` is None) that did not pass ``num_cpus``.
          Existing-cluster connects (``address="auto"`` / a real address) and
          explicit ``num_cpus`` are passed through unchanged. ``@wraps`` keeps
          the full ``ray.init`` signature, so the signature guard holds.
Scope    : only when a fresh local ray cluster is started on Iluvatar (e.g. the
          DP MoE actor manager path with ``--data-parallel-backend ray``).
Remove   : when ray/vllm auto-caps pre-start workers or vllm passes ``num_cpus``
          to ``ray.init``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)


def _wrap_ray_init(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``ray.init`` to cap ``num_cpus`` for fresh local cluster starts.

    Only injects ``num_cpus`` when BOTH hold:
      - the caller did not specify ``num_cpus``, AND
      - no ``address`` is given (a fresh local cluster is being started).

    Connecting to an existing cluster (``address="auto"`` or a real address) and
    explicit ``num_cpus`` are passed through unchanged.
    """

    @wraps(original)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        import vllm.envs as envs

        address = args[0] if args else kwargs.get("address")
        if address is None and kwargs.get("num_cpus") is None:
            kwargs["num_cpus"] = envs.VLLM_ILU_RAY_NUM_CPUS
        return original(*args, **kwargs)

    return _wrapped


def register() -> None:
    """Register the ray.init num_cpus cap wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="ray.init num_cpus cap for fresh local cluster starts",
    )


def _apply() -> None:
    import ray as _ray

    original = _ray.init
    wrapped = _wrap_ray_init(original)
    # [ILU override] ray.init - delta: inject a capped num_cpus for fresh local
    # cluster starts so the raylet does not pre-start host-count workers and
    # thrash the driver connect. @wraps preserves the signature.
    if inspect.signature(original) != inspect.signature(wrapped):
        raise RuntimeError(
            "ILU patch: ray.init signature mismatch after wrap: "
            f"community {inspect.signature(original)} != wrapped "
            f"{inspect.signature(wrapped)}"
        )
    _ray.init = wrapped
