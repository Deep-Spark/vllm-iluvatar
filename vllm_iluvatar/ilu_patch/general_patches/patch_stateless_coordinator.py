# SPDX-License-Identifier: Apache-2.0
"""Wiring for the stateless coordinator communicator rebind.

Corresponds to: vllm/distributed/stateless_coordinator.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

This patch is a pure module-global rebind with no override implementation of its
own, so no upstream-mirrored implementation sibling is needed.

Target   : vllm.distributed.stateless_coordinator (module-global
           ``CudaCommunicator`` symbol used by StatelessGroupCoordinator.__init__)
Baseline : main_6e448d0
Why      : the stateless / elastic-EP path hardcodes CudaCommunicator and asserts
           ``device_comm_cls == CudaCommunicator``, rejecting OOT communicators.
           IluPlatform returns IluCommunicator, so the assert would fail.
How      : rebind the module-global ``CudaCommunicator`` name to IluCommunicator
           (zero method-body copy). Both the assert (``device_comm_cls ==
           CudaCommunicator``) and the construction (``CudaCommunicator(...)``)
           then resolve to IluCommunicator. The whole __init__ body keeps coming
           from community and tracks upstream automatically.
Scope    : only when enable_elastic_ep (StatelessGroupCoordinator is constructed).
Remove   : when upstream stateless path instantiates the resolved
           device_comm_cls(...) (then this rebind degrades to a no-op).
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)


def register() -> None:
    """Register the CudaCommunicator -> IluCommunicator rebind."""
    p.register_once(__name__, _apply, detail="CudaCommunicator rebind")


def _apply() -> None:
    import vllm.distributed.stateless_coordinator as _m

    if not hasattr(_m, "CudaCommunicator"):
        logger.warning(
            "ILU stateless_coordinator patch: community module has no "
            "'CudaCommunicator' symbol; skipping rebind."
        )
        return

    from vllm_iluvatar.distributed.device_communicators.ilu_communicator import (
        IluCommunicator,
    )

    # [ILU override] stateless_coordinator.CudaCommunicator -> IluCommunicator
    # delta: make the hardcoded CudaCommunicator assert/construction use the
    # Iluvatar communicator without copying StatelessGroupCoordinator.__init__.
    _m.CudaCommunicator = IluCommunicator
