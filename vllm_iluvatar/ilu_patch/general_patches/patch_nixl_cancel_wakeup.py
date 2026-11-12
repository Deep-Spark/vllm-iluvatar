# SPDX-License-Identifier: Apache-2.0
"""Drain late producer cancellation proofs without waiting for a new request."""
from __future__ import annotations

import hashlib
import inspect
import textwrap

from vllm_iluvatar.ilu_patch import general_patch as p

_FINGERPRINTS = {
    "_process_input_queue": "6e929f1af79e75a2d3bb89aed3883bddea4ddf9c1956ba77235600e16506d2b0",
    "_handle_client_request": "199cda971217582a43a0355970e47ad9f616c738ffd01a93d9e82302820522d5",
}


def register() -> None:
    p.register_once(__name__, _apply, detail="wake NIXL producer cancellation polling")


def _apply() -> None:
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.cancel_progress import (
        install_engine_cancel_wakeup,
    )
    if getattr(EngineCoreProc._process_input_queue, "_ilu_nixl_cancel_wakeup", False):
        return
    for name, expected in _FINGERPRINTS.items():
        source = textwrap.dedent(inspect.getsource(getattr(EngineCoreProc, name))).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError("Engine cancellation wakeup source drift: " + name)
    install_engine_cancel_wakeup(EngineCoreProc, EngineCoreRequestType.WAKEUP)
