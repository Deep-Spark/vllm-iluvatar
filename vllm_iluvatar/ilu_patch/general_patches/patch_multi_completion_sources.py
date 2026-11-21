# SPDX-License-Identifier: Apache-2.0
"""Keep child completion ownership and distinct-worker completion accounting.

Corresponds to fixed vLLM MultiConnector at 6919d781a6. The private shim
uses the upstream extra-save/rank counters with duplicate notifications removed.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap

from vllm_iluvatar.ilu_patch import general_patch as p

_FINGERPRINTS = {
    "get_transfer_results": "8cdbd7f1da90fa4363f36f4c595bd3f14990f11bcb51a5bb4e8a2c5bde601cec",
    "build_connector_worker_meta": "0495c8722fe0c133cc1f9fe5b84c5685ec0438d4fdbe142e7e318848596c0f1e",
    "update_connector_output": "3abc26be6160159819351da55a7403fcf6e6f394b3baeb7b205c51481ad0db68",
}
_AGGREGATE_FINGERPRINT = "06d2bf78873d3d50d004c0cbd83945ce62e4786dee8fc972e7834a46ab7e7c15"


def register() -> None:
    p.register_once(__name__, _apply, detail="isolate and deduplicate Multi completions")


def _apply() -> None:
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.multi_completion import (
        install_multi_completion_sources,
        install_multi_rank_completion_dedup,
    )
    installed = [
        getattr(getattr(MultiConnector, name), "_ilu_multi_completion_sources", False)
        for name in _FINGERPRINTS
    ]
    installed.extend((
        getattr(MultiConnector.get_transfer_results, "_ilu_multi_distinct_child_sends", False),
        getattr(MultiConnector.get_transfer_results, "_ilu_multi_mp_store_terminal_completion", False),
        getattr(KVOutputAggregator.aggregate, "_ilu_multi_distinct_worker_sends", False),
    ))
    if all(installed):
        return
    if any(installed):
        raise RuntimeError("Partial or older Multi completion patch is already installed")
    for name, expected in _FINGERPRINTS.items():
        source = textwrap.dedent(inspect.getsource(getattr(MultiConnector, name))).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError("MultiConnector source drift: " + name)
    source = textwrap.dedent(inspect.getsource(KVOutputAggregator.aggregate)).strip()
    if hashlib.sha256(source.encode()).hexdigest() != _AGGREGATE_FINGERPRINT:
        raise RuntimeError("KVOutputAggregator source drift: aggregate")
    install_multi_completion_sources(MultiConnector)
    install_multi_rank_completion_dedup(KVOutputAggregator)
