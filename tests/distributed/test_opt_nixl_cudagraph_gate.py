# SPDX-License-Identifier: Apache-2.0
"""Full CUDA graphs are fine on a pure consumer and must be refused on a producer.

A producer publishes push readiness from ``save_kv_layer()``: per-layer Python
that a full CUDA graph replay skips, so the KV push for that step is never
issued and the decoder waits for KV that does not come. A consumer never
pushes: both ``save_kv_layer_cuda`` and ``save_kv_layer_host`` return
immediately on ``is_consumer`` and ``wait_for_layer_load`` is a no-op.

vLLM asks ``requires_piecewise_for_cudagraph`` with only
``kv_connector_extra_config``; newer versions pass the whole
``KVTransferConfig``. Neither carries the role, so the gate cannot tell the
roles apart. It used to answer True for both, and under the 1P1D
``CompilationMode.NONE`` vLLM then downgraded the forced PIECEWISE again to
NONE, so the decoder silently ran every step eager. The gate now never asks,
and ``__init__``, which sees the role and the resolved mode, refuses the one
unsafe combination at startup.
"""

from __future__ import annotations

import pytest
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

import vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.connector as connector_module
from vllm_iluvatar.distributed.kv_transfer import apply_kv_transfer_registrations
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.connector import (
    NixlConnector,
)


def _config(role: str, cudagraph_mode: CUDAGraphMode) -> VllmConfig:
    # The 1P1D settings: CompilationMode.NONE, where PIECEWISE is not available.
    apply_kv_transfer_registrations()
    return VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=cudagraph_mode
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OptNixlConnector", kv_role=role
        ),
    )


@pytest.mark.parametrize(
    "extra_config",
    [
        {},
        {"something_else": True},
        # Newer vLLM passes the whole config; the hook still cannot see a role
        # it should act on, so the answer stays False either way.
        KVTransferConfig(kv_connector="OptNixlConnector", kv_role="kv_producer"),
        KVTransferConfig(kv_connector="OptNixlConnector", kv_role="kv_consumer"),
    ],
)
def test_gate_never_asks_for_piecewise(extra_config: object) -> None:
    assert NixlConnector.requires_piecewise_for_cudagraph(extra_config) is False


def test_consumer_keeps_full_decode_graphs() -> None:
    cfg = _config("kv_consumer", CUDAGraphMode.FULL_DECODE_ONLY)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


@pytest.mark.parametrize(
    "cudagraph_mode", [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL]
)
@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_pushing_role_with_full_graphs_is_refused(
    role: str, cudagraph_mode: CUDAGraphMode
) -> None:
    # kv_both satisfies is_kv_consumer as well, so the guard has to key on
    # "can this role push" (is_kv_producer) rather than "is it a consumer".
    cfg = _config(role, cudagraph_mode)
    with pytest.raises(ValueError, match="--enforce-eager"):
        NixlConnector(cfg, KVConnectorRole.SCHEDULER, kv_cache_config=None)


@pytest.mark.parametrize(
    ("role", "cudagraph_mode"),
    [
        # NONE is what --enforce-eager resolves to, which is how P runs.
        ("kv_producer", CUDAGraphMode.NONE),
        ("kv_both", CUDAGraphMode.NONE),
        ("kv_consumer", CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_safe_combinations_are_accepted(
    role: str, cudagraph_mode: CUDAGraphMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connector_module, "NixlConnectorScheduler", lambda *a: None)
    cfg = _config(role, cudagraph_mode)
    NixlConnector(cfg, KVConnectorRole.SCHEDULER, kv_cache_config=None)
