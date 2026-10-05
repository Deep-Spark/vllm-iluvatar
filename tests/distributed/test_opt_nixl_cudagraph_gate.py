# SPDX-License-Identifier: Apache-2.0
"""The connector's PIECEWISE requirement must be producer-only.

A producer publishes push readiness from ``save_kv_layer()``, so its forward
has to leave room for that Python hook and cannot run as one full CUDA graph.
A consumer never pushes: both ``save_kv_layer_cuda`` and ``save_kv_layer_host``
return immediately on ``is_consumer`` and ``wait_for_layer_load`` is a no-op.

Forcing PIECEWISE on the consumer too is not merely conservative: the 1P1D
deployment compiles with ``CompilationMode.NONE``, where PIECEWISE is not
available, so vLLM downgrades it again to ``CUDAGraphMode.NONE`` and the
decoder silently runs every step eager.
"""

from __future__ import annotations

import pytest
from vllm.config.kv_transfer import KVTransferConfig

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.connector import (
    NixlConnector,
)


def _ktc(role: str, **extra: object) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="OptNixlConnector",
        kv_role=role,
        kv_connector_extra_config=dict(extra),
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_producer_requires_piecewise(role: str) -> None:
    assert NixlConnector.requires_piecewise_for_cudagraph(_ktc(role)) is True


def test_consumer_does_not_require_piecewise() -> None:
    assert NixlConnector.requires_piecewise_for_cudagraph(_ktc("kv_consumer")) is False


def test_decision_is_role_based_not_extra_config_based() -> None:
    # No deployment flag is involved any more: the role alone decides, so an
    # unrelated extra_config entry must not move the answer either way.
    assert (
        NixlConnector.requires_piecewise_for_cudagraph(
            _ktc("kv_consumer", something_else=True)
        )
        is False
    )
    assert (
        NixlConnector.requires_piecewise_for_cudagraph(
            _ktc("kv_producer", something_else=True)
        )
        is True
    )
