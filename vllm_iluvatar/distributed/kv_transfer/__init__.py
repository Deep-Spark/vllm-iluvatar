# SPDX-License-Identifier: Apache-2.0
"""Iluvatar KV-transfer connector registration."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ILU_KV_CONNECTORS = (
    # ilu_nixl (the PULL connector) is not ported to 0.30 yet: it subclasses
    # upstream vllm.distributed.kv_transfer.kv_connector.v1.nixl.* internals,
    # which were substantially rewritten for 0.30 (base/pull/push split).
    (
        "IluNixlPushConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl",
        "IluNixlPushConnector",
    ),
    (
        "OptNixlConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl",
        "OptNixlConnector",
    ),
)

__all__ = ["apply_kv_transfer_registrations"]


def apply_kv_transfer_registrations() -> None:
    """Register Iluvatar KV connectors with the vLLM connector factory."""
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for name, module_path, class_name in _ILU_KV_CONNECTORS:
        registered = KVConnectorFactory._registry
        if name in registered:
            current_cls = registered[name]()
            if current_cls.__module__.startswith("vllm_iluvatar."):
                logger.debug("ILU KV connector already registered: %s", name)
                continue
            logger.info(
                "ILU KV connector overriding existing registration: %s -> %s:%s",
                name,
                module_path,
                class_name,
            )
            del registered[name]
        KVConnectorFactory.register_connector(name, module_path, class_name)
        logger.info(
            "ILU KV connector registered: %s -> %s:%s",
            name,
            module_path,
            class_name,
        )
