# SPDX-License-Identifier: Apache-2.0
"""Iluvatar KV-transfer connector registration."""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_ILU_KV_CONNECTORS = (
    (
        "IluNixlConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl",
        "IluNixlConnector",
    ),
    (
        "IluNixlPullConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl",
        "IluNixlPullConnector",
    ),
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
    (
        "IluMooncakeStoreConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store",
        "IluMooncakeStoreConnector",
    ),
    (
        "MooncakeStoreConnector",
        "vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store",
        "MooncakeStoreConnector",
    ),
)

__all__ = ["apply_kv_transfer_registrations"]


def _registered_to_iluvatar(loader: object) -> bool:
    closure = getattr(loader, "__closure__", None)
    if closure:
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, str) and val.startswith("vllm_iluvatar."):
                return True
        return False
    try:
        current_cls = loader()  # type: ignore[operator]
        return current_cls.__module__.startswith("vllm_iluvatar.")
    except Exception:
        return False


def apply_kv_transfer_registrations() -> None:
    """Register Iluvatar KV connectors with the vLLM connector factory."""
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for name, module_path, class_name in _ILU_KV_CONNECTORS:
        registered = KVConnectorFactory._registry
        if name in registered:
            if _registered_to_iluvatar(registered[name]):
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
        logger.debug(
            "ILU KV connector registered: %s -> %s:%s",
            name,
            module_path,
            class_name,
        )
