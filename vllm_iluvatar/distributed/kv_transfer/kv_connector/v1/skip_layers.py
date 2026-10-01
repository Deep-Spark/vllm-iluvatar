# SPDX-License-Identifier: Apache-2.0
"""Layers the model asks KV connectors to leave out of the transfer.

The model fills this set while it is built (same worker process, before
``register_kv_caches``), under the same name it puts into
``static_forward_context``, which is also the key of that layer's KV cache.
Connectors read it when registering: a listed layer is neither registered
nor transferred, and never shows up in the connector's per-layer mappings.

It is per layer rather than per KV group because the layers that need it
share groups with layers that do transfer (on DeepSeek-V4.1 at TP2/PP8,
group 19 is [layers.20, layers.30] and only layers.30 is a forwarder), so
``KVCacheGroupSpec.enable_kv_transfer`` cannot express it.
"""

_SKIP_LAYERS: set[str] = set()


def add_skip_layers(*layer_names: str) -> None:
    _SKIP_LAYERS.update(layer_names)


def get_skip_layers() -> frozenset[str]:
    return frozenset(_SKIP_LAYERS)
