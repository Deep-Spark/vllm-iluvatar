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

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_SKIP_LAYERS: set[str] = set()


def add_skip_layers(*layer_names: str) -> None:
    _SKIP_LAYERS.update(layer_names)


def get_skip_layers() -> frozenset[str]:
    return frozenset(_SKIP_LAYERS)


def producer_last_layer(vllm_config: "VllmConfig") -> int | None:
    """Last layer a PD prefiller has to run, or None to run the whole model.

    DeepSeek-V4.1 publishes compressed KV on ``kv_source_layer_ids`` only
    (2, 8, 14, 20 of 40). Layers after that still own a sliding window the
    decoder reads. Truncating the prefiller is valid only when the decoder
    rebuilds that window (``swa_bounded_replay``). Otherwise those caches
    stay empty on D and decode is garbage.

    This is a pure function of the config so that the model and the connector
    worker reach the same bound without having to agree on an ordering.  Set
    ``VLLM_ILU_PD_PREFILL_FULL_DEPTH=1`` to keep the full depth even when
    bounded replay is on. A missing value truncates: vLLM does not forward
    this variable to workers.
    """
    transfer = vllm_config.kv_transfer_config
    if transfer is None or transfer.kv_role != "kv_producer":
        return None
    raw = os.getenv("VLLM_ILU_PD_PREFILL_FULL_DEPTH", "0").lower()
    if raw in ("1", "true", "yes"):
        return None
    # Truncation leaves layers above the last kv source without a window.
    # That is valid only when the decoder rebuilds the window.
    if not getattr(vllm_config.cache_config, "swa_bounded_replay", False):
        return None
    config = vllm_config.model_config.hf_text_config
    sources = getattr(config, "kv_source_layer_ids", None) or ()
    if not sources:
        return None
    return max(sources)
