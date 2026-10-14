# SPDX-License-Identifier: Apache-2.0
"""Host-KV stand-ins for the 0.30 port.

The 0.27.1 connector could stage KV through a pinned host cache
(``vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.host_kv``).  That
package is intentionally not part of the 0.30 port yet, so this module supplies
no-op stand-ins and every ``host_kv_enabled()`` branch in the worker stays in
place but unreachable.

To restore host KV:

1. Port ``host_kv/`` from the 0.27.1 tree (it also needs
   ``v1/attention/backends/mla/host_metadata.py`` and the
   ``get_host_layer_plans`` hook in ``gpu/attn_utils.py``).
2. Replace the ``host_kv_stub`` imports in ``worker.py`` with the real package.
3. ``KVCacheConfig`` now owns the host pool directly via ``host_group_ids``,
   ``hisparse_host_num_blocks`` and ``hisparse_host_block_stride``, and
   ``KVCacheGroupSpec.host_resident`` marks a group as host-backed.  Prefer
   those over ``may_override_kv_cache_config`` rewriting the config by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig

T = TypeVar("T")


def host_kv_enabled() -> bool:
    return False


def may_override_kv_cache(kv_caches: dict[str, torch.Tensor]):
    return kv_caches


def may_override_kv_cache_config(kv_cache_config: KVCacheConfig):
    return kv_cache_config
