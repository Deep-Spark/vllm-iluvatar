# SPDX-License-Identifier: Apache-2.0
"""KV cache int8 quantization infrastructure (VLLM_ILU_FLASH_ATTN_QUANT_TYPE=KVINT8).

Patches two community objects so that KVINT8 KV cache memory is sized correctly:

* ``AttentionSpec.page_size_bytes`` — returns the int8 planes-only main buffer
  size (K scale is a separate standalone tensor, not part of page_size).
* ``get_kv_cache_config_from_groups`` — recalculates num_blocks based on the
  int8 page size and creates separate K-scale buffer entries.

Target   : vllm.v1.kv_cache_interface.AttentionSpec.page_size_bytes
           vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
Scope    : only active when VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8"
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="may override get_kv_cache_config_from_groups and get_kv_cache_configs functions",
    )


def _apply() -> None:
    import vllm.envs as envs

    if (
        envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == ""
        and envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == ""
    ):
        return

    from vllm.v1.core import kv_cache_utils as _m

    from vllm_iluvatar.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_configs,
    )

    _m.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    # NOTE: on register Stage, we can get the world size info,
    # so we have to patch get_kv_cache_configs to fix the num_blocks between different workers
    p.rebind_module_func(_m, "get_kv_cache_configs", get_kv_cache_configs)
