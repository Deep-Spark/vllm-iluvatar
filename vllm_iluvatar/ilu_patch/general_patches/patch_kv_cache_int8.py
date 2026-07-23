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


def register() -> None:
    import vllm.envs as envs

    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "KVINT8":
        return

    from vllm.v1.kv_cache_interface import AttentionSpec

    from vllm_iluvatar.v1.attention.static_quant_utils import (
        int8_page_size_bytes,
        int8_scale_page_bytes,
    )
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    AttentionSpec.page_size_bytes = property(int8_page_size_bytes)

    from vllm.v1.core import kv_cache_utils as _kvu
    from vllm.v1.kv_cache_interface import KVCacheTensor

    _orig_get_config = _kvu.get_kv_cache_config_from_groups

    def _get_kv_cache_config_from_groups_int8(
        vllm_config, kv_cache_groups, available_memory
    ):
        config = _orig_get_config(vllm_config, kv_cache_groups, available_memory)
        attn_specs = {
            name: g.kv_cache_spec
            for g in kv_cache_groups
            for name in g.layer_names
            if isinstance(g.kv_cache_spec, AttentionSpec)
        }
        old_blocks = config.num_blocks
        if not attn_specs or old_blocks <= 0:
            return config

        slots = []
        for t in config.kv_cache_tensors:
            page = t.size // old_blocks
            attn = [n for n in t.shared_by if n in attn_specs]
            non_attn = [n for n in t.shared_by if n not in attn_specs]
            attn_main_page = int8_page_size_bytes(attn_specs[attn[0]]) if attn else 0
            scale_page = int8_scale_page_bytes(attn_specs[attn[0]]) if attn else 0
            slots.append((attn, attn_main_page, scale_page, non_attn, page if non_attn else 0))

        total_old = sum(t.size for t in config.kv_cache_tensors) // old_blocks
        total_new = sum(
            (attn_main_page + len(attn) * scale_page if attn else 0) + na_page
            for attn, attn_main_page, scale_page, non_attn, na_page in slots
        )
        if vllm_config.cache_config.num_gpu_blocks_override is not None:
            num_blocks = old_blocks
        else:
            num_blocks = max(old_blocks * total_old // total_new, 1)

        tensors = []
        for attn, attn_main_page, scale_page, non_attn, na_page in slots:
            if attn:
                tensors.append(KVCacheTensor(size=attn_main_page * num_blocks, shared_by=attn))
            if non_attn:
                tensors.append(KVCacheTensor(size=na_page * num_blocks, shared_by=non_attn))
        config.num_blocks = num_blocks
        config.kv_cache_tensors = tensors
        logger.info(
            "KV cache int8: separate K-scale buffers, num_blocks %d -> %d",
            old_blocks, num_blocks,
        )
        return config

    _kvu.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups_int8
    logger.info("KVINT8: patched AttentionSpec.page_size_bytes + get_kv_cache_config_from_groups")
