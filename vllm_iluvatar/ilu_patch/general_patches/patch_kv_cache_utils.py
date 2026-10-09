# SPDX-License-Identifier: Apache-2.0
"""KV cache int8 quantization infrastructure
(VLLM_ILU_FLASH_ATTN_QUANT_TYPE=KVINT8/KINT8, VLLM_ILU_MLA_ATTN_QUANT_TYPE=INT8).

Patches community objects so int8 KV cache memory is sized correctly, aligned
with the vLLM 0.31.0 community mechanism (B2 "shared backing buffer"):

* ``AttentionSpec.state_content_size_bytes`` -- overridden (when any int8
  quant is enabled) to return the int8 per-cell byte size; see
  patch_kv_cache_interface.py.
* ``get_kv_cache_config_from_groups`` -- replans ``num_blocks`` against a
  scale-aware budget and appends the per-tensor K-scale pages as a contiguous
  tail inside the SAME community backing allocation; attaches ``scale_size``
  to each ``KVCacheTensor``.
* ``_pool_bytes_per_block`` -- overridden to return
  ``kv_bytes_per_block + scale_per_block_total`` so the community shrink loop in
  ``get_kv_cache_configs`` re-plans each worker with a scale-aware budget. This
  keeps the override *pure* (``num_blocks = available // _pool_bytes_per_block``),
  so all ranks still converge to the same ``num_blocks`` after the single
  re-plan.

Note: ``get_kv_cache_configs`` is not overridden on 0.31.0. Its shrink loop
re-plans each worker by calling this override, so ``scale_size`` is recomputed
consistently during re-plan -- the old ILU manual proportional scale shrink is
obsolete and would leave strides/offsets inconsistent with the shrunken
allocation.

Target   : vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
           vllm.v1.core.kv_cache_utils._pool_bytes_per_block
Baseline : v0.31.0 (db9527a468)
Scope    : only active when VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "" or
           VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="may override get_kv_cache_config_from_groups and _pool_bytes_per_block",
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
        _ilu_pool_bytes_per_block,
        get_kv_cache_config_from_groups,
    )

    _m.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    _m._pool_bytes_per_block = _ilu_pool_bytes_per_block
