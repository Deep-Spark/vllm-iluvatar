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

from dataclasses import dataclass, field
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="may add properties to AttentionSpec/MLAAttentionSpec/KVCacheTensor",
    )


def _apply() -> None:
    import vllm.envs as envs

    if (
        envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == ""
        and envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == ""
    ):
        return

    # add property to spec, those properties will be used by:
    #  get_kv_cache_config_from_groups from vllm_iluvatar.v1.core.kv_cache_utils
    #  _reshape_kv_cache_tensors_with_scales from vllm_iluvatar.v1.worker.gpu_model_runner
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "":
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        from vllm_iluvatar.v1.kv_cache_interface import (
            _full_attention_fixed_size_bytes,
            _full_attention_real_page_size_bytes,
            _full_attention_scale_page_size_bytes,
            _sliding_window_fixed_size_bytes,
            _sliding_window_real_page_size_bytes,
            _sliding_window_scale_page_size_bytes,
        )

        FullAttentionSpec.real_page_size_bytes = property(
            _full_attention_real_page_size_bytes
        )
        FullAttentionSpec.scale_page_size_bytes = property(
            _full_attention_scale_page_size_bytes
        )
        FullAttentionSpec.fixed_size_bytes = property(_full_attention_fixed_size_bytes)
        SlidingWindowSpec.real_page_size_bytes = property(
            _sliding_window_real_page_size_bytes
        )
        SlidingWindowSpec.scale_page_size_bytes = property(
            _sliding_window_scale_page_size_bytes
        )
        SlidingWindowSpec.fixed_size_bytes = property(_sliding_window_fixed_size_bytes)

    if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE != "":
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        from vllm_iluvatar.v1.kv_cache_interface import (
            _mla_attention_fixed_size_bytes,
            _mla_attention_real_page_size_bytes,
            _mla_attention_scale_page_size_bytes,
        )

        MLAAttentionSpec.real_page_size_bytes = property(
            _mla_attention_real_page_size_bytes
        )
        MLAAttentionSpec.scale_page_size_bytes = property(
            _mla_attention_scale_page_size_bytes
        )
        MLAAttentionSpec.fixed_size_bytes = property(_mla_attention_fixed_size_bytes)

    from vllm.v1.kv_cache_interface import KVCacheTensor

    def _ensure_dataclass_field(
        cls, name: str, annotation: object, default: object
    ) -> None:
        if name in cls.__dataclass_fields__:
            return
        cls.__annotations__[name] = annotation
        setattr(cls, name, field(default=default))
        dataclass(cls)

    # add fixed_size and scale_size to KVCacheSpec, those properties will be used by:
    #  get_kv_cache_config_from_groups from vllm_iluvatar.v1.core.kv_cache_utils
    #  _reshape_kv_cache_tensors_with_scales from vllm_iluvatar.v1.worker.gpu_model_runner
    _ensure_dataclass_field(KVCacheTensor, "fixed_size", int, 0)
    _ensure_dataclass_field(KVCacheTensor, "scale_size", int, 0)
    old_init = KVCacheTensor.__init__
    if getattr(old_init, "_register", False):
        return

    @wraps(old_init)
    def __init__(self, *args, fixed_size: int = 0, scale_size: int = 0, **kwargs):
        old_init(self, *args, **kwargs)
        self.fixed_size = fixed_size
        self.scale_size = scale_size

    __init__._register = True
    KVCacheTensor.__init__ = __init__
