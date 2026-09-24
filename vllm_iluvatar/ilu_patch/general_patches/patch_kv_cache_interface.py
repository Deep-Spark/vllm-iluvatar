# SPDX-License-Identifier: Apache-2.0
"""KV cache int8 quantization infrastructure
(VLLM_ILU_FLASH_ATTN_QUANT_TYPE=KVINT8/KINT8, VLLM_ILU_MLA_ATTN_QUANT_TYPE=INT8).

Patches community objects so int8 KV cache memory is sized correctly, aligned
with the vLLM 0.30.0 community mechanism:

* ``AttentionSpec.state_content_size_bytes`` -- overridden (when any int8
  quant is enabled) to return the int8 per-cell byte size. The community
  formula ``unpadded_page_size_bytes = num_heads * num_states *
  state_content_size_bytes`` (and its alias ``real_page_size_bytes``, plus
  ``compute_logical_shape``'s C dim) then naturally yields int8 sizing for
  every spec (FullAttention / SlidingWindow via flash int8, MLAAttentionSpec /
  SlidingWindowMLASpec via MLA int8). MLA is single-sided (head_size_v == 0),
  so (head_size + head_size_v) collapses to head_size. ``state_content_bytes``
  (nvfp4 / fp8_ds_mla / ...) is honored via the fallback.
* ``AttentionSpec.num_heads`` -- overridden to reject ``num_head_slots``
  (incompatible with the ILU int8 layout).
* ``get_kv_cache_config_from_groups`` -- replans ``num_blocks`` against a
  scale-aware budget and co-locates/appends the per-tensor K-scale pages in
  the SAME community backing allocation; attaches ``scale_size`` /
  ``scale_offset`` to each ``KVCacheTensor``.

Target   : vllm.v1.kv_cache_interface.AttentionSpec.state_content_size_bytes
           vllm.v1.kv_cache_interface.AttentionSpec.num_heads
           vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
           vllm.v1.core.kv_cache_utils._pool_bytes_per_block
Baseline : v0.30.0 (ced6857afa)
Scope    : only active when VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "" or
           VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="may add properties to AttentionSpec/KVCacheTensor",
    )


def _apply() -> None:
    import vllm.envs as envs

    if (
        envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == ""
        and envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == ""
    ):
        return

    from vllm.v1.kv_cache_interface import AttentionSpec

    from vllm_iluvatar.v1.kv_cache_interface import (
        attention_num_heads,
        attention_state_content_size_bytes,
    )

    # Override the community per-cell size so the community formula
    # ``unpadded_page_size_bytes = num_heads * num_states *
    # state_content_size_bytes`` (and the ``real_page_size_bytes`` alias, plus
    # ``compute_logical_shape``'s C dim) all yield int8 sizing. Applied on the
    # base class so every subclass inherits it; the function dispatches by
    # env var and spec type (flash int8 vs MLA int8). int8 layout assumes one
    # slot per KV head.
    AttentionSpec.state_content_size_bytes = property(
        attention_state_content_size_bytes
    )
    AttentionSpec.num_heads = property(attention_num_heads)

    # scale_page_size_bytes describes the K-scale storage
    # sharing the KV backing buffer. Used by:
    #  get_kv_cache_config_from_groups from vllm_iluvatar.v1.core.kv_cache_utils
    #  allocate_kv_cache_scale from vllm_iluvatar.v1.worker.gpu.model_runner
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "":
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        from vllm_iluvatar.v1.kv_cache_interface import (
            _full_attention_scale_page_size_bytes,
            _sliding_window_scale_page_size_bytes,
        )

        FullAttentionSpec.scale_page_size_bytes = property(
            _full_attention_scale_page_size_bytes
        )
        SlidingWindowSpec.scale_page_size_bytes = property(
            _sliding_window_scale_page_size_bytes
        )

    if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE != "":
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        from vllm_iluvatar.v1.kv_cache_interface import (
            _mla_attention_scale_page_size_bytes,
        )

        MLAAttentionSpec.scale_page_size_bytes = property(
            _mla_attention_scale_page_size_bytes
        )

    from vllm.v1.kv_cache_interface import KVCacheTensor

    def _ensure_dataclass_field(
        cls, name: str, annotation: object, default: object
    ) -> None:
        if name in cls.__dataclass_fields__:
            return
        cls.__annotations__[name] = annotation
        setattr(cls, name, field(default=default))
        dataclass(cls)

    # add scale_size and scale_offset to KVCacheTensor; those
    # properties will be used by:
    #  get_kv_cache_config_from_groups from vllm_iluvatar.v1.core.kv_cache_utils
    #  allocate_kv_cache_scale from vllm_iluvatar.v1.worker.gpu.model_runner
    # scale_offset >= 0  -> block-outermost: scale is co-located inside each
    #                       block at this byte offset.
    # scale_offset == -1 -> layer-outermost: scale lives in a contiguous tail
    #                       (offset resolved by allocate_kv_cache_scale).
    _ensure_dataclass_field(KVCacheTensor, "scale_size", int, 0)
    _ensure_dataclass_field(KVCacheTensor, "scale_offset", int, -1)
    old_init = KVCacheTensor.__init__
    if getattr(old_init, "_register", False):
        return

    @wraps(old_init)
    def __init__(
        self,
        *args,
        scale_size: int = 0,
        scale_offset: int = -1,
        **kwargs,
    ):
        old_init(self, *args, **kwargs)
        self.scale_size = scale_size
        self.scale_offset = scale_offset

    __init__._register = True
    KVCacheTensor.__init__ = __init__
