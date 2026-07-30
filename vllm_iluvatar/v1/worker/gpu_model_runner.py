# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPU model runner — Attention/MLA KV cache support.

Active when ``VLLM_ILU_FLASH_ATTN_QUANT_TYPE!=""`` or ``VLLM_ILU_MLA_ATTN_QUANT_TYPE!=""``.
For more details, see vllm/v1/worker/gpu_worker.py.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from math import prod
from typing import Any

import torch
import vllm.envs as envs
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import bind_kv_cache

from vllm_iluvatar.v1.attention.ilu_static_quant import (
    load_static_v_scales,
)

logger = init_logger("vllm." + __name__)


def _bind_kv_cache_scale(
    kv_caches_scale: dict[str, torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    runner_kv_caches_scale: list[torch.Tensor | list[torch.Tensor]],
    num_attn_module: int = 1,
) -> None:
    """Bind ILU KV-cache scale tensors to runner and attention layers."""
    assert len(runner_kv_caches_scale) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches_scale:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches_scale.append(kv_caches_scale[layer_name])

    for layer_name, kv_cache_scale in kv_caches_scale.items():
        forward_context[layer_name].kv_cache_scale = kv_cache_scale


def _reshape_attention_kv_cache_with_scale(
    kv_raw_tensor: torch.Tensor,
    kv_cache_spec: AttentionSpec,
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...],
    num_blocks: int,
    packing: tuple[int, int] | None,
    layer_name: str,
    kv_raw_scale_tensor: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    # NOTE: >>> iluvatar specific logic
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8":
            if (
                kv_cache_spec.cache_dtype_str is not None
                and kv_cache_spec.cache_dtype_str == "origin"
            ):
                dtype = kv_cache_spec.dtype
            else:
                dtype = torch.int8
        else:
            dtype = kv_cache_spec.dtype
    else:
        if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != "":
            dtype = torch.int8
        else:
            dtype = kv_cache_spec.dtype
    # <<< iluvatar specific logic
    if packing is not None:
        offset, block_stride = packing
        assert inv_order[0] == 0
        page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
        kv_cache = (
            kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
            .view(dtype)
            .view(kv_cache_shape)
        )
    elif kv_cache_spec.page_size_padded is not None:
        # Use a strided view to skip the padding between physical pages.
        #
        # Only num-blocks-first layouts are supported (the block dimension is
        # dim 0 of the unpermuted shape). kv-first layouts such as ROCm's
        # ``(2, num_blocks, ...)`` are intentionally not supported here. For a
        # num-blocks-first layout the only stride that must change is the block
        # stride: every other (contiguous) stride already steps within the
        # unpadded region of a page, so no further adjustment is needed.
        assert kv_cache_shape[0] == num_blocks, (
            "Padded KV pages require a num-blocks-first KV cache layout (got "
            f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
            "kv-first layouts are not supported."
        )
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size

        num_blocks_dim = inv_order[0]
        strides = list(torch.empty(permuted_kv_cache_shape).stride())
        strides[num_blocks_dim] = page_stride

        kv_cache = torch.as_strided(
            kv_raw_tensor.view(dtype),
            size=permuted_kv_cache_shape,
            stride=tuple(strides),
        )
    else:
        kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

    # NOTE: >>> iluvatar specific logic
    # the following code is for MLA/Flash attention KV cache scales
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        kv_scale_caches = []
        if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8" and not (
            kv_cache_spec.cache_dtype_str is not None
            and kv_cache_spec.cache_dtype_str == "origin"
        ):
            kv_scale_caches_shape = (kv_cache.shape[0], kv_cache.shape[1], 2)
            kv_scale_caches = [
                kv_raw_scale_tensor[0].view(torch.float32).view(kv_scale_caches_shape),
                None,
            ]
    else:
        if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
            layer_idx = extract_layer_index(layer_name)
            if "mtp" in layer_name:
                layer_idx = "mtp." + str(layer_idx)
            else:
                layer_idx = str(layer_idx)
            static_value_caches_scales = load_static_v_scales(
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                kv_raw_scale_tensor[0].device,
                layer_idx,
            )
            if layer_idx not in static_value_caches_scales:
                if "mtp" not in layer_name:
                    raise RuntimeError(
                        "For KVINT8, the none mtp model must have v_scale in the provided v_scale file"
                    )
                fallback_idx = max(static_value_caches_scales)
                logger.warning(
                    "[KVINT8] layer_idx=%d not in v_scale file, "
                    "fallback to layer_idx:%d",
                    layer_idx,
                    fallback_idx,
                )
                layer_idx = fallback_idx
            v_scale = static_value_caches_scales[layer_idx]
            key_scale_caches_shape = [
                num_blocks,
                kv_cache_spec.num_kv_heads,
                -1,
                # kv_cache_spec.block_size for mamba models, the block size may be padded, use -1 to automatically determine the block size
            ]
            value_scale_caches_shape = [
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            ]
            k_scale_cache = (
                kv_raw_scale_tensor[0].view(torch.float32).view(key_scale_caches_shape)
            )
            v_scale_cache = (
                kv_raw_scale_tensor[1]
                .view(torch.float32)
                .view(value_scale_caches_shape)
            )
            # load static value cache scale precomputed outside
            if static_value_caches_scales is not None:
                v_scale_cache.copy_(v_scale)
            kv_scale_caches = [k_scale_cache, v_scale_cache]
        elif envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KINT8":
            # num_blocks, num_heads, block_size for k
            kv_caches_scale_shape = [
                num_blocks,
                kv_cache_spec.num_kv_heads,
                -1,
                # kv_cache_spec.block_size for mamba models, the block size may be padded, use -1 to automatically determine the block size
            ]
            kv_scale_caches = [
                kv_raw_scale_tensor[0].view(torch.float32).view(kv_caches_scale_shape),
                None,
            ]

    return kv_cache, kv_scale_caches
    # <<< iluvatar specific logic


class IluvatarGPUModelRunner(GPUModelRunner):
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ):
        super().initialize_kv_cache(kv_cache_config, is_profiling)
        # NOTE >>> iluvatar specific logic
        # this part is needed for register kv scale tensors.
        kv_scale_caches = self.kv_scale_caches
        if has_kv_transfer_group() and not is_profiling:
            kv_transfer_group = get_kv_transfer_group()
            try:
                kv_transfer_group.register_kv_scale_caches(kv_scale_caches)
            except Exception as e:
                logger.error("Failed to register kv scale caches: %s", e)
        # <<< iluvatar specific logic

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        # NOTE: this function will be called when:
        #   determine_available_memory for cuda graph memory calculation
        #   get_kv_cache_configs for init available num blocks.
        self.kv_scale_caches = []

        # TODO: support uniform kv cache quant
        if self.use_uniform_kv_cache(self.attn_groups):
            raise RuntimeError("ILU KV-cache scale does not support uniform KV cache.")

        # first, use super to allocate and reshape kv cache tensors
        kv_cache_raw_tensors = super()._allocate_kv_cache_tensors(kv_cache_config)
        # then, allocate and reshape kv cache scale tensors
        kv_cache_scale_raw_tensors = self._allocate_kv_cache_scale_tensors(
            kv_cache_config
        )
        kv_caches, kv_scale_caches = self._reshape_kv_cache_tensors_with_scales(
            kv_cache_raw_tensors, kv_cache_scale_raw_tensors, kernel_block_sizes
        )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]
            kv_scale_caches[layer_name] = kv_scale_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )

        # finally, bind kv cache and kv cache scale tensors to the forward context
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        _bind_kv_cache_scale(
            kv_scale_caches,
            self.compilation_config.static_forward_context,
            self.kv_scale_caches,
            num_attn_module,
        )

        # TODO: we need register kv caches.
        return kv_caches

    def _allocate_kv_cache_scale_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, list[torch.Tensor]]:
        kv_cache_scale_raw_tensors: dict[str, list[torch.Tensor]] = {}

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = None
            fixed_tensor = None
            if kv_cache_tensor.scale_size:
                tensor = torch.zeros(
                    kv_cache_tensor.scale_size, dtype=torch.int8, device=self.device
                )
            if kv_cache_tensor.fixed_size:
                fixed_tensor = torch.zeros(
                    kv_cache_tensor.fixed_size,
                    dtype=torch.int8,
                    device=self.device,
                )
            res_tensor = [tensor, fixed_tensor]

            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_scale_raw_tensors[layer_name] = res_tensor

        return kv_cache_scale_raw_tensors

    def _reshape_kv_cache_tensors_with_scales(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kv_cache_scale_raw_tensors: dict[str, list[torch.Tensor]],
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.
        Then reshape the KV cache scale tensors.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kv_cache_scale_raw_tensors: The KV cache scale buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]: A tuple
            containing two dictionaries:
            - The first dictionary maps layer names to their corresponding memory
              buffer for KV cache.
            - The second dictionary maps layer names to their corresponding memory
              buffer for KV cache scale.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        # NOTE: >>> iluvatar specific logic
        kv_scale_caches: dict[str, list[torch.Tensor]] = {}
        # <<< iluvatar specific logic

        # Map layer names to (offset, block_stride) within the packed
        # backing tensor so we can create strided views per layer.
        layer_packing: dict[str, tuple[int, int]] = {}
        for kv_tensor in self.kv_cache_config.kv_cache_tensors:
            if kv_tensor.block_stride > 0:
                for ln in kv_tensor.shared_by:
                    layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                packing = layer_packing.get(layer_name)
                if packing is not None:
                    _, blk_stride = packing
                    num_blocks = raw_tensor.numel() // blk_stride
                else:
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    # For MLA with compression, storage_block_size != block_size
                    if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                        shape_block_size = kv_cache_spec.storage_block_size
                    else:
                        shape_block_size = kernel_block_size

                    # NOTE: >>> iluvatar specific logic
                    raw_tensor_scales = kv_cache_scale_raw_tensors[layer_name]
                    if not (
                        envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8"
                        and getattr(kv_cache_spec, "cache_dtype_str", None) is not None
                        and kv_cache_spec.cache_dtype_str == "origin"
                    ):
                        assert (
                            raw_tensor_scales[0].numel()
                        ) % kv_cache_spec.scale_page_size_bytes == 0
                        assert (
                            raw_tensor_scales[0].numel()
                            // kv_cache_spec.scale_page_size_bytes
                            == num_blocks
                        )
                    # <<< iluvatar specific logic

                    # Skipped layers (--kv-cache-dtype-skip-layers) need
                    # the unquantized shape.
                    layer_cache_dtype_str = (
                        "auto"
                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                        else getattr(
                            kv_cache_spec,
                            "cache_dtype_str",
                            None,
                        )
                        or self.cache_config.cache_dtype
                    )
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        shape_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=layer_cache_dtype_str,
                    )
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    # NOTE: >>> iluvatar specific logic
                    kv_cache, kv_cache_scale = _reshape_attention_kv_cache_with_scale(
                        raw_tensor,
                        kv_cache_spec,
                        kv_cache_shape,
                        kv_cache_stride_order,
                        kernel_num_blocks,
                        packing,
                        layer_name,
                        raw_tensor_scales,
                    )
                    kv_caches[layer_name] = kv_cache
                    kv_scale_caches[layer_name] = kv_cache_scale
                    # <<< iluvatar specific logic

                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    page_size_bytes = kv_cache_spec.page_size_bytes
                    # Hold a single contiguous [num_blocks, 1, 1, page_size_bytes]
                    # int8 page view per layer; the layer's bind_kv_cache unpacks
                    # each block's bytes into its conv/ssm state views. Keeping
                    # one tensor per layer lets the KV connector register it
                    # without special-casing Mamba.
                    kv_caches[layer_name] = raw_tensor[
                        : num_blocks * page_size_bytes
                    ].view(num_blocks, 1, 1, page_size_bytes)
                else:
                    raise NotImplementedError

        # Reconcile divergent KV layouts to blocks-first. Triggered by hybrid
        # attention/mamba models, and by encoder-decoder models whose shared
        # decoder/cross-attention allocation mixes K/V-first and blocks-first
        # backends (see _has_mixed_attention_kv_layout).
        if has_attn and (
            has_mamba or self._has_mixed_attention_kv_layout(kernel_block_sizes)
        ):
            self._update_hybrid_attention_mamba_layout(kv_caches, kernel_block_sizes)

        return kv_caches, kv_scale_caches

    def _init_kv_zero_meta(self) -> None:
        hidden = set(getattr(self, "_kv_cache_int8_scales", {}))
        if not hidden:
            return super()._init_kv_zero_meta()
        saved = self.runner_only_attn_layers
        self.runner_only_attn_layers = saved | hidden
        try:
            super()._init_kv_zero_meta()
        finally:
            self.runner_only_attn_layers = saved

    @staticmethod
    def use_uniform_kv_cache(*args, **kwargs) -> bool:
        return False

    def _cleanup_profiling_kv_cache(self) -> None:
        # NOTE: this function will be called when determine_available_memory or shutdown are called.
        # TODO: invalid device context happened here.
        # torch.accelerator.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        if hasattr(self, "kv_scale_caches"):
            for i in range(len(self.kv_scale_caches)):
                self.kv_scale_caches[i] = None  # type: ignore[assignment]
            self.kv_scale_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None

        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            if hasattr(layer, "kv_scale_caches"):
                kv_scale = layer.kv_scale_caches
                layer.kv_scale_caches = (
                    torch.tensor([]) if isinstance(kv_scale, torch.Tensor) else []
                )
            # Clean up quantized KV cache scale views
            # (int8_per_token_head, fp8_per_token_head)
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None

        gc.collect()
        # invalid device context
        # torch.accelerator.empty_cache()
        logger.debug("Cleaned up profiling KV cache and CUDA graphs")
