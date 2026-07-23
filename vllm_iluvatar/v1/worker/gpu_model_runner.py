# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPU model runner — KVINT8 KV cache support.

Subclasses the community GPUModelRunner to override KV cache tensor
reshaping and scale binding for the KVINT8 attention quantisation scheme.

Only active when ``VLLM_ILU_FLASH_ATTN_QUANT_TYPE=KVINT8``.
Injected via ``IluWorker.init_device()`` (temporary replacement during init).
"""

from __future__ import annotations

import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_iluvatar.v1.attention.static_quant_utils import (
    KV_NUM_PLANES,
    int8_page_size_bytes,
    load_static_v_scales,
)
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)


class IluvatarGPUModelRunner(GPUModelRunner):

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kv_cache_scale_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> tuple[dict, dict]:
        import sys
        print(
            f"[KVINT8] _reshape_kv_cache_tensors called, "
            f"layers={len(kv_cache_raw_tensors)}, "
            f"block_sizes={kernel_block_sizes}",
            file=sys.stderr, flush=True,
        )
        kv_caches: dict[str, torch.Tensor] = {}
        self._kv_cache_int8_scales: dict[str, dict] = {}
        static_v_scales: dict[int, torch.Tensor] | None = None
        attn_layer_names: set[str] = set()

        groups = list(self._kv_cache_spec_attn_group_iterator())
        for group in groups:
            spec = group.kv_cache_spec
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            if not isinstance(spec, AttentionSpec):
                continue

            attn_layer_names.update(group.layer_names)
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            num_heads = spec.num_kv_heads
            head_size = spec.head_size

            if static_v_scales is None:
                static_v_scales = load_static_v_scales(
                    num_heads, head_size, self.device
                )
                print(
                    f"[KVINT8] loaded v_scales: {len(static_v_scales)} layers, "
                    f"keys={sorted(static_v_scales.keys())[:5]}..., "
                    f"sample shape={next(iter(static_v_scales.values())).shape}",
                    file=sys.stderr, flush=True,
                )

            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                raw_tensor = kv_cache_raw_tensors[layer_name]
                page_bytes = int8_page_size_bytes(spec)
                assert raw_tensor.numel() % page_bytes == 0
                num_blocks = raw_tensor.numel() // page_bytes
                kernel_num_blocks = num_blocks * (
                    spec.block_size // kernel_block_size
                )
                plane = num_heads * kernel_block_size * head_size

                main_bytes = KV_NUM_PLANES * kernel_num_blocks * plane
                assert main_bytes <= raw_tensor.numel()
                planes = raw_tensor[:main_bytes].view(
                    KV_NUM_PLANES,
                    kernel_num_blocks,
                    num_heads,
                    kernel_block_size,
                    head_size,
                )
                kv_caches[layer_name] = planes

                k_scale_cache = torch.zeros(
                    kernel_num_blocks,
                    num_heads,
                    kernel_block_size,
                    dtype=torch.float32,
                    device=raw_tensor.device,
                )

                v_scale = None
                try:
                    layer_idx = extract_layer_index(layer_name)
                except (AssertionError, ValueError):
                    layer_idx = -1
                if layer_idx not in static_v_scales:
                    fallback_idx = max(static_v_scales)
                    logger.warning(
                        "[KVINT8] layer %s (idx=%d) not in v_scale file, "
                        "fallback to layer_%d",
                        layer_name, layer_idx, fallback_idx,
                    )
                    layer_idx = fallback_idx
                v_scale = static_v_scales[layer_idx]

                print(
                    f"[KVINT8] {layer_name}: planes={list(planes.shape)}, "
                    f"k_scale={list(k_scale_cache.shape)}, "
                    f"v_scale(layer_{layer_idx}) "
                    f"min={v_scale.min().item():.6f} "
                    f"max={v_scale.max().item():.6f} "
                    f"mean={v_scale.mean().item():.6f}",
                    file=sys.stderr, flush=True,
                )

                self._kv_cache_int8_scales[layer_name] = {
                    "key_cache": planes[0],
                    "value_cache": planes[1],
                    "k_scale_cache": k_scale_cache,
                    "v_scale": v_scale,
                }

        non_attn_raw = {
            n: t
            for n, t in kv_cache_raw_tensors.items()
            if n not in attn_layer_names
        }
        if non_attn_raw:
            saved = self.runner_only_attn_layers
            self.runner_only_attn_layers = saved | attn_layer_names
            try:
                non_attn_caches, _ = super()._reshape_kv_cache_tensors(
                    kv_cache_raw_tensors,
                    kv_cache_scale_raw_tensors,
                    kernel_block_sizes,
                )
                kv_caches.update(non_attn_caches)
            finally:
                self.runner_only_attn_layers = saved

        return kv_caches, {}

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> tuple[dict, dict]:
        kv_caches, kv_caches_scale = super().initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        scales = getattr(self, "_kv_cache_int8_scales", {})
        ctx = self.compilation_config.static_forward_context
        for layer_name, scale in scales.items():
            if layer_name in ctx:
                ctx[layer_name].kv_cache_scale = scale
        for layer_name, target in self.shared_kv_cache_layers.items():
            if layer_name in ctx and target in scales:
                ctx[layer_name].kv_cache_scale = scales[target]

        import sys
        bound = [n for n in scales if n in ctx]
        missed = [n for n in scales if n not in ctx]
        print(
            f"[KVINT8] initialize_kv_cache_tensors: "
            f"bound={len(bound)} layers, missed={len(missed)}, "
            f"shared_kv={len(self.shared_kv_cache_layers)}",
            file=sys.stderr, flush=True,
        )
        if missed:
            print(
                f"[KVINT8] WARNING missed layers: {missed[:5]}",
                file=sys.stderr, flush=True,
            )
        return kv_caches, kv_caches_scale

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
