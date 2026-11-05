# SPDX-License-Identifier: Apache-2.0
# Source: ilu_flash_attn.py at 47fb804; independent INT8 cache writer binding.
"""INT8 quant and cache writer entry points for the ixaite attention path.

Bindings never substitute ``torch.ops._C``; baseline keeps the native writers.

KINT8 binds INT8 K/BF16 V and token/head FP32 K scales; KVINT8 binds INT8 K/V
plus immutable layer/head/channel FP32 V scales. Pages are zero-copy HND views;
the framework owns allocation, valid unique slots and lifetime. Block resets
and relocation update data and K scales together, including shared backing.

VLLM_ILU_FLASH_ATTN_V_SCALE_PATH supplies the shared ixaite calibration format:
``torch.save({"global layer index": FP32[global KV heads, D]})``. Scales must be
finite and positive and map to the requested layer and TP shard or replica.
Do not overwrite a loaded file or mutate a scale bound to a live cache.
Validated for ivcore11, ordinary BF16 MHA/GQA, HND page=16 and D=128/256.
The matching Attention backend owns reads and any fused decode KV writes.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy

import torch

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op

_RESHAPE_WITH_SCALE_IMPL: Callable[..., object] | None = None
_RESHAPE_MIX_IMPL: Callable[..., object] | None = None
_QUANT_IMPL: Callable[..., object] | None = None


def configure_int8_kv_cache_writer_providers() -> None:
    """Bind both writer contracts without ever substituting the plugin native op."""
    global _RESHAPE_WITH_SCALE_IMPL, _RESHAPE_MIX_IMPL
    implementations = []
    for name in ("reshape_and_cache_with_scale", "reshape_and_cache_mix"):
        implementation, reason = load_ixaite_python_op(name, name)
        if implementation is None:
            raise RuntimeError(f"ixaite {name} is unavailable: {reason}")
        implementations.append(implementation)
    _RESHAPE_WITH_SCALE_IMPL, _RESHAPE_MIX_IMPL = implementations


def configure_int8_attention_quant_provider() -> None:
    """Bind ``scaled_int8_quant_for_attn`` for KINT8/KVINT8 forwards."""
    global _QUANT_IMPL
    implementation, reason = load_ixaite_python_op(
        "scaled_int8_quant_for_attn",
        "scaled_int8_quant_for_attn",
    )
    if implementation is None:
        raise RuntimeError(
            f"ixaite scaled_int8_quant_for_attn is unavailable: {reason}"
        )
    _QUANT_IMPL = implementation


def configure_int8_attention_providers() -> None:
    """Bind INT8 quant and both cache writers as one init contract."""
    configure_int8_attention_quant_provider()
    configure_int8_kv_cache_writer_providers()


def scaled_int8_quant_for_attn(
    input: torch.Tensor,
    quant_dim: int,
    transpose_scale: bool = False,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _QUANT_IMPL is None:
        raise RuntimeError(
            "ixaite INT8 attention quant is not configured; "
            "call configure_int8_attention_providers() first"
        )
    return _QUANT_IMPL(input, quant_dim, transpose_scale, scale)


def reshape_and_cache_with_scale(*args, **kwargs) -> None:
    if _RESHAPE_WITH_SCALE_IMPL is None:
        raise RuntimeError(
            "ixaite reshape_and_cache_with_scale is not configured; "
            "call configure_int8_attention_providers() first"
        )
    _RESHAPE_WITH_SCALE_IMPL(*args, **kwargs)


def reshape_and_cache_mix(*args, **kwargs) -> None:
    if _RESHAPE_MIX_IMPL is None:
        raise RuntimeError(
            "ixaite reshape_and_cache_mix is not configured; "
            "call configure_int8_attention_providers() first"
        )
    _RESHAPE_MIX_IMPL(*args, **kwargs)


# Keep baseline's loader unchanged while sharing the calibrated file format.
def load_ixaite_static_v_scales(heads, dim, device, layer_idx):
    import vllm.envs as envs
    from ixaite.utility.static_v_scale import load_static_v_scale
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    return {
        layer_idx: load_static_v_scale(
            envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH,
            layer_idx,
            heads,
            dim,
            tp.rank_in_group,
            tp.world_size,
            device,
        )
    }


def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
    """Reserve fixed V-scale bytes before sizing data and per-block K scales."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_iluvatar.v1.core import kv_cache_utils as common

    config = common.origin_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    )
    blocks = config.num_blocks
    if blocks == 0:
        return config

    specs = {}
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        specs.update(
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else dict.fromkeys(group.layer_names, spec)
        )
    scale_specs = [
        common._get_shared_scale_kv_spec(tensor.shared_by, specs)
        for tensor in config.kv_cache_tensors
    ]
    data_bytes = sum(tensor.size for tensor in config.kv_cache_tensors)
    assert data_bytes <= available_memory or available_memory == 0
    new_blocks = blocks
    if available_memory != 0:
        fixed = sum(spec.fixed_size_bytes for spec in scale_specs if spec is not None)
        scale_per_block = sum(
            spec.scale_page_size_bytes for spec in scale_specs if spec is not None
        )
        per_block = data_bytes // blocks + scale_per_block
        new_blocks = min(blocks, max(0, (available_memory - fixed) // per_block))
        if new_blocks == 0:
            raise ValueError(
                "KV cache budget cannot hold one data/scale block and static V scales"
            )
    for tensor, spec in zip(config.kv_cache_tensors, scale_specs):
        assert tensor.size % blocks == 0
        tensor.size = tensor.size // blocks * new_blocks
        if spec is not None:
            tensor.scale_size = spec.scale_page_size_bytes * new_blocks
            tensor.fixed_size = spec.fixed_size_bytes
    config.num_blocks = new_blocks
    return config


# Cache data, dynamic scales and framework block IDs must move as one unit.
def apply_int8_cache_updates(runner, scheduler_output) -> None:
    copies = scheduler_output.kv_cache_block_copies
    zero_ids = scheduler_output.new_block_ids_to_zero
    if not copies and not zero_ids:
        return
    # Layers with matching geometry can share indices within this scheduler step.
    zero_pages, copy_slots = {}, {}
    for cache in runner._ixaite_int8_caches:
        pages_per_block, remainder = divmod(
            cache.key.shape[0], runner.kv_cache_config.num_blocks
        )
        if remainder:
            raise ValueError("INT8 cache pages must divide into framework blocks")
        device = cache.key.device
        if zero_ids:
            layout = (device, pages_per_block)
            if layout not in zero_pages:
                ids = torch.tensor(zero_ids, device=device, dtype=torch.int64)
                zero_pages[layout] = (
                    ids[:, None] * pages_per_block
                    + torch.arange(pages_per_block, device=device)
                ).reshape(-1)
            pages = zero_pages[layout]
            for tensor in (cache.key, cache.value, cache.key_scale):
                tensor.index_fill_(0, pages, 0)
        if copies:
            span = pages_per_block * cache.key.shape[2]
            layout = (device, span)
            if layout not in copy_slots:
                indices = torch.tensor(copies, device=device, dtype=torch.int64)
                offsets = torch.arange(span, device=device)
                source = (indices[:, 0, None] * span + offsets).reshape(-1)
                destination = (indices[:, 1, None] * span + offsets).reshape(-1)
                copy_slots[layout] = (destination, source)
            cache.copy_slots(*copy_slots[layout])


# Bind views after allocation, and preserve the binding when the runner class is restored.
def bind_int8_kv_caches(runner) -> None:
    import vllm.envs as envs

    mode = envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE
    if not mode:
        return
    if mode not in ("KINT8", "KVINT8"):
        raise ValueError(f"unsupported ixaite KV cache mode: {mode!r}")
    from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
        _get_k_int8_cache_views,
    )

    binding, reason = load_ixaite_python_op("attention.int8_kv_cache", "Int8KVCache")
    if binding is None:
        raise RuntimeError(f"ixaite INT8 KV cache is unavailable: {reason}")
    caches = {}
    for layer_name, layer in runner.compilation_config.static_forward_context.items():
        data = getattr(layer, "kv_cache", None)
        scales = getattr(layer, "kv_cache_scale", None)
        if not isinstance(data, torch.Tensor) or data.numel() == 0:
            continue
        if (
            data.ndim != 5
            or data.shape[0] != (3 if mode == "KINT8" else 2)
            or scales is None
        ):
            raise ValueError(
                "ixaite INT8 cache binding requires ordinary MHA/GQA pages and K scales"
            )
        key, value = (
            _get_k_int8_cache_views(data, torch.bfloat16)
            if mode == "KINT8"
            else data.unbind(0)
        )
        identity = (key.data_ptr(), value.data_ptr(), scales[0].data_ptr())
        if identity not in caches:
            cache = binding(
                key, value, scales[0], scales[1] if mode == "KVINT8" else None
            )
            if mode == "KVINT8" and hasattr(runner, "model_config"):
                from ixaite.utility.static_v_scale import load_static_v_scale
                from vllm.distributed.parallel_state import get_tp_group
                from vllm.model_executor.models.utils import extract_layer_index

                # A shared alias can precede its target in the forward context.
                target = getattr(runner, "shared_kv_cache_layers", {}).get(
                    layer_name, layer_name
                )
                tp = get_tp_group()
                expected = load_static_v_scale(
                    envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH,
                    str(extract_layer_index(target)),
                    key.shape[1],
                    key.shape[3],
                    tp.rank_in_group,
                    tp.world_size,
                    "cpu",
                    total_heads=runner.model_config.get_total_num_kv_heads(),
                )
                if not torch.equal(cache.value_scale.cpu(), expected):
                    raise ValueError(
                        "allocated INT8 KV V scales do not match calibrated values"
                    )
            caches[identity] = cache
        elif mode == "KVINT8" and not torch.equal(
            caches[identity].value_scale, scales[1]
        ):
            raise ValueError("shared INT8 KV data must use the same static V scales")
        layer._ixaite_int8_cache = caches[identity]
    if not caches:
        raise ValueError("no MHA/GQA INT8 KV caches were allocated")
    runner._ixaite_int8_caches = tuple(caches.values())
    if hasattr(runner, "_ixaite_original_update_states"):
        return
    runner._ixaite_original_update_states = runner._update_states

    # The host byte-copy assumes block-major storage; KINT8 can be planes-first.
    def update_states(output):
        apply_int8_cache_updates(runner, output)
        without_cache_updates = copy(output)
        without_cache_updates.new_block_ids_to_zero = []
        without_cache_updates.kv_cache_block_copies = []
        return runner._ixaite_original_update_states(without_cache_updates)

    runner._update_states = update_states
