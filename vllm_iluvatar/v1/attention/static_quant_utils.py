# SPDX-License-Identifier: Apache-2.0
"""KV cache int8 quantization utilities for KVINT8 attention.

Provides page-size helpers and V-scale loading used by both the KV cache
patch (patch_kv_cache_int8.py) and the model runner (gpu_model_runner.py).
"""

from __future__ import annotations

import os

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

KV_NUM_PLANES = 2


def is_kvint8_enabled() -> bool:
    return envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8"


def int8_page_size_bytes(spec: AttentionSpec) -> int:
    """Main int8 KV buffer size per block (int8 planes only).

    The fp32 K scale lives in a separate buffer.
    """
    return KV_NUM_PLANES * spec.block_size * spec.num_kv_heads * spec.head_size


def int8_scale_page_bytes(spec: AttentionSpec) -> int:
    """fp32 K scale buffer size per block: one value per (block_token, head)."""
    return spec.block_size * spec.num_kv_heads * 4


def load_static_v_scales(
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Load static V cache scales from the calibration file.

    Returns {layer_idx: [num_kv_heads, head_size] fp32 scale tensor}.
    Layer indices may be sparse for hybrid (linear + full attention) models.
    """
    scale_path = envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH
    if not scale_path or not os.path.isfile(scale_path):
        raise ValueError(
            f"VLLM_ILU_FLASH_ATTN_QUANT_TYPE=KVINT8 requires a calibrated V "
            f"scale file, got VLLM_ILU_FLASH_ATTN_V_SCALE_PATH={scale_path!r}"
        )
    raw = torch.load(scale_path, map_location="cpu")
    tp_rank = get_tp_group().rank_in_group
    tp_world = get_tp_group().world_size
    raw_num_kv_heads = raw[next(iter(raw))].shape[0]
    tp_slices = raw_num_kv_heads // num_kv_heads

    logger.info(
        "[KVINT8 v_scale] path=%s, num_layers=%d, raw_shape=%s, "
        "raw_num_kv_heads=%d, num_kv_heads=%d, head_size=%d, "
        "tp_rank=%d/%d, tp_slices=%d",
        scale_path, len(raw),
        list(raw[next(iter(raw))].shape),
        raw_num_kv_heads, num_kv_heads, head_size,
        tp_rank, tp_world, tp_slices,
    )

    scales: dict[int, torch.Tensor] = {}
    for key, full in raw.items():
        idx = int(key.rsplit("_", 1)[1])
        part = full.float().view(tp_slices, num_kv_heads, head_size)[
            tp_rank % tp_slices
        ]
        scales[idx] = part.contiguous().to(device)

    sample_idx = next(iter(scales))
    sample = scales[sample_idx]
    logger.info(
        "[KVINT8 v_scale] loaded %d layers, sample layer_%d: "
        "shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
        len(scales), sample_idx, list(sample.shape), sample.dtype,
        sample.min().item(), sample.max().item(), sample.mean().item(),
    )
    return scales
