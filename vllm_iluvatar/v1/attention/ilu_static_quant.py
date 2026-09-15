# SPDX-License-Identifier: Apache-2.0
"""Static scale utilities for Iluvatar flash-attention quantization.

Provides V-scale calibration used by the ilu_flash_attn backend (ilu_flash_attn.py).
Provides V-scale loading used by the model runner (gpu_model_runner.py).
Also provides per-layer Q/K/V per-tensor FP8 calibration and loading.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

_FP8_SCALE_NAMES = ("q", "k", "v")
_fp8_scale_file_cache: dict[str, dict[str, torch.Tensor]] = {}


class StaticQuantManager:
    init_layer_index_count: dict[str, int] = {}
    init_layer_index_count_backup: list[str] = []
    runtime_scales: dict[str, torch.Tensor] = {}

    def __init__(
        self,
        total_num_kv_heads: int,
        layer_id: str,
        shape: tuple,
        dtype: torch.dtype,
        file_save_path: str,
        device: str = None,
        tp_size: int = None,
        tp_rank: int = None,
        save_step: int = 100,
        info_step: int = 100,
    ):
        if layer_id not in StaticQuantManager.init_layer_index_count:
            StaticQuantManager.init_layer_index_count[layer_id] = 1
            StaticQuantManager.init_layer_index_count_backup.append(layer_id)
        else:
            raise RuntimeError(
                f"layer_id already record in StaticQuantManager, please check the layer_id: {layer_id}"
            )
        # update parament
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()

        if total_num_kv_heads < tp_size:
            self.kv_index = []
            repeat = tp_size // total_num_kv_heads
            ranges = (
                torch.arange(0, total_num_kv_heads)
                .view(-1, 1)
                .repeat((1, repeat))
                .flatten()
                .tolist()
            )
            for i in range(total_num_kv_heads):
                self.kv_index.append(ranges.index(i))
        else:
            self.kv_index = None

        if device is None:
            device = "cuda"

        # check parament
        assert file_save_path not in [None, ""], (
            "file_save_path should not be None or empty"
        )

        para_dir = os.path.dirname(file_save_path)
        assert os.path.exists(para_dir), (
            f"StaticQuantManager workdir {para_dir} not exist!"
        )
        assert not os.path.exists(file_save_path), "file_save_path should not exist"

        assert layer_id is not None

        self.operator = tp_rank == 0
        self.file_save_path = file_save_path
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.layer_id = layer_id
        self.total_layer_index: list[str] = []
        self.save_step = save_step
        self.info_step = info_step
        self.update_count = 0
        self.warmup_done = False
        assert len(shape) == 2, "shape should be (num_kv_heads, head_size)"
        self.scales = torch.zeros(shape, dtype=dtype, device=device)
        logger.debug(
            "StaticQuantManager info: tp_rank:%d tp_size:%d layer_id:%s scale shape:%s  self.scales:%s",
            self.tp_rank,
            self.tp_size,
            self.layer_id,
            shape,
            self.scales.device,
        )

    def update_data(self, data):
        if not self.warmup_done:
            warmup_done = all(
                value == 0
                for value in StaticQuantManager.init_layer_index_count.values()
            )
            if warmup_done:
                self.warmup_done = True
                self.total_layer_index = (
                    StaticQuantManager.init_layer_index_count_backup
                )
                self.operator = self.operator and (
                    self.layer_id == self.total_layer_index[-1]
                )
                logger.info(
                    "StaticQuantManager init done, total_layer_index: %s, kv_index: %s",
                    self.total_layer_index,
                    self.kv_index,
                )
            else:
                StaticQuantManager.init_layer_index_count[self.layer_id] -= 1

        self.scales = torch.max(data, self.scales)
        # save file
        self.update_count += 1
        if (self.update_count % self.info_step == 0) and self.operator:
            logger.info("StaticQuantManager run update_data %d step", self.update_count)

        if self.update_count % self.save_step == 0:
            gather_scales = get_tp_group().gather(self.scales, dst=0, dim=-2)
            if gather_scales is not None:
                if self.kv_index is not None:
                    gather_scales = gather_scales[self.kv_index]
                StaticQuantManager.runtime_scales[self.layer_id] = gather_scales.cpu()

                if len(StaticQuantManager.runtime_scales) == len(
                    self.total_layer_index
                ):
                    torch.save(StaticQuantManager.runtime_scales, self.file_save_path)
                    logger.info(
                        "StaticQuantManager save to %s with %d step",
                        self.file_save_path,
                        self.update_count,
                    )
                    StaticQuantManager.runtime_scales = {}


class FP8StaticQuantManager:
    """Collect and persist per-layer, per-tensor Q/K/V FP8 scales."""

    init_layer_index_count: dict[str, int] = {}
    init_layer_index_count_backup: list[str] = []
    runtime_scales: dict[str, torch.Tensor] = {}

    def __init__(
        self,
        layer_id: str,
        file_save_path: str,
        device: torch.device,
        save_step: int = 100,
        info_step: int = 100,
    ) -> None:
        if layer_id in FP8StaticQuantManager.init_layer_index_count:
            raise RuntimeError(
                f"layer_id already recorded in FP8StaticQuantManager: {layer_id}"
            )
        FP8StaticQuantManager.init_layer_index_count[layer_id] = 1
        FP8StaticQuantManager.init_layer_index_count_backup.append(layer_id)

        assert file_save_path, "file_save_path should not be empty"
        para_dir = os.path.dirname(file_save_path) or "."
        assert os.path.exists(para_dir), (
            f"FP8StaticQuantManager workdir {para_dir} does not exist"
        )
        assert not os.path.exists(file_save_path), "file_save_path should not exist"

        tp_group = get_tp_group()
        self.operator = tp_group.rank_in_group == 0
        self.file_save_path = file_save_path
        self.layer_id = layer_id
        self.save_step = save_step
        self.info_step = info_step
        self.update_count = 0
        self.warmup_done = False
        self.total_layer_index: list[str] = []
        self.scales = torch.zeros(3, dtype=torch.float32, device=device)

    def update_data(self, scales: torch.Tensor) -> None:
        """Update running maxima from ``[q_scale, k_scale, v_scale]``."""
        assert scales.shape == (3,) and scales.dtype == torch.float32
        scales = scales.detach().clone()

        if not self.warmup_done:
            warmup_done = all(
                value == 0
                for value in FP8StaticQuantManager.init_layer_index_count.values()
            )
            if warmup_done:
                self.warmup_done = True
                self.total_layer_index = (
                    FP8StaticQuantManager.init_layer_index_count_backup
                )
                self.operator = self.operator and (
                    self.layer_id == self.total_layer_index[-1]
                )
                logger.info(
                    "FP8StaticQuantManager init done, layers: %s",
                    self.total_layer_index,
                )
            else:
                FP8StaticQuantManager.init_layer_index_count[self.layer_id] -= 1
                return

        self.scales = torch.max(self.scales.unsqueeze(0), scales.unsqueeze(0)).squeeze(
            0
        )
        self.update_count += 1
        if self.update_count % self.info_step == 0 and self.operator:
            logger.info("FP8StaticQuantManager collected %d steps", self.update_count)

        if self.update_count % self.save_step != 0:
            return
        tp_group = get_tp_group()
        reduced_scales = self.scales.clone()
        if tp_group.world_size > 1:
            dist.all_reduce(
                reduced_scales,
                op=dist.ReduceOp.MAX,
                group=tp_group.device_group,
            )
        if tp_group.rank_in_group == 0:
            FP8StaticQuantManager.runtime_scales[self.layer_id] = reduced_scales.cpu()
        if self.operator and len(FP8StaticQuantManager.runtime_scales) == len(
            self.total_layer_index
        ):
            torch.save(
                FP8StaticQuantManager.runtime_scales,
                self.file_save_path,
            )
            logger.info(
                "FP8StaticQuantManager saved Q/K/V scales to %s at step %d",
                self.file_save_path,
                self.update_count,
            )
            FP8StaticQuantManager.runtime_scales = {}


def load_static_fp8_qkv_scales(
    layer_idx: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one layer's per-tensor Q/K/V FP8 dequantization scales."""
    scale_path = envs.VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH
    if not scale_path or not os.path.isfile(scale_path):
        raise ValueError(
            "FP8 attention requires a calibrated scale file, got "
            f"VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH={scale_path!r}"
        )
    if scale_path not in _fp8_scale_file_cache:
        raw = torch.load(scale_path, map_location="cpu")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Invalid FP8 attention scale file: {scale_path}")
        _fp8_scale_file_cache[scale_path] = raw

    raw_scales = _fp8_scale_file_cache[scale_path]
    if layer_idx not in raw_scales:
        raise KeyError(
            f"Layer {layer_idx!r} is missing from FP8 scale file {scale_path}"
        )
    scales = raw_scales[layer_idx].float().flatten()
    if scales.numel() != 3 or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(
            f"Layer {layer_idx!r} must contain three positive finite "
            f"Q/K/V scales, got {scales}"
        )
    scales = scales.to(device=device).contiguous()
    logger.info_once(
        "[FP8 attention] loaded %s scales for layer %s: %s",
        "/".join(_FP8_SCALE_NAMES),
        layer_idx,
        tuple(scales.tolist()),
        scope=f"fp8-attention-layer-{layer_idx}",
    )
    q_scale, k_scale, v_scale = scales.unbind()
    return q_scale.reshape(1), k_scale.reshape(1), v_scale.reshape(1)


def load_static_v_scales(
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
    layer_idx: str | None = None,
) -> dict[str, torch.Tensor]:
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
        scale_path,
        len(raw),
        list(raw[next(iter(raw))].shape),
        raw_num_kv_heads,
        num_kv_heads,
        head_size,
        tp_rank,
        tp_world,
        tp_slices,
    )

    scales: dict[str, torch.Tensor] = {}
    for key, full in raw.items():
        part = full.float().view(tp_slices, num_kv_heads, head_size)[
            tp_rank % tp_slices
        ]
        scales[key] = part.contiguous().to(device)

    back = False
    if layer_idx is not None:
        try:
            sample_idx = layer_idx
            sample = scales[sample_idx]
        except Exception:
            back = True
    if back:
        sample_idx = next(iter(scales))
        sample = scales[sample_idx]
    logger.info(
        "[KVINT8 v_scale] loaded %d layers, sample layer: %s: "
        "shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
        len(scales),
        sample_idx,
        list(sample.shape),
        sample.dtype,
        sample.min().item(),
        sample.max().item(),
        sample.mean().item(),
    )
    return scales
