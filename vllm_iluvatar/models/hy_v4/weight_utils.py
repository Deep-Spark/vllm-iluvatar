# SPDX-License-Identifier: Apache-2.0
"""Load-time dequant for the HY V4 indexer ``wk`` projection.

Corresponds to: vllm/model_executor/models/deepseek_v2.py::_try_load_fp8_indexer_wk

Target   : the community helper of that name, which HY V4's ``load_weights``
           would otherwise call.
Baseline : community derives one block size for *both* dims
           (``GroupShape(block, block)`` with ``block = in / scale_cols``) and
           feeds ``scale_inv`` to ``scaled_dequantize`` unchanged.
Why      : Hy4-preview stores ``indexer.wk.weight`` as MXFP8 --
           ``[128, 6144]`` e4m3 plus a ``[128, 192]`` **uint8 E8M0** scale, i.e.
           a ``(1, 32)`` block. Community would compute ``6144 / 192 = 32`` and
           then use ``GroupShape(32, 32)``, claiming 32 rows share a scale when
           each row has its own, and would hand ``scaled_dequantize`` a uint8
           tensor whose bytes are exponents rather than floats. Both are silent
           corruption, not an error. Upstream #54160 fixed the community helper;
           the Iluvatar baseline must stay byte-identical, so the fix lives here.
How      : derive the block size per dimension, and decode E8M0 exponent bytes
           to fp32 before dequantizing. The E8M0 decode is the same one the
           MXFP8 MoE path uses, so both share ``_e8m0_to_fp32``.
Scope    : HY V4 checkpoints whose indexer ``wk`` is quantized (Hy4-preview
           MXFP8). BF16 ``wk`` never reaches here.
Remove   : when the Iluvatar vLLM baseline picks up the upstream helper.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    _e8m0_to_fp32 as e8m0_to_fp32,
)


def try_load_quantized_indexer_wk(
    name: str,
    tensor: torch.Tensor,
    buf: dict[str, dict[str, torch.Tensor]],
    params_dict: dict,
    loaded_params: set[str],
    pp_missing_layer_names: list[str],
) -> bool:
    """Buffer and dequantize a quantized indexer ``wk`` weight/scale pair.

    ``wk`` and ``weights_proj`` are fused into one GEMM, but the checkpoint
    quantizes only ``wk``. Upcasting it to BF16 while loading keeps the fusion,
    so the fused parameter stays unquantized and no MXFP8 kernel is needed for
    a ``[128 + n_head, hidden]`` projection.

    Returns:
        True when the tensor was consumed here (buffered, dropped as
        PP-missing, or dequantized and stored); False when it is not a
        quantized isolated ``wk`` tensor and the caller should handle it.
    """
    if "indexer.wk." not in name or "wk_weights" in name:
        return False  # Not an isolated WK weight for the indexer, ignore.
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale" in name
    if not is_weight and not is_scale:
        return False  # WK is not quantized, ignore.

    # Buffer this tensor (weight or scale) until both have arrived.
    layer_prefix = name.rsplit(".wk.", 1)[0]  # "model.layers.0.self_attn.indexer"
    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    if any(
        name.startswith(missing_layer_name)
        for missing_layer_name in pp_missing_layer_names
    ):
        return True
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True  # still waiting for the other param

    weight_fp8, scale = entry["weight"], entry["scale"]
    del buf[layer_prefix]

    # E8M0 scales are power-of-two exponents stored as raw bytes; decode before
    # dequantizing. ``scaled_dequantize`` multiplies, so a uint8 exponent would
    # otherwise be used as a literal multiplier.
    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
        scale = e8m0_to_fp32(scale)

    if scale.ndim == 1:
        # Per-channel scale: one scale per row of [out, in].
        group_shape = GroupShape(1, weight_fp8.shape[1])
    else:
        # Per dimension, so per-channel ([out, 1]), MX ([out, in / 32]) and 2D
        # block ([out / 128, in / 128]) layouts are all described correctly.
        group_shape = GroupShape(
            weight_fp8.shape[0] // scale.shape[0],
            weight_fp8.shape[1] // scale.shape[1],
        )
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale,
        group_shape=group_shape,
        out_dtype=torch.bfloat16,
    )

    # Load the dequantized weight into shard 0 of the fused buffer.
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True
