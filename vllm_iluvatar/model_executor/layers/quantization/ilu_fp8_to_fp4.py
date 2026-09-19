# SPDX-License-Identifier: Apache-2.0
"""Source-layout dispatch for FP8 checkpoint shards -> ix30 FP4 weights.

Corresponds to: no community module (Iluvatar-only conversion step).

Target   : the ``weights.source`` contract of an ``additional_config.ilu_quant``
           group whose ``loading`` is ``transform`` @ ``weight_load``
           (see ``ilu_runtime_quant_config.py``).
Baseline : main_9c12a81ac
Why      : ``IluFp8ToFp4OnlineMoEMethod`` converts one checkpoint shard at a
           time and must not hard-code "the source is DeepSeek block-FP8". The
           same ix30 target is reachable from per-channel and per-tensor FP8
           (they are just ``block_shape=[1, K]`` / ``[N, K]``), and later from
           ``e5m2`` weights or ``e8m0`` scales. Keeping that surface in one
           dataclass means the loader never grows format branches.
How      : ``Fp8SourceLayout`` captures the checkpoint side; ``convert_to_ix30_fp4``
           validates it against the implemented set and forwards to ixformer's
           ``preprocess_fp8_block_weight_for_ixmm``. **No numerics live here** --
           the reference implementation, the CUDA kernel and the backend choice
           are all ixformer's (``ixformer/inference/functions/fp4/gemm/
           fp8_block_preprocess.py``), so the load-time path and any future
           bulk path cannot drift apart.
Scope    : block-FP8 ``float8_e4m3fn`` weights with multiplicative float32
           block scales, the only layout GLM-5.2 / DeepSeek-V3 style
           checkpoints use today.
Remove   : never as a whole; individual branches go away if ixformer grows a
           single op that takes the source layout as a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Formats the dispatcher knows how to name. Everything outside
# ``_IMPLEMENTED_FORMATS`` is rejected with the supported list, never silently
# treated as block-FP8.
_KNOWN_FORMATS = ("fp8-block", "fp8-channel", "fp8-tensor")
_IMPLEMENTED_FORMATS = ("fp8-block",)

_SUPPORTED_WEIGHT_DTYPES = (torch.float8_e4m3fn,)
_SUPPORTED_SCALE_DTYPES = (torch.float32,)

IX30_GROUP_SIZE = 256


@dataclass(frozen=True)
class Fp8SourceLayout:
    """How one FP8 checkpoint tensor and its scale relate to each other.

    ``scale_semantics`` follows the checkpoint's own convention: DeepSeek-style
    ``weight_scale_inv`` is ``"mul"`` (``w = q * s``) despite the ``_inv`` name.
    """

    format: str
    block_shape: tuple[int, int]
    scale_semantics: str = "mul"
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.format not in _KNOWN_FORMATS:
            raise ValueError(
                f"unknown FP8 source format {self.format!r}; "
                f"known formats: {list(_KNOWN_FORMATS)}"
            )
        if len(self.block_shape) != 2 or any(b <= 0 for b in self.block_shape):
            raise ValueError(
                f"block_shape must be two positive ints, got {self.block_shape!r}"
            )

    def validate_implemented(self) -> None:
        """Fail fast on layouts that are describable but not yet converted."""
        if self.format not in _IMPLEMENTED_FORMATS:
            raise NotImplementedError(
                f"FP8 source format {self.format!r} is not implemented; "
                f"implemented: {list(_IMPLEMENTED_FORMATS)}. "
                f"(fp8-channel is block_shape=[1, K] and fp8-tensor is [N, K]; "
                f"the ix30 conversion already handles a generic block shape, so "
                f"enabling them is a schema change plus tests, not a new kernel.)"
            )
        if self.scale_semantics != "mul":
            raise NotImplementedError(
                f"scale_semantics={self.scale_semantics!r} is not implemented; "
                f"only 'mul' (w = q * s, DeepSeek weight_scale_inv) is supported"
            )
        if self.weight_dtype not in _SUPPORTED_WEIGHT_DTYPES:
            raise NotImplementedError(
                f"weight dtype {self.weight_dtype} is not implemented; "
                f"supported: {list(_SUPPORTED_WEIGHT_DTYPES)}"
            )
        if self.scale_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise NotImplementedError(
                f"scale dtype {self.scale_dtype} is not implemented; "
                f"supported: {list(_SUPPORTED_SCALE_DTYPES)}"
            )


def convert_to_ix30_fp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    source: Fp8SourceLayout,
    out_weight: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[E, n, k]`` FP8 + block scale to the ix30 FP4 pair.

    Args:
        weight:       ``[E, n, k]`` in ``source.weight_dtype``.
        weight_scale: ``[E, ceil(n/bn), ceil(k/bk)]`` in ``source.scale_dtype``.
        out_weight:   optional ``[E, kp/2, n]`` uint8 destination. May be a
                      narrowed (non-contiguous) view of a resident parameter;
                      ixformer's facade handles the placement.
        out_scale:    optional ``[E, kp/256, n]`` float32 running-acc destination.
        backend:      ixformer registry backend name (``"torch"`` /
                      ``"ixformer_native"``); ``None`` lets the registry choose
                      and fall back for unsupported shapes.

    Returns:
        ``(out_weight, out_scale)`` -- the caller's tensors when provided.
    """
    source.validate_implemented()

    if weight.dtype is not source.weight_dtype:
        raise ValueError(
            f"convert_to_ix30_fp4: weight dtype {weight.dtype} does not match "
            f"the declared source dtype {source.weight_dtype}"
        )
    if weight_scale.dtype is not source.scale_dtype:
        raise ValueError(
            f"convert_to_ix30_fp4: scale dtype {weight_scale.dtype} does not "
            f"match the declared source dtype {source.scale_dtype}"
        )

    # Lazy import keeps this module importable without ixformer (CPU-only unit
    # tests), matching the NVFP4/MXFP4 methods' lazy-import discipline.
    import ixformer.functions as ixfops

    return ixfops.preprocess_fp8_block_weight_for_ixmm(
        weight,
        weight_scale,
        source.block_shape,
        out_weight=out_weight,
        out_scale=out_scale,
        scale_format="running_acc",
        backend=backend,
    )


def ix30_fp4_shapes(n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """``((kp/2, n), (kp/256, n))`` for a single expert's ``[n, k]`` shard.

    ``kp = align(k, 256)``: the ix30 group is 256 elements along K, and a
    trailing partial group is zero padded (the compute path pads the activation
    to match, so the added columns contribute exactly zero).
    """
    padded_k = (k + IX30_GROUP_SIZE - 1) // IX30_GROUP_SIZE * IX30_GROUP_SIZE
    return (padded_k // 2, n), (padded_k // IX30_GROUP_SIZE, n)
