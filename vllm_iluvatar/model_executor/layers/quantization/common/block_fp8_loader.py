# SPDX-License-Identifier: Apache-2.0
"""Per-expert block-FP8 loading into compute-selected final weight contracts."""

from dataclasses import dataclass
from math import gcd

import torch

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeightSpec,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_fp4 import (
    Fp8SourceLayout,
    convert_to_ix30_fp4,
)

from .paired_moe_loader import PairedMoEWeightLoader, _native_converter


@dataclass(frozen=True)
class BlockFp8LoadSpec:
    experts: int
    hidden: int
    intermediate: int
    block_shape: tuple[int, int]
    tp_size: int = 1
    tp_rank: int = 0
    runtime_intermediate: int | None = None
    w13_k_padded: int | None = None
    w2_n_padded: int | None = None
    w2_k_padded: int | None = None


class BlockFp8WeightLoader(PairedMoEWeightLoader):
    """Consume the community w1/w2/w3 callback, irrespective of model namespace.

    No complete layer of source weights is resident. The paired reader retains
    deferred checkpoint views; one local projection is sliced and uploaded at a
    time. FP32 and BF16 multiplicative source scales share FP32 upload storage.
    """

    source_name = "block-FP8"
    scale_parameter_name = "weight_scale_inv"
    # How the checkpoint encodes the block scale. A source whose scale is not
    # already a multiplicative float overrides these and ``_decode_scale``;
    # staging, TP slicing and the native conversion then stay shared. See
    # ``common/mxfp8_loader.py``.
    source_scale_dtypes = (torch.bfloat16, torch.float32)
    source_scale_metadata = ("F32", "BF16")

    def __init__(self, weight_spec, load_spec):
        self.weight_spec, self.load_spec = weight_spec, load_spec
        self.source = Fp8SourceLayout("fp8-block", load_spec.block_shape)
        self.source.validate_implemented()
        c = load_spec
        if (
            min(c.experts, c.hidden, c.intermediate, *c.block_shape) <= 0
            or not 0 <= c.tp_rank < c.tp_size
            or c.hidden % 32
            or c.intermediate % 32
            or self.w13_half_n < c.intermediate
            or self.w13_k < c.hidden
            or self.w2_n < c.hidden
            or self.w2_k < self.w13_half_n
            or any(
                value % 32
                for value in (self.w13_half_n, self.w13_k, self.w2_n, self.w2_k)
            )
        ):
            raise ValueError(f"Invalid {self.source_name} load geometry")
        self._layer = self._pending = self._upload_buffer = self._load_session = None
        self._loaded = set()
        self._finalized = self._failed = False

    def source_shape(self, group, is_scale):
        c = self.load_spec
        n, k = (
            (c.intermediate * c.tp_size, c.hidden)
            if group == "w13"
            else (c.hidden, c.intermediate * c.tp_size)
        )
        if is_scale:
            bn, bk = c.block_shape
            return ((n + bn - 1) // bn, (k + bk - 1) // bk)
        return (n, k)

    def _local_geometry(self, group):
        c = self.load_spec
        n, k = (
            (c.intermediate, c.hidden) if group == "w13" else (c.hidden, c.intermediate)
        )
        bn, bk = c.block_shape
        # Refine the scale grid before TP slicing. Every fine block inherits
        # its original scale, including partitions starting inside a block.
        if group == "w13":
            bn = gcd(bn, c.intermediate)
        else:
            bk = gcd(bk, c.intermediate)
        return n, k, bn, bk

    @property
    def w13_half_n(self):
        c = self.load_spec
        return (
            c.intermediate if c.runtime_intermediate is None else c.runtime_intermediate
        )

    @property
    def w13_k(self):
        c = self.load_spec
        return c.hidden if c.w13_k_padded is None else c.w13_k_padded

    @property
    def w2_n(self):
        c = self.load_spec
        return c.hidden if c.w2_n_padded is None else c.w2_n_padded

    @property
    def w2_k(self):
        k = self.load_spec.w2_k_padded
        return self.w13_half_n if k is None else k

    def _runtime_geometry(self, group):
        _, _, bn, bk = self._local_geometry(group)
        n, k = (
            (self.w13_half_n, self.w13_k) if group == "w13" else (self.w2_n, self.w2_k)
        )
        return n, k, bn, bk

    @property
    def staging_bytes(self):
        return max(
            n * k + 4 * ((n + bn - 1) // bn) * ((k + bk - 1) // bk)
            for n, k, bn, bk in (self._runtime_geometry(g) for g in ("w13", "w2"))
        )

    def validate_source_metadata(self, group, is_scale, shape, dtype):
        allowed = self.source_scale_metadata if is_scale else ("F8_E4M3", "F8_E4M3FN")
        if tuple(shape) != self.source_shape(group, is_scale) or dtype not in allowed:
            raise ValueError(
                f"Invalid {self.source_name} metadata: "
                f"{group} scale={is_scale} {shape} {dtype}"
            )

    def _decode_scale(self, scale):
        """Return the multiplicative block scale this source's encoding means."""
        return scale

    def _slice(self, tensor, group, is_scale):
        c = self.load_spec
        n, k, bn, bk = self._local_geometry(group)
        if not is_scale:
            return tensor.narrow(
                0 if group == "w13" else 1, c.tp_rank * c.intermediate, c.intermediate
            )
        # This is CPU source-grid interpretation, not device dequantization.
        # Index the original block grid directly, without expanding per element.
        src_bn, src_bk = c.block_shape
        row_offset = c.tp_rank * c.intermediate if group == "w13" else 0
        col_offset = c.tp_rank * c.intermediate if group == "w2" else 0
        rows = (
            torch.arange((n + bn - 1) // bn, device="cpu") * bn + row_offset
        ) // src_bn
        cols = (
            torch.arange((k + bk - 1) // bk, device="cpu") * bk + col_offset
        ) // src_bk
        return self._decode_scale(tensor.index_select(0, rows).index_select(1, cols))

    def load_shard(self, group, is_scale, tensor, shard_id, expert_id):
        if self._layer is None or self._finalized or self._failed:
            raise RuntimeError(
                f"{self.source_name} loader is uninitialized, finalized or failed"
            )
        expert = self._layer._map_global_expert_id_to_local_expert_id(expert_id)
        if expert == -1:
            return False
        if not 0 <= expert < self.load_spec.experts:
            raise ValueError("Invalid local expert mapping")
        if shard_id not in (("w1", "w3") if group == "w13" else ("w2",)):
            raise ValueError(
                f"{self.source_name} shard does not belong to this projection"
            )
        key = (expert, shard_id)
        if key in self._loaded or self._pending == (key, is_scale):
            raise ValueError(f"Duplicate {self.source_name} shard")
        allowed = self.source_scale_dtypes if is_scale else (torch.float8_e4m3fn,)
        if (
            tuple(tensor.shape) != self.source_shape(group, is_scale)
            or tensor.dtype not in allowed
        ):
            raise ValueError(
                f"Invalid {self.source_name} {shard_id} shape/dtype: "
                f"{tensor.shape} {tensor.dtype}"
            )
        if self._load_session is not None and self._load_session.collecting:
            self._load_session.add(
                self, group, is_scale, tensor, shard_id, expert_id, expert
            )
            return True
        slot = self._upload_buffer
        if slot is None or slot.host is None:
            raise RuntimeError(
                f"{self.source_name} requires a paired checkpoint upload session"
            )
        owner = (id(self), key)
        if slot.owner not in (None, owner) or (
            self._pending and self._pending[0] != key
        ):
            raise RuntimeError(
                f"{self.source_name} checkpoint reader must deliver adjacent pairs"
            )
        if tensor.device.type != "cpu":
            raise ValueError(
                f"{self.source_name} paired reader must supply CPU tensors"
            )
        n, k, bn, bk = self._runtime_geometry(group)
        scale_shape = ((n + bn - 1) // bn, (k + bk - 1) // bk)
        nw, ns = n * k, 4 * scale_shape[0] * scale_shape[1]
        try:
            value = self._slice(tensor, group, is_scale)
            slot.owner = owner
            if is_scale:
                staged = slot.host[nw : nw + ns].view(torch.float32).view(scale_shape)
            else:
                staged = slot.host[:nw].view(torch.float8_e4m3fn).view(n, k)
            if staged.shape != value.shape:
                if is_scale:
                    # Retain the original scale of any partially padded block.
                    staged.fill_(1)
                else:
                    staged.zero_()
                staged = staged[: value.shape[0], : value.shape[1]]
            staged.copy_(value)
            if self._pending is None:
                self._pending = (key, is_scale)
                return True
            slot.gpu[: nw + ns].copy_(slot.host[: nw + ns], non_blocking=True)
            weight = slot.gpu[:nw].view(torch.float8_e4m3fn).view(n, k)
            scale = (
                slot.gpu[nw : nw + ns]
                .view(torch.float32)
                .view((n + bn - 1) // bn, (k + bk - 1) // bk)
            )
            self._conversion_block = (bn, bk)
            self._write(group, expert, shard_id, weight, scale)
            slot.synchronize()
        except Exception:
            self._failed = True
            self._pending = None
            raise
        self._pending = slot.owner = None
        self._loaded.add(key)
        return True


class BlockFp8ToW4A8Loader(BlockFp8WeightLoader):
    def __init__(self, weight_spec, load_spec):
        super().__init__(weight_spec, load_spec)
        _native_converter("fp8_block_to_w4a8")

    def convert_shard(self, weight, scale, target, target_scale):
        fmt = {
            "ixmm_w4a8_nn": "ixmm_nn",
            "ixinfer_w4a8_nn_v1": "ixinfer_nn_v1",
            "ixinfer_w4a8_nn_v2": "ixinfer_nn",
            "ixinfer_w4a8_tn_v2": "ixinfer_tn",
        }[self.weight_spec.format_id]
        _native_converter("fp8_block_to_w4a8")(
            weight,
            scale,
            target,
            target_scale,
            block_shape=self._conversion_block,
            weight_format=fmt,
            backend="ixformer_native",
        )


class BlockFp8ToFP4FP8Loader(BlockFp8WeightLoader):
    def __init__(self, weight_spec, load_spec):
        super().__init__(weight_spec, load_spec)
        if any(
            n % 128 == 0 and k % 256 == 0
            for n, k, _, _ in (self._local_geometry(group) for group in ("w13", "w2"))
        ):
            _native_converter("preprocess_fp8_block_weight_for_ixmm")

    def convert_shard(self, weight, scale, target, target_scale):
        convert_to_ix30_fp4(
            weight.unsqueeze(0),
            scale.unsqueeze(0),
            source=Fp8SourceLayout("fp8-block", self._conversion_block),
            out_weight=target.unsqueeze(0),
            out_scale=target_scale.unsqueeze(0),
            # Preserve the existing shape fallback, but never silently use a
            # reference when this shape is part of the native load path.
            backend="ixformer_native"
            if weight.shape[0] % 128 == 0 and weight.shape[1] % 256 == 0
            else "torch",
        )


def make_block_fp8_loader(weight_spec, load_spec):
    if weight_spec.format_id == "ixinfer_w4a8_nn_v1" and load_spec.tp_size != 1:
        raise ValueError(
            "NN/v1 row scales require the full expert K before quantization"
        )
    contracts = {
        "ixinfer_w4a8_nn_v1": (
            BlockFp8ToW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_nn_v1", "NN", torch.int8),
        ),
        "ixinfer_w4a8_tn_v2": (
            BlockFp8ToW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_tn_v2", "TN", torch.int8, code_multiplier=18),
        ),
        "ixinfer_w4a8_nn_v2": (
            BlockFp8ToW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18),
        ),
        "ixmm_w4a8_nn": (
            BlockFp8ToW4A8Loader,
            MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8),
        ),
        "ixmm_fp4fp8": (
            BlockFp8ToFP4FP8Loader,
            MoEWeightSpec(
                "ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256, alignment=32
            ),
        ),
    }
    entry = contracts.get(weight_spec.format_id)
    if entry is None or entry[1] != weight_spec:
        raise ValueError(f"No block-FP8 loader for weight contract {weight_spec}")
    return entry[0](weight_spec, load_spec)
