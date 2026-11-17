# SPDX-License-Identifier: Apache-2.0
"""MXFP4 checkpoint loaders: convert paired shards directly to final storage.

This module owns source interpretation, TP slicing and bounded load-time staging.
Expert compute sees only MoEWeights and never imports these loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeights,
    MoEWeightSpec,
)


@lru_cache(maxsize=2)
def _native_converter(name):
    from ixformer.inference import functions as ops
    from ixformer.inference.backend_registry import registry

    # The registry permits a reference fallback. Checkpoint loading requires
    # the native converter: a missing/newer binding must fail before staging.
    if "ixformer_native" not in registry.available_backends(name):
        raise RuntimeError(
            f"MXFP4 loading requires native {name}; install the matching IxFormer build"
        )
    return getattr(ops, name)


@dataclass(frozen=True)
class Mxfp4LoadSpec:
    experts: int
    hidden: int
    intermediate: int
    tp_size: int = 1
    tp_rank: int = 0


class Mxfp4UploadBuffer:
    """One synchronous upload slot shared by all layers in a load session."""

    def __init__(self, capacity: int, device: torch.device):
        self.device = torch.device(device)
        self.host = torch.empty(
            capacity,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self.gpu = torch.empty(capacity, dtype=torch.uint8, device=self.device)
        self.owner = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def close(self):
        try:
            self.synchronize()
        finally:
            self.owner = None
            self.host = self.gpu = None


class Mxfp4WeightLoader(ABC):
    """One loader per layer. A shard is committed only after both parts arrive.

    create_weights binds the community callback signature to load_shard. There
    is no full-layer source allocation and no post-load conversion fallback.
    """

    def __init__(self, weight_spec: MoEWeightSpec, load_spec: Mxfp4LoadSpec):
        self.weight_spec = weight_spec
        self.load_spec = load_spec
        if (
            load_spec.experts < 1
            or not 0 <= load_spec.tp_rank < load_spec.tp_size
            or load_spec.hidden % weight_spec.alignment
            or load_spec.intermediate % weight_spec.alignment
        ):
            raise ValueError("Invalid or unaligned MXFP4 load geometry")
        self._layer = None
        self._pending = None
        self._upload_buffer = None
        self._load_session = None
        self._loaded = set()
        self._finalized = False
        self._failed = False

    @property
    def staging_bytes(self):
        cfg = self.load_spec
        return cfg.hidden * cfg.intermediate // 2 + cfg.hidden * cfg.intermediate // 32

    def bind_upload_buffer(self, buffer):
        if self._upload_buffer is not None or self._loaded or self._pending:
            raise RuntimeError("MXFP4 upload session cannot be rebound")
        if buffer.host.numel() < self.staging_bytes:
            raise ValueError("MXFP4 upload buffer is too small")
        if self._layer.w13_weight.device != buffer.device:
            raise ValueError("MXFP4 upload buffer is on the wrong device")
        self._upload_buffer = buffer

    def source_shape(self, group, is_scale):
        cfg = self.load_spec
        divisor = 32 if is_scale else 2
        global_i = cfg.intermediate * cfg.tp_size
        return (
            (global_i, cfg.hidden // divisor)
            if group == "w13"
            else (cfg.hidden, global_i // divisor)
        )

    def validate_source_metadata(self, group, is_scale, shape, dtype):
        allowed = ("U8", "F8_E8M0") if is_scale else ("U8", "I8")
        if tuple(shape) != self.source_shape(group, is_scale) or dtype not in allowed:
            raise ValueError(
                f"Invalid MXFP4 metadata: {group} scale={is_scale} {shape} {dtype}"
            )

    def create_weights(self, layer, weight_attrs):
        if self._layer is not None:
            raise RuntimeError("MXFP4 loader cannot be reused")
        self._layer = layer
        cfg = self.load_spec
        attrs = {k: v for k, v in weight_attrs.items() if k != "weight_loader"}
        for group, n, k in (
            ("w13", 2 * cfg.intermediate, cfg.hidden),
            ("w2", cfg.hidden, cfg.intermediate),
        ):
            shapes = self.weight_spec.shapes(cfg.experts, n, k)
            for is_scale, suffix, shape, dtype in (
                (False, "weight", shapes[0], self.weight_spec.weight_dtype),
                (True, "weight_scale", shapes[1], self.weight_spec.scale_dtype),
            ):
                param = torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                )
                set_weight_attrs(
                    param, {**attrs, "weight_loader": self._callback(group, is_scale)}
                )
                layer.register_parameter(f"{group}_{suffix}", param)

    def _callback(self, group, is_scale):
        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            loaded = self.load_shard(
                group, is_scale, loaded_weight, shard_id, expert_id
            )
            return loaded if return_success else None

        load.mxfp4_loader = self
        return load

    def load_shard(self, group, is_scale, tensor, shard_id, expert_id):
        if self._layer is None or self._finalized or self._failed:
            raise RuntimeError("MXFP4 loader is uninitialized, finalized or failed")
        expert = self._layer._map_global_expert_id_to_local_expert_id(expert_id)
        if expert == -1:
            return False
        cfg = self.load_spec
        if not 0 <= expert < cfg.experts:
            raise ValueError("Invalid local expert mapping")
        if shard_id not in (("w1", "w3") if group == "w13" else ("w2",)):
            raise ValueError("MXFP4 shard does not belong to this projection")
        key = (expert, shard_id)
        pending = self._pending
        if key in self._loaded or (pending is not None and pending == (key, is_scale)):
            raise ValueError("Duplicate MXFP4 shard")
        if is_scale and tensor.dtype == torch.float8_e8m0fnu:
            tensor = tensor.view(torch.uint8)
        allowed = (torch.uint8,) if is_scale else (torch.int8, torch.uint8)
        divisor = 32 if is_scale else 2
        shape = self.source_shape(group, is_scale)
        if tensor.dtype not in allowed or tuple(tensor.shape) != shape:
            raise ValueError(
                f"MXFP4 {shard_id}: expected {allowed} {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if self._load_session is not None and self._load_session.collecting:
            self._load_session.add(
                self, group, is_scale, tensor, shard_id, expert_id, expert
            )
            return True
        # TP partitioning occurs before quantization, matching the existing
        # per-partition conversion semantics. EP supplies tp_size=1 here.
        dim, length = (
            (0, cfg.intermediate)
            if group == "w13"
            else (1, cfg.intermediate // divisor)
        )
        tensor = tensor.narrow(dim, cfg.tp_rank * length, length)
        buffer = self._upload_buffer
        if buffer is None or buffer.host is None:
            raise RuntimeError("MXFP4 requires a paired checkpoint upload session")
        owner = (id(self), key)
        if buffer.owner not in (None, owner) or (pending and pending[0] != key):
            raise RuntimeError("MXFP4 checkpoint reader must deliver adjacent pairs")
        if tensor.device.type != "cpu":
            raise ValueError("MXFP4 paired reader must supply CPU tensors")
        n, k = (
            (cfg.intermediate, cfg.hidden)
            if group == "w13"
            else (cfg.hidden, cfg.intermediate)
        )
        nw, ns = n * k // 2, n * k // 32
        try:
            buffer.owner = owner
            # Copy a possibly strided TP view directly into contiguous staging.
            offset, size = (nw, ns) if is_scale else (0, nw)
            buffer.host[offset : offset + size].view(tensor.shape).copy_(
                tensor.view(torch.uint8)
            )
            if pending is None:
                self._pending = (key, is_scale)
                return True
            buffer.gpu[: nw + ns].copy_(buffer.host[: nw + ns], non_blocking=True)
            weight = buffer.gpu[:nw].view(n, k // 2)
            scale = buffer.gpu[nw : nw + ns].view(n, k // 32)
            self._write(group, expert, shard_id, weight, scale)
            # Protect both pinned CPU storage and GPU inputs before reuse.
            buffer.synchronize()
        except Exception:
            # A failed conversion may have touched final storage; never expose
            # or retry it as a successfully loaded layer.
            self._failed = True
            self._pending = None
            raise
        self._pending = None
        buffer.owner = None
        self._loaded.add(key)
        return True

    @torch.no_grad()
    def _write(self, group, expert, shard, weight, scale):
        target = getattr(self._layer, group + "_weight")[expert]
        scales = getattr(self._layer, group + "_weight_scale")[expert]
        if group == "w13":
            n = self.load_spec.intermediate
            offset = n if shard == "w3" else 0
            if self.weight_spec.layout == "TN":
                target = target.narrow(0, offset, n)
            elif self.weight_spec.scale_group_size == 256:
                target = target.narrow(-1, offset, n)
            else:
                target = target.narrow(-1, offset // 2, n // 2)
            scales = scales.narrow(-1, offset, n)
        self.convert_shard(weight, scale, target, scales)

    @abstractmethod
    def convert_shard(self, weight, scale, target, target_scale):
        """Convert one local source shard into its final parameter slice."""

    def finalize(self) -> MoEWeights:
        if self._layer is None or self._finalized or self._failed:
            raise RuntimeError("MXFP4 loader is uninitialized, finalized or failed")
        expected = {
            (e, s) for e in range(self.load_spec.experts) for s in ("w1", "w2", "w3")
        }
        if self._pending or self._loaded != expected:
            raise RuntimeError(
                f"Missing MXFP4 shards: {sorted(expected - self._loaded)[:16]}"
            )
        weights = MoEWeights.from_layer(self._layer, self.weight_spec)
        for scale in (weights.w13_weight_scale, weights.w2_weight_scale):
            if not torch.isfinite(scale).all():
                self._failed = True
                raise ValueError("Invalid or non-finite MXFP4 weights")
        self._finalized = True
        return weights


class Mxfp4ToIxInferW4A8Loader(Mxfp4WeightLoader):
    def convert_shard(self, weight, scale, target, target_scale):
        _native_converter("mxfp4_to_w4a8")(
            weight,
            scale,
            target,
            target_scale,
            weight_format="ixinfer_" + self.weight_spec.layout.lower(),
            backend="ixformer_native",
        )


class Mxfp4ToIxMMW4A8Loader(Mxfp4WeightLoader):
    def convert_shard(self, weight, scale, target, target_scale):
        _native_converter("mxfp4_to_w4a8")(
            weight,
            scale,
            target,
            target_scale,
            weight_format="ixmm_nn",
            backend="ixformer_native",
        )


class Mxfp4ToIxMMFP4FP8Loader(Mxfp4WeightLoader):
    def convert_shard(self, weight, scale, target, target_scale):
        _native_converter("mxfp4_to_fp4fp8")(
            weight,
            scale,
            target,
            target_scale,
            backend="ixformer_native",
        )


def make_mxfp4_loader(weight_spec, load_spec):
    # Match complete semantic contracts, not just a shared TN/NN spelling.
    contracts = {
        "ixinfer_w4a8_tn_v2": (
            Mxfp4ToIxInferW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_tn_v2", "TN", torch.int8, code_multiplier=18),
        ),
        "ixinfer_w4a8_nn_v2": (
            Mxfp4ToIxInferW4A8Loader,
            MoEWeightSpec("ixinfer_w4a8_nn_v2", "NN", torch.int8, code_multiplier=18),
        ),
        "ixmm_w4a8_nn": (
            Mxfp4ToIxMMW4A8Loader,
            MoEWeightSpec("ixmm_w4a8_nn", "NN", torch.int8),
        ),
        "ixmm_fp4fp8": (
            Mxfp4ToIxMMFP4FP8Loader,
            MoEWeightSpec("ixmm_fp4fp8", "fp4fp8", torch.uint8, scale_group_size=256),
        ),
    }
    entry = contracts.get(weight_spec.format_id)
    if entry is None or entry[1] != weight_spec:
        raise ValueError(f"No MXFP4 loader for weight contract {weight_spec}")
    return entry[0](weight_spec, load_spec)
