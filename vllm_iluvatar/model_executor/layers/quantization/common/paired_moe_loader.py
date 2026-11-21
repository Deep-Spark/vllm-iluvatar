# SPDX-License-Identifier: Apache-2.0
"""Source-independent paired expert loading and final parameter ownership."""

from functools import lru_cache

import torch
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEWeights,
)


@lru_cache(maxsize=4)
def _native_converter(name):
    from ixformer.inference import functions as ops
    from ixformer.inference.backend_registry import registry

    # The registry permits a reference fallback. Checkpoint loading requires
    # the native converter: a missing/newer binding must fail before staging.
    if "ixformer_native" not in registry.available_backends(name):
        raise RuntimeError(
            f"Expert loading requires native {name}; install the matching IxFormer build"
        )
    return getattr(ops, name)


class ExpertUploadBuffer:
    """One synchronous upload slot shared by all layers in a load session."""

    def __init__(
        self, capacity: int, device: torch.device, *, conversion_capacity: int = 0
    ):
        self.device = torch.device(device)
        self.host = torch.empty(
            capacity,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self.gpu = torch.empty(capacity, dtype=torch.uint8, device=self.device)
        self.conversion = torch.empty(
            conversion_capacity, dtype=torch.uint8, device=self.device
        )
        self.owner = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def close(self):
        try:
            self.synchronize()
        finally:
            self.owner = None
            self.host = self.gpu = self.conversion = None


class PairedMoEWeightLoader:
    scale_parameter_name = "weight_scale"

    @property
    def conversion_bytes(self):
        return 0

    def expected_pairs(self):
        return (
            (e, shard)
            for e in range(self.load_spec.experts)
            for shard in ("w1", "w2", "w3")
        )

    def bind_upload_buffer(self, buffer):
        if self._upload_buffer is not None or self._loaded or self._pending:
            raise RuntimeError(f"{self.source_name} upload session cannot be rebound")
        if buffer.host.numel() < self.staging_bytes:
            raise ValueError(f"{self.source_name} upload buffer is too small")
        if buffer.conversion.numel() < self.conversion_bytes:
            raise ValueError("Paired conversion buffer is too small")
        if self._layer.w13_weight.device != buffer.device:
            raise ValueError(f"{self.source_name} upload buffer is on the wrong device")
        self._upload_buffer = buffer

    def create_weights(self, layer, weight_attrs):
        if self._layer is not None:
            raise RuntimeError(f"{self.source_name} loader cannot be reused")
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
                layer.register_parameter(
                    f"{group}_{self.scale_parameter_name if is_scale else suffix}",
                    param,
                )

    def _callback(self, group, is_scale):
        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            loaded = self.load_shard(
                group, is_scale, loaded_weight, shard_id, expert_id
            )
            return loaded if return_success else None

        load.paired_weight_loader = self
        return load

    @torch.no_grad()
    def _write(self, group, expert, shard, weight, scale):
        target = getattr(self._layer, group + "_weight")[expert]
        scales = getattr(self._layer, group + "_" + self.scale_parameter_name)[expert]
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

    def finalize(self) -> MoEWeights:
        if self._layer is None or self._finalized or self._failed:
            raise RuntimeError(
                f"{self.source_name} loader is uninitialized, finalized or failed"
            )
        expected = {
            (e, s) for e in range(self.load_spec.experts) for s in ("w1", "w2", "w3")
        }
        if self._pending or self._loaded != expected:
            raise RuntimeError(
                f"Missing {self.source_name} shards: {sorted(expected - self._loaded)[:16]}"
            )
        if self.scale_parameter_name != "weight_scale":
            for group in ("w13", "w2"):
                param = self._layer._parameters.pop(
                    f"{group}_{self.scale_parameter_name}"
                )
                self._layer.register_parameter(f"{group}_weight_scale", param)
        weights = MoEWeights.from_layer(self._layer, self.weight_spec)
        for scale in (weights.w13_weight_scale, weights.w2_weight_scale):
            if not torch.isfinite(scale).all():
                self._failed = True
                raise ValueError(f"Invalid or non-finite {self.source_name} weights")
        self._finalized = True
        return weights
