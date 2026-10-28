# SPDX-License-Identifier: Apache-2.0
"""Bounded load-time MXFP4 -> channel W4A8 TN/v2 conversion.

The source uses adjacent E2M1 nibbles and UE8M0 blocks of 32. The target
uses TN 16+16 nibble packing, INT8 channel scales, and the v2 kernel's fixed
INT4-to-INT8 factor 18. No checkpoint files or non-expert methods change.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8 import (
    IluCompressedTensorsW4A8Int8MoEMethod,
    _pack_int4,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 import (
    IluMxfp4MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_quant_registry import (
    IluvatarQuantRegistry,
)


def quantize_mxfp4_to_w4a8_tn(weight, block_scale):
    """[N,K/2] int8/uint8 E2M1 + [N,K/32] uint8 UE8M0 -> TN INT4, [1,N] FP32.

    Computation is FP32, per output channel over the entire K. Zero channels
    use scale 1. Reject NaN exponent 255 and overflow instead of writing an
    apparently valid packed tensor. This is a loading operation, not Graph
    safe: finite checks intentionally synchronize before accepting weights.
    """
    if (
        weight.dtype not in (torch.int8, torch.uint8)
        or block_scale.dtype != torch.uint8
    ):
        raise ValueError("MXFP4 requires int8/uint8 packed weights and UE8M0 bytes")
    if weight.ndim != 2 or block_scale.ndim != 2:
        raise ValueError("MXFP4 expert weight and scale must be matrices")
    weight = weight.view(torch.uint8)
    n, half_k = weight.shape
    if not n or not half_k or half_k % 16:
        raise ValueError("MXFP4 requires positive N and 32-aligned K")
    if block_scale.shape != (n, half_k // 16):
        raise ValueError("MXFP4 UE8M0 block-scale shape mismatch")
    if weight.device != block_scale.device:
        raise ValueError("MXFP4 weight and scale must share a device")
    if (block_scale == 255).any():
        raise ValueError("MXFP4 UE8M0 exponent 255 is NaN")
    # MXFP4 low/high nibbles are adjacent K elements; TN packs half-tiles.
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(n, -1)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
        dtype=torch.float32,
    )
    values = lut[codes.long()].reshape(n, -1, 32)
    scales = torch.exp2(block_scale.float() - 127)
    values = (values * scales[..., None]).reshape(n, -1)
    if not torch.isfinite(values).all():
        raise ValueError("MXFP4 dequantization is not finite in FP32")
    channel = values.abs().amax(-1, keepdim=True) / 127
    channel = torch.where(channel > 0, channel, torch.ones_like(channel))
    q8 = torch.round(values / channel).clamp(-127, 127)
    q4 = torch.round(q8 / 18).clamp(-8, 7).to(torch.int8)
    # TN/v2 applies 18 in its kernel. Folding 18 here would scale twice.
    return _pack_int4(q4), channel.T.contiguous()


@IluvatarQuantRegistry.register_method
class IluMxfp4ToW4A8TNMoEMethod(IluCompressedTensorsW4A8Int8MoEMethod):
    """Unsharded local MXFP4 experts with the existing TN/v2 compute surface.

    EP owns complete experts even when attention uses multiple TP ranks:
    upstream FusedMoEParallelConfig sets the expert TP size to one. Global
    expert IDs are mapped before staging, so nonlocal shards allocate nothing.
    EP placement must be equal contiguous ranges, as required by the TN
    consumer. Tensor-sharded experts without EP and EPLB remain unsupported.
    """

    def __init__(self, layer, *, runtime_context=None, max_pending_shards=8):
        super().__init__(
            weight_quant=QuantizationArgs(
                num_bits=4, type="int", strategy="channel", symmetric=True
            ),
            input_quant=QuantizationArgs(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
            moe=layer.moe_config,
            ilu_params={"gemm_format": "TN", "pack_version": 2},
        )
        self._runtime_context = runtime_context
        self.max_pending_shards = max_pending_shards
        self._pending = {}
        self._converted = set()
        self._finalized = False
        self._layer = None

    @classmethod
    def from_ilu_runtime_config(cls, context):
        g = context.group
        w = g.weights
        a = g.input_activations
        if not isinstance(context.layer, RoutedExperts) or not isinstance(
            context.original_method, IluMxfp4MoEMethod
        ):
            raise TypeError("MXFP4 -> W4A8 TN requires MXFP4 routed experts")
        if g.require.original_disposition != "quantized":
            raise ValueError("MXFP4 -> W4A8 TN requires quantized disposition")
        if w.loading.mode != "transform" or w.loading.stage != "weight_load":
            raise ValueError("MXFP4 -> W4A8 TN requires transform@weight_load")
        if (
            w.source.format != "mxfp4"
            or not set(w.source.dtypes) <= {"int8", "uint8"}
            or dict(w.source.extra) != {"block_size": 32, "scale_format": "ue8m0"}
        ):
            raise ValueError("source must be MXFP4 with 32-element UE8M0 blocks")
        if (
            w.target.format != "int4-packed-tn"
            or w.target.dtype != "int8"
            or w.target.strategy != "channel"
            or dict(w.target.extra) != {"pack_version": 2}
        ):
            raise ValueError("target must be channel INT4 TN/v2")
        if (
            a is None
            or a.num_bits != 8
            or a.type != "int"
            or a.strategy != "token"
            or not a.dynamic
            or bool(a.extra)
            or g.output_activations is not None
        ):
            raise ValueError("W4A8 TN requires dynamic per-token INT8 activations")
        if set(g.options) - {"max_pending_shards"}:
            raise ValueError("unknown MXFP4 -> W4A8 TN option")
        pending = g.options.get("max_pending_shards", 8)
        if type(pending) is not int or pending < 1:
            raise ValueError("max_pending_shards must be a positive integer")
        return cls(context.layer, runtime_context=context, max_pending_shards=pending)

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        moe = layer.moe_config
        h, i = hidden_size, intermediate_size_per_partition
        if (
            moe.tp_size != 1
            or getattr(layer, "enable_eplb", False)
            or self.has_bias
            or not moe.is_act_and_mul
            or moe.hidden_dim_unpadded != h
            or moe.intermediate_size_per_partition_unpadded not in (None, i)
            or h % 32
            or i % 32
        ):
            raise ValueError(
                "MXFP4 -> W4A8 TN requires expert TP1 (EP allowed), unpadded bias-free gate/up experts"
            )
        ep_size = getattr(moe, "ep_size", 1)
        if ep_size > 1 or layer.expert_map is not None:
            mapping = layer.expert_map
            rank = getattr(moe, "ep_rank", -1)
            if (
                mapping is None
                or mapping.ndim != 1
                or mapping.numel() != num_experts * ep_size
                or not 0 <= rank < ep_size
            ):
                raise ValueError("W4A8 TN EP requires equally partitioned experts")
            expected = torch.full_like(mapping, -1)
            expected[rank * num_experts : (rank + 1) * num_experts] = torch.arange(
                num_experts, device=mapping.device, dtype=mapping.dtype
            )
            # Loading-time synchronization is deliberate; the inherited TN
            # consumer derives expert IDs from ep_rank * local_num_experts.
            if not torch.equal(mapping, expected):
                raise ValueError(
                    "W4A8 TN EP requires contiguous linear expert placement"
                )
        config = get_current_vllm_config()
        fmt = config.load_config.load_format
        if str(getattr(fmt, "value", fmt)) not in {
            "auto",
            "safetensors",
            "pt",
            "npcache",
        }:
            raise ValueError("MXFP4 streaming conversion requires sequential loading")
        if self._layer is not None:
            raise RuntimeError(
                "MXFP4 runtime method cannot be reused for another layer"
            )
        super().create_weights(layer, num_experts, h, i, params_dtype, **attrs)
        self._layer = layer
        self._hidden, self._intermediate, self._num_experts = h, i, num_experts
        for group in ("w13", "w2"):
            for scale in (False, True):
                suffix = "weight_scale" if scale else "weight"
                getattr(layer, f"{group}_{suffix}").weight_loader = self._loader(
                    group, scale
                )

    def _loader(self, group, is_scale):
        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            layer = self._layer
            expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if expert == -1:
                return False if return_success else None
            if shard_id not in (("w1", "w3") if group == "w13" else ("w2",)):
                raise ValueError("checkpoint shard does not belong to target weight")
            unit = (group, expert, shard_id)
            part = "scale" if is_scale else "weight"
            if (
                self._finalized
                or unit in self._converted
                or part in self._pending.get(unit, {})
            ):
                raise ValueError(f"duplicate/finalized MXFP4 shard: {unit}/{part}")
            if (
                unit not in self._pending
                and len(self._pending) >= self.max_pending_shards
            ):
                raise RuntimeError(
                    "MXFP4 shards exceed max_pending_shards; check checkpoint ordering"
                )
            n, k = (
                (self._intermediate, self._hidden)
                if group == "w13"
                else (self._hidden, self._intermediate)
            )
            shape = (n, k // (32 if is_scale else 2))
            allowed_dtypes = (torch.uint8,) if is_scale else (torch.int8, torch.uint8)
            if (
                loaded_weight.dtype not in allowed_dtypes
                or tuple(loaded_weight.shape) != shape
            ):
                raise ValueError(
                    f"{weight_name}: expected {allowed_dtypes} {shape}, got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                )
            # Own the staging storage: safetensors iterators may release/reuse it.
            parts = self._pending.setdefault(unit, {})
            parts[part] = loaded_weight.to(device=param.device, copy=True)
            if len(parts) == 2:
                packed, channel = quantize_mxfp4_to_w4a8_tn(
                    parts["weight"], parts["scale"]
                )
                target = getattr(layer, f"{group}_weight").data[expert]
                scales = getattr(layer, f"{group}_weight_scale").data[expert]
                if group == "w13":
                    offset = 0 if shard_id == "w1" else self._intermediate
                    target = target.narrow(0, offset, n)
                    scales = scales.narrow(1, offset, n)
                target.copy_(packed)
                scales.copy_(channel)
                del self._pending[unit]
                self._converted.add(unit)
            return True if return_success else None

        return load

    def process_weights_after_loading(self, layer):
        if self._layer is not layer or self._finalized:
            raise RuntimeError("TN finalization requires its own unfinalized layer")
        expected = {
            (g, e, s)
            for e in range(self._num_experts)
            for g, s in (("w13", "w1"), ("w13", "w3"), ("w2", "w2"))
        }
        if self._converted != expected or self._pending:
            raise RuntimeError(
                f"missing MXFP4 expert shards: {sorted(expected - self._converted)[:16]}"
            )
        super().process_weights_after_loading(layer)
        self._finalized = True
        if self._runtime_context is not None:
            self._runtime_context.record_shape_result(transformed=True)

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self._finalized:
            raise RuntimeError("MXFP4 runtime weights have not been finalized")
        if self._runtime_context is not None:
            self._runtime_context.assert_validation_complete()
        return super().apply(
            layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
