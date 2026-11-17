# SPDX-License-Identifier: Apache-2.0
"""Streaming block-FP8 checkpoint -> W4A8 NN routed experts.

Only the compute surface is shared with compressed-tensors W4A8. Checkpoint
parameters remain FP8/block-scale inputs to our loaders, never TN/v2 tensors.
Resident weights use the ixmm NN layout from allocation onwards. Activation quantization is
dynamic per-token INT8; prefill and decode use the ivcore30 NN method.
"""

from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp4.fp8_moe import (
    IluFp8MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.w4a8.nn_moe import (
    IluW4A8NNMoEMethod,
    validate_nn_pack_version,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

__all__ = [
    "IluFp8ToW4A8NNOnlineMoEMethod",
    "quantize_fp8_to_w4a8_nn",
]


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack each 32 INT4 values into 16 bytes, low half then high half."""
    *lead, length = values.shape
    if length % 32:
        raise ValueError(f"value last dim {length} must be a multiple of 32")
    v = values.reshape(*lead, length // 32, 32).to(torch.int16)
    low, high = v[..., :16], v[..., 16:]
    packed = ((high & 0x0F) << 4) | (low & 0x0F)
    packed = torch.where(packed > 127, packed - 256, packed)
    return packed.reshape(*lead, length // 2).to(torch.int8)


def quantize_fp8_to_w4a8_nn(
    weight, block_scale, block_shape, *, k_start=0, k_size=None, pad_k=None
):
    """One [N,K] expert shard -> NN packed [K,N/2], FP32 scales [1,N].

    Match the offline NN/v1 quantizer: dequantize block FP8 in FP32, quantize
    to symmetric per-channel INT8, then INT4 and pack directly to NN/v1.
    Compute both scales over the FULL checkpoint K before selecting a TP shard.
    For a nonzero channel the full INT8 amax is 127, so the second scale is
    round(127/7) = 18. Recomputing it on the local shard is NOT equivalent.
    """
    n, k = weight.shape
    bn, bk = block_shape
    if weight.dtype != torch.float8_e4m3fn or n % bn or k % bk:
        raise ValueError("W4A8 NN requires aligned float8_e4m3fn block weights")
    if tuple(block_scale.shape) != (n // bn, k // bk):
        raise ValueError("FP8 block-scale shape does not match its weight")
    value = weight.float().reshape(n // bn, bn, k // bk, bk)
    value = (value * block_scale.float()[:, None, :, None]).reshape(n, k)
    k_size = k if k_size is None else k_size
    if k_start < 0 or k_size <= 0 or k_start + k_size > k:
        raise ValueError("invalid checkpoint K slice")
    if n % 32 or k_start % 32 or k_size % 32:
        raise ValueError("NN packing requires 32-aligned dimensions and K slices")
    if pad_k is not None and (pad_k < k_size or pad_k % 32):
        raise ValueError("pad_k must cover the local K and be 32-aligned")
    scale = value.abs().amax(dim=-1, keepdim=True) / 127.0
    q4_scale = torch.where(scale > 0, 18.0, 1.0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    value = value.narrow(1, k_start, k_size)
    q8 = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    q4 = torch.round(q8 / q4_scale).clamp(-8, 7).to(torch.int8)
    if pad_k is not None and pad_k != k_size:
        q4 = torch.nn.functional.pad(q4, (0, pad_k - k_size))
    local_k = q4.shape[1]
    nn = q4.T.contiguous().reshape(local_k // 32, 2, 16, n // 32, 2, 16)
    nn = nn.permute(0, 4, 2, 3, 1, 5).contiguous().reshape(local_k, n)
    return _pack_int4(nn), (scale * q4_scale).T.contiguous()


@IluvatarQuantRegistry.register_method
class IluFp8ToW4A8NNOnlineMoEMethod(IluW4A8NNMoEMethod):
    """Official block-FP8 loading contract with NN W4A8 compute."""

    def __init__(
        self, quant_config, layer, *, runtime_context=None, max_pending_shards=8
    ):
        super().__init__(
            weight_quant=QuantizationArgs(
                num_bits=4, type="int", strategy="channel", symmetric=True
            ),
            input_quant=QuantizationArgs(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
            moe=layer.moe_config,
            ilu_params={"gemm_format": "NN"},
        )
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = True
        self._runtime_context = runtime_context
        self.max_pending_shards = max_pending_shards
        self._layer = None
        self._weights = {}
        self._scales = set()
        self._converted = set()
        self._finalized = False

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        group = context.group
        w = group.weights
        source, target, activation = w.source, w.target, group.input_activations
        if not isinstance(context.layer, RoutedExperts) or not isinstance(
            context.original_method, IluFp8MoEMethod
        ):
            raise TypeError("FP8 -> W4A8 NN requires FP8 routed experts")
        if group.require.original_disposition != "quantized":
            raise ValueError("FP8 -> W4A8 NN requires original_disposition=quantized")
        if w.loading.mode != "transform" or w.loading.stage != "weight_load":
            raise ValueError("FP8 -> W4A8 NN requires transform@weight_load")
        if (
            source.format != "fp8-block"
            or tuple(source.dtypes) != ("float8_e4m3fn",)
            or set(source.extra) - {"block_shape"}
            or tuple(source.extra.get("block_shape", (128, 128))) != (128, 128)
            or tuple(context.base_quant_config.weight_block_size or ()) != (128, 128)
            or not context.base_quant_config.is_checkpoint_fp8_serialized
            or context.base_quant_config.activation_scheme != "dynamic"
        ):
            raise ValueError("FP8 -> W4A8 NN requires dynamic 128x128 block-FP8 source")
        if (
            target.format != "int4-packed-nn"
            or target.dtype != "int8"
            or target.strategy != "channel"
            or set(target.extra) - {"pack_version"}
        ):
            raise ValueError("FP8 -> W4A8 NN requires a channel-wise ixmm NN target")
        validate_nn_pack_version(target.extra.get("pack_version", 1))
        if (
            activation is None
            or activation.num_bits != 8
            or activation.type != "int"
            or activation.strategy != "token"
            or not activation.dynamic
            or group.output_activations is not None
        ):
            raise ValueError("W4A8 NN requires dynamic per-token INT8 activations")
        if set(group.options) - {"max_pending_shards"}:
            raise ValueError("W4A8 NN only accepts the max_pending_shards option")
        pending = group.options.get("max_pending_shards", 8)
        if type(pending) is not int or pending < 1:
            raise ValueError("max_pending_shards must be a positive integer")
        return cls(
            context.base_quant_config,
            context.layer,
            runtime_context=context,
            max_pending_shards=pending,
        )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        from vllm.config import get_current_vllm_config

        h, i = hidden_size, intermediate_size_per_partition
        moe = layer.moe_config
        if (
            self.has_bias
            or not moe.is_act_and_mul
            or getattr(layer, "enable_eplb", False)
            or moe.hidden_dim_unpadded != h
            or moe.intermediate_size_per_partition_unpadded not in (None, i)
        ):
            raise ValueError(
                "W4A8 NN needs bias-free gate/up experts without EPLB/source padding"
            )
        if tuple(self.weight_block_size or ()) != (128, 128) or h % 128 or i % 128:
            raise ValueError("W4A8 NN requires 128-aligned block-FP8 TP shards")
        try:
            config = get_current_vllm_config()
        except AssertionError:
            config = None
        load_format = getattr(
            getattr(config, "load_config", None), "load_format", "auto"
        )
        if str(getattr(load_format, "value", load_format)) not in {
            "auto",
            "safetensors",
            "pt",
            "npcache",
        }:
            raise ValueError("W4A8 NN streaming conversion requires sequential loading")
        self._layer = layer
        self._hidden, self._intermediate, self._num_experts = h, i, num_experts
        self.gemm1_info = W4A8GroupGemmInfo("NN", h, 2 * i)
        self.gemm2_info = W4A8GroupGemmInfo("NN", i, h)
        if self.gemm1_info.pad_k != h:
            raise ValueError("W4A8 NN does not support hidden K padding")
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        for group, info, block_shape in (
            ("w13", self.gemm1_info, (2 * i // 128, h // 128)),
            # Keep full checkpoint block scales for down_proj's global-K amax.
            ("w2", self.gemm2_info, (h // 128, i * moe.tp_size // 128)),
        ):
            weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, *info.get_pad_weight_shape(), dtype=torch.int8
                ),
                requires_grad=False,
            )
            layer.register_parameter(f"{group}_weight", weight)
            set_weight_attrs(weight, attrs)
            weight.weight_loader = self._loader(layer, group, scale=False)
            scale = torch.nn.Parameter(
                torch.empty(num_experts, 1, info.n, dtype=torch.float32),
                requires_grad=False,
            )
            scale.ilu_derived_at_load = True
            layer.register_parameter(f"{group}_weight_scale", scale)
            blocks = torch.nn.Parameter(
                torch.empty(num_experts, *block_shape, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter(f"{group}_weight_scale_inv", blocks)
            set_weight_attrs(
                blocks,
                {**attrs, "quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
            )
            blocks.weight_loader = self._loader(layer, group, scale=True)
            for suffix in ("input_scale", "bias", "i8_weight_scale", "i8_weight_zero"):
                setattr(layer, f"{group}_{suffix}", None)

    def _loader(self, layer, group, *, scale):
        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            local = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local == -1:
                return False if return_success else None
            if shard_id not in (("w1", "w3") if group == "w13" else ("w2",)):
                raise ValueError("checkpoint shard does not belong to target weight")
            dtype = torch.float32 if scale else torch.float8_e4m3fn
            if loaded_weight.ndim != 2 or loaded_weight.dtype != dtype:
                raise ValueError(
                    f"expected 2-D {dtype} checkpoint tensor: {weight_name}"
                )
            if getattr(param, "is_transposed", False) or getattr(
                param, "use_bitsandbytes_4bit", False
            ):
                raise ValueError("transposed/BitsAndBytes checkpoint is unsupported")
            unit = (group, local, shard_id)
            if (
                self._finalized
                or unit in self._converted
                or unit in (self._scales if scale else self._weights)
            ):
                raise ValueError(f"duplicate/finalized checkpoint shard: {unit}")
            pending = self._weights.keys() | self._scales | {unit}
            if len(pending) > self.max_pending_shards:
                raise RuntimeError(
                    "FP8 shards exceed max_pending_shards; check checkpoint ordering"
                )
            if scale:
                if group == "w2":
                    if loaded_weight.shape != param[local].shape:
                        raise ValueError(
                            "down_proj requires full checkpoint K block scales"
                        )
                    param.data[local].copy_(loaded_weight)
                else:
                    layer.weight_loader(
                        param, loaded_weight, weight_name, shard_id, expert_id
                    )
                self._scales.add(unit)
            else:
                h, i = self._hidden, self._intermediate
                shape = (
                    (2 * i, h) if group == "w13" else (h, i * layer.moe_config.tp_size)
                )
                staging = torch.empty(shape, device=param.device, dtype=dtype)
                args = dict(
                    expert_data=staging,
                    loaded_weight=loaded_weight,
                    tp_rank=layer.moe_config.tp_rank,
                )
                if group == "w13":
                    layer._load_w13(shard_dim=0, shard_id=shard_id, **args)
                    staging = staging.narrow(0, 0 if shard_id == "w1" else i, i)
                else:
                    # The sequential HF loader gives each rank the whole expert
                    # tensor. Retain it only until its scales arrive. Slicing K
                    # here would lose the offline per-channel scale statistics.
                    if loaded_weight.shape != staging.shape:
                        raise ValueError("down_proj requires full checkpoint K weights")
                    staging.copy_(loaded_weight)
                self._weights[unit] = staging
            if unit in self._weights and unit in self._scales:
                self._convert(layer, unit)
            return True if return_success else None

        return load

    def _convert(self, layer, unit):
        group, expert, shard = unit
        source = self._weights[unit]
        blocks = getattr(layer, f"{group}_weight_scale_inv")[expert]
        weight = getattr(layer, f"{group}_weight").data[expert]
        scale = getattr(layer, f"{group}_weight_scale").data[expert]
        info = self.gemm1_info if group == "w13" else self.gemm2_info
        if group == "w13":
            i = self._intermediate
            offset = 0 if shard == "w1" else i
            blocks = blocks.narrow(0, offset // 128, i // 128)
            weight = weight.narrow(1, offset // 2, i // 2)
            scale = scale.narrow(1, offset, i)
        packed, channel_scale = quantize_fp8_to_w4a8_nn(
            source,
            blocks,
            self.weight_block_size,
            pad_k=info.pad_k,
            k_start=(
                layer.moe_config.tp_rank * self._intermediate if group == "w2" else 0
            ),
            k_size=self._intermediate if group == "w2" else self._hidden,
        )
        weight.copy_(packed)
        scale.copy_(channel_scale)
        del self._weights[unit]
        self._scales.remove(unit)
        self._converted.add(unit)

    def process_weights_after_loading(self, layer):
        if self._layer is not layer or self._finalized:
            raise RuntimeError(
                "W4A8 NN finalization requires its own unfinalized layer"
            )
        expected = {
            (g, e, s)
            for e in range(self._num_experts)
            for g, s in (("w13", "w1"), ("w13", "w3"), ("w2", "w2"))
        }
        if self._converted != expected or self._weights or self._scales:
            raise RuntimeError(
                f"missing FP8 expert shards: {sorted(expected - self._converted)[:16]}"
            )
        del layer.w13_weight_scale_inv
        del layer.w2_weight_scale_inv
        self._finalized = True
        if self._runtime_context is not None:
            self._runtime_context.record_shape_result(transformed=True)
