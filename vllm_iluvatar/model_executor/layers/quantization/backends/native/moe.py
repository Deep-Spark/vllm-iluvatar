# SPDX-License-Identifier: Apache-2.0
"""Stream native expert shards into the IxInfer W4A8 weight contract."""

import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    get_moe_backend,
)
from vllm_iluvatar.model_executor.layers.quantization.common.moe_compute_binding import (
    MoEComputeBindingMixin,
)


def _pack_weight(weight, layout, k_start=0, k_size=None):
    n, k = weight.shape
    value = weight.float()
    # w2's channel scale belongs to the full checkpoint K, including under TP.
    scale = value.abs().amax(dim=1, keepdim=True) / 127
    scale = torch.where(scale == 0, 1.0, scale)
    k_size = k if k_size is None else k_size
    q8 = (value.narrow(1, k_start, k_size) / scale).round().clamp_(-127, 127)
    q4 = (q8 / 18).round().to(torch.int8)
    if layout == "NN":
        tiles = q4.T.reshape(k_size // 32, 2, 16, n // 32, 2, 16)
        q4 = tiles.permute(0, 4, 2, 3, 1, 5).reshape(k_size, n)
    pairs = q4.reshape(*q4.shape[:-1], -1, 32)
    packed = ((pairs[..., :16] & 15) | (pairs[..., 16:] << 4)).flatten(-2)
    # IxInfer applies the code multiplier in the kernel.
    return packed, scale.T


class IluNativeToW4A8MoEMethod(MoEComputeBindingMixin, FusedMoEMethodBase):
    compute_source = "native W4A8 experts"

    def __init__(self, moe, selection):
        super().__init__(moe)
        self.selection = selection
        self._compute = self._runtime_weights = None
        self._loaded = set()

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if self.moe.has_bias or not self.moe.is_act_and_mul:
            raise ValueError("Native W4A8 requires gated experts without bias")
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Native W4A8 requires BF16/FP16 expert weights")
        hidden, intermediate = hidden_size, intermediate_size_per_partition
        backend = get_moe_backend(self.selection)
        layout = self.selection.layout
        error = backend.support_error(self._compute_config(layer), self.selection)
        if error is not None:
            raise ValueError(error)
        spec = backend.weight_specs[layout]
        if layout == "NN" and intermediate % 32:
            raise ValueError("IxInfer NN gate/up shards must be 32-aligned")
        for group, n, k in (
            ("w13", 2 * intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
            weight_shape, scale_shape = spec.shapes(num_experts, n, k)
            weight = torch.nn.Parameter(
                torch.empty(weight_shape, dtype=spec.weight_dtype), requires_grad=False
            )
            scale = torch.nn.Parameter(
                torch.empty(scale_shape, dtype=spec.scale_dtype), requires_grad=False
            )
            scale.ilu_derived_at_load = True
            layer.register_parameter(f"{group}_weight", weight)
            layer.register_parameter(f"{group}_weight_scale", scale)
            weight.weight_loader = self._make_loader(
                layer, group, scale, hidden, intermediate
            )
        self._num_experts = num_experts

    def _make_loader(self, layer, group, scale, hidden, intermediate):
        layout = self.selection.layout
        tp = self.moe.moe_parallel_config.tp_size
        rank = self.moe.tp_rank

        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            local = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local < 0:
                return False if return_success else None
            if loaded_weight.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(f"{weight_name} must contain BF16/FP16 weights")
            is_gate_up = group == "w13"
            if is_gate_up:
                if shard_id not in ("w1", "w3") or loaded_weight.shape != (
                    intermediate * tp,
                    hidden,
                ):
                    raise ValueError(
                        f"Invalid gate/up expert shard {weight_name}: {loaded_weight.shape}"
                    )
                source = loaded_weight.narrow(0, rank * intermediate, intermediate)
                offset = 0 if shard_id == "w1" else intermediate
                target = param.data[local]
                if layout == "NN":
                    target = target.narrow(1, offset // 2, intermediate // 2)
                else:
                    target = target.narrow(0, offset, intermediate)
                target_scale = scale.data[local].narrow(1, offset, intermediate)
                k_start, k_size = 0, hidden
            else:
                if shard_id != "w2" or loaded_weight.shape != (
                    hidden,
                    intermediate * tp,
                ):
                    raise ValueError(
                        f"Invalid down expert shard {weight_name}: {loaded_weight.shape}"
                    )
                source = loaded_weight
                target, target_scale = param.data[local], scale.data[local]
                k_start, k_size = rank * intermediate, intermediate
            packed, channel_scale = _pack_weight(
                source.to(param.device), layout, k_start, k_size
            )
            target.copy_(packed)
            target_scale.copy_(channel_scale)
            self._loaded.add((local, shard_id))
            return True if return_success else None

        return load

    def process_weights_after_loading(self, layer):
        from vllm.config import get_current_vllm_config

        if get_current_vllm_config().load_config.load_format != "dummy":
            expected = 3 * self._num_experts
            if len(self._loaded) != expected:
                raise ValueError(
                    f"{layer.layer_name}: loaded {len(self._loaded)} of {expected} expert shards"
                )
        self._bind_compute(layer)
        self._loaded.clear()
