# SPDX-License-Identifier: Apache-2.0
"""Iluvatar serialized FP4-E2M1 weight / FP8-E4M3 activation MoE."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_f4f8_experts import (
    F4F8_RUNNER_ALL2ALL_BACKENDS,
    IluF4F8Experts,
    ilu_f4f8_moe_backends_available,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    IluQuantConfig,
)

_GROUP = 256
logger = init_logger(__name__)


def _absolute_to_running_scale_(scale: torch.Tensor) -> None:
    for group in range(scale.shape[-2] - 1):
        scale[..., group, :].div_(scale[..., group + 1, :])


class IluF4F8MoEMethod(FusedMoEMethodBase):
    """Load kernel-packed U8 experts and execute ix30 F4F8 grouped GEMM."""

    def __init__(self, quant_config: IluF4F8Config, layer: RoutedExperts):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self._weights_are_raw = True
        self._experts: IluF4F8Experts | None = None
        self._validate_config(layer)
        logger.info_once(
            "IluF4F8MoEMethod: serialized FP4-E2M1/group256 weights, "
            "dynamic per-token FP8-E4M3 activations, ixmm backend"
        )

    def _validate_config(self, layer: RoutedExperts) -> None:
        section = self.quant_config.routed_section
        expected = {
            "quant_dtype": "fp4_e2m1",
            "group_size": _GROUP,
            "storage_dtype": "uint8",
            "weight_layout": "ix30_kmajor_pack32_v1",
            "scale_dtype": "float32",
            "scale_form": "absolute",
        }
        for key, value in expected.items():
            if section.get(key) != value:
                raise ValueError(
                    f"ilu_f4f8 requires routed_experts.{key}={value!r}, "
                    f"got {section.get(key)!r}"
                )
        activation = section.get("input_activations", {})
        if (
            activation.get("quant_dtype") != "fp8_e4m3fn"
            or activation.get("strategy") != "token"
            or activation.get("dynamic") is not True
        ):
            raise ValueError(
                "ilu_f4f8 requires dynamic per-token FP8-E4M3FN activations"
            )
        if self.moe.in_dtype != torch.bfloat16:
            raise ValueError("ilu_f4f8 requires BF16 model activations")
        if self.moe.activation != MoEActivation.SILU:
            raise ValueError("ilu_f4f8 requires SILU routed experts")
        if self.moe.has_bias:
            raise ValueError("ilu_f4f8 v1 does not support expert bias")
        if self.moe.is_lora_enabled:
            raise ValueError("ilu_f4f8 v1 does not support LoRA")
        parallel = self.moe.moe_parallel_config
        if parallel.pcp_size != 1:
            raise ValueError("ilu_f4f8 requires PCP=1")
        if parallel.enable_eplb:
            raise ValueError("ilu_f4f8 v1 does not support EPLB")
        if parallel.all2all_backend not in F4F8_RUNNER_ALL2ALL_BACKENDS:
            raise ValueError(
                "ilu_f4f8 supports runner-managed allgather_reducescatter only; "
                f"all2all_backend={parallel.all2all_backend!r} is not integrated"
            )
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.parallel_config.enable_dbo:
            raise ValueError("ilu_f4f8 does not support DBO")
        supported, reason = ilu_f4f8_moe_backends_available()
        if not supported:
            raise RuntimeError(reason)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        inter = intermediate_size_per_partition
        if hidden_size % _GROUP or inter % _GROUP:
            raise ValueError(
                "ilu_f4f8 requires hidden_size and local intermediate_size "
                "to be divisible by 256"
            )
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = inter
        self._loader_layer = layer
        f4f8_weight_attrs = dict(extra_weight_attrs)
        f4f8_weight_attrs["weight_loader"] = self.weight_loader

        params = {
            "w13_weight": torch.empty(
                num_experts,
                hidden_size // 2,
                2 * inter,
                dtype=torch.uint8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                inter // 2,
                hidden_size,
                dtype=torch.uint8,
            ),
            "w13_weight_scale": torch.empty(
                num_experts,
                hidden_size // _GROUP,
                2 * inter,
                dtype=torch.float32,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                inter // _GROUP,
                hidden_size,
                dtype=torch.float32,
            ),
        }
        for name, data in params.items():
            parameter = torch.nn.Parameter(data, requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, f4f8_weight_attrs)
        layer.w13_bias = None
        layer.w2_bias = None
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_id = self._loader_layer._map_global_expert_id_to_local_expert_id(
            expert_id
        )
        if local_expert_id == -1:
            return False if return_success else None
        self.load_expert_tensor(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            local_expert_id=local_expert_id,
            tp_rank=self.moe.tp_rank,
            tp_size=self.moe.tp_size,
        )
        return True if return_success else None

    def load_expert_tensor(
        self,
        *,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        local_expert_id: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        is_scale = "weight_scale" in weight_name
        expected_dtype = torch.float32 if is_scale else torch.uint8
        if loaded_weight.dtype != expected_dtype:
            raise TypeError(
                f"{weight_name} must contain {expected_dtype}, got {loaded_weight.dtype}"
            )
        inter = self.intermediate_size_per_partition
        hidden = self.hidden_size
        expert_data = param.data[local_expert_id]

        if shard_id in ("w1", "w3"):
            expected = (
                (hidden // _GROUP, inter * tp_size)
                if is_scale
                else (hidden // 2, inter * tp_size)
            )
            if tuple(loaded_weight.shape) != expected:
                raise ValueError(
                    f"invalid F4F8 {shard_id} tensor shape: "
                    f"expected {expected}, got {tuple(loaded_weight.shape)}"
                )
            source = loaded_weight.narrow(1, tp_rank * inter, inter)
            destination_offset = 0 if shard_id == "w1" else inter
            expert_data.narrow(1, destination_offset, inter).copy_(source)
        elif shard_id == "w2":
            rows_per_rank = inter // (_GROUP if is_scale else 2)
            expected = (rows_per_rank * tp_size, hidden)
            if tuple(loaded_weight.shape) != expected:
                raise ValueError(
                    f"invalid F4F8 w2 tensor shape: expected {expected}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            source = loaded_weight.narrow(0, tp_rank * rows_per_rank, rows_per_rank)
            expert_data.copy_(source)
        else:
            raise ValueError(f"invalid F4F8 expert shard id: {shard_id}")
        self._weights_are_raw = True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise TypeError("ilu_f4f8 weights must be uint8")
        if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
            raise TypeError("ilu_f4f8 scales must be float32")
        if not all(t.is_contiguous() for t in (w13, w2, w13_scale, w2_scale)):
            raise ValueError("ilu_f4f8 weights and scales must be contiguous")

        experts, packed_hidden, width = w13.shape
        hidden = packed_hidden * 2
        inter = width // 2
        expected = {
            "w2_weight": (experts, inter // 2, hidden),
            "w13_weight_scale": (experts, hidden // _GROUP, 2 * inter),
            "w2_weight_scale": (experts, inter // _GROUP, hidden),
        }
        actual = {
            "w2_weight": tuple(w2.shape),
            "w13_weight_scale": tuple(w13_scale.shape),
            "w2_weight_scale": tuple(w2_scale.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(
                    f"invalid {name} shape: expected {shape}, got {actual[name]}"
                )
        if hidden % _GROUP or inter % _GROUP:
            raise ValueError("ilu_f4f8 runtime K dimensions must divide 256")
        for name, scale in (
            ("w13_weight_scale", w13_scale),
            ("w2_weight_scale", w2_scale),
        ):
            if not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
                raise ValueError(f"{name} must contain finite positive raw scales")

        if self._weights_are_raw:
            _absolute_to_running_scale_(w13_scale.data)
            _absolute_to_running_scale_(w2_scale.data)
            self._weights_are_raw = False

        self._experts = IluF4F8Experts(
            getattr(layer, "swiglu_limit", self.moe.swiglu_limit)
        )

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # IluMoERunner performs AG/RS once around this local-expert method.
        return None

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
            _fp8_moe_clamp_limit,
        )

        clamp_limit = _fp8_moe_clamp_limit(layer)
        return mxfp4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=clamp_limit,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> FusedMoEExpertsModular:
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.batch_experts import (
            IluMoEBatchedExpertsFp8,
        )
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
            _shrink_ll_max_tokens_to_decode,
        )
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
            _get_ilu_output_scale,
        )

        if (
            prepare_finalize.activation_format
            != FusedMoEActivationFormat.BatchedExperts
        ):
            raise NotImplementedError(
                "IluMoEBatchedExpertsFp8 requires the batched activation format, "
                f"got {prepare_finalize.activation_format}"
            )
        _shrink_ll_max_tokens_to_decode(prepare_finalize)
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        if max_num_tokens is None:
            raise ValueError("batched FP8 experts require max_num_tokens")
        if self.moe_quant_config is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        return IluMoEBatchedExpertsFp8(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            swiglu_limit=self.moe_quant_config.gemm1_clamp_limit,
            runner_scaling_factor=_get_ilu_output_scale(layer),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._experts is None:
            raise RuntimeError("ilu_f4f8 weights have not been processed")
        if layer.apply_router_weight_on_input:
            raise ValueError("ilu_f4f8 requires router weights on expert outputs")
        return self._experts.apply(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=layer.expert_map,
            shared_expert_output=shared_experts_input,
        )


@register_quantization_config("ilu_f4f8")
class IluF4F8Config(IluQuantConfig):
    """Model-independent mixed checkpoint with routed ix30 F4F8 experts."""

    @classmethod
    def get_name(cls) -> str:
        return "ilu_f4f8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluF4F8Config:
        if config.get("format") != "ilu-mixed-f4f8" or config.get("version") != 1:
            raise ValueError("ilu_f4f8 requires ilu-mixed-f4f8 format version 1")
        groups = {}
        for name in ("dense_linears", "shared_experts"):
            section = config.get(name) or {}
            if not section:
                continue
            targets = (
                [r"re:.*\.shared_expert(?:s)?(?:\.|$)"]
                if name == "shared_experts"
                else list(section["targets"])
            )
            weights = {
                key: section[key]
                for key in ("num_bits", "type", "symmetric", "strategy", "group_size")
                if key in section
            }
            weights["dynamic"] = False
            groups[name] = {
                "format": "int-quantized",
                "targets": targets,
                "weights": weights,
                "input_activations": dict(section["input_activations"]),
                "output_activations": None,
                "iluvatar": {
                    "gemm_format": "TN",
                    "mixed_strategy": "CHANNEL",
                    "gemm_mode": "PERF",
                },
            }
        ignore = list(
            dict.fromkeys(
                list(config.get("ignore") or [])
                + list(config.get("bf16_layers") or [])
                + ["lm_head"]
            )
        )
        result = super().from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "iluvatar-mixed",
                "config_groups": groups,
                "ignore": ignore,
            }
        )
        result.routed_section = dict(config.get("routed_experts") or {})
        return result

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: Any,
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        del user_quant, hf_config
        if (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") == cls.get_name()
        ):
            return cls.get_name()
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return IluF4F8MoEMethod(self, layer)
        return super().get_quant_method(layer, prefix)


__all__ = [
    "IluF4F8Config",
    "IluF4F8MoEMethod",
]
