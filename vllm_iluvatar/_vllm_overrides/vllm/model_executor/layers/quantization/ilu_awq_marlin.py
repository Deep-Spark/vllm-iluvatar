# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import ixformer.inference.functions as ixfops
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.nn import Parameter
from transformers import PretrainedConfig
from vllm import _custom_ops as ops
from vllm import envs
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    verify_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# AWQ uses a non-standard packing order within int32 values.
# For 4-bit: standard order stores values at bit positions [0,4,8,12,16,20,24,28]
# for indices [0,1,2,3,4,5,6,7], while AWQ stores them for indices
# [0,4,1,5,2,6,3,7]. This permutation reverses that ordering.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


class IluAWQMarlinConfig(QuantizationConfig):
    """Config class for AWQ Marlin"""

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None,
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config

        if self.weight_bits not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported num_bits = {self.weight_bits}. "
                f"Supported num_bits = {self.TYPE_MAP.keys()}"
            )

        self.quant_type = self.TYPE_MAP[self.weight_bits]

        verify_marlin_supported(
            self.quant_type, group_size=self.group_size, has_zp=self.zero_point
        )

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "awq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        # Skip override to marlin kernels, as they are not
        # batch invariant
        if envs.VLLM_BATCH_INVARIANT:
            return None

        can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "awq_marlin"
        )

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info(
                "Detected that the model can run with awq_marlin"
                ", however you specified quantization=awq explicitly,"
                " so forcing awq. Use quantization=awq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            # Check if the layer is supported by AWQMarlin.
            if not check_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    "Layer '%s' is not supported by AWQMarlin. Falling back to unoptimized AWQ kernels.",  # noqa: E501
                    prefix,
                )
                return AWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            quant_method = IluAWQMarlinLinearMethod(self)
            quant_method.input_dtype = get_marlin_input_dtype(prefix)
            return quant_method
        elif isinstance(layer, RoutedExperts):
            # from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            # if is_layer_skipped(
            #     prefix,
            #     getattr(self, "modules_to_not_convert", []),
            #     skip_with_substr=True,
            # ):
            #     return UnquantizedFusedMoEMethod(layer.moe_config)
            # if not check_moe_marlin_supports_layer(layer, self.group_size):
            #     logger.warning_once(
            #         f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
            #         "Falling back to Moe WNA16 kernels."
            #     )
            #     return MoeWNA16Config.from_config(self.full_config).get_quant_method(
            #         layer, prefix
            #     )
            # moe_quant_method = AWQMarlinMoEMethod(self, layer.moe_config)
            # moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            # return moe_quant_method
            return IluAWQMarlinMoEMethod(self, layer.moe_config)
        return None

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if not current_platform.is_cuda_alike():
            return False

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class IluAWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ Marlin.

    Uses choose_mp_linear_kernel to select the best available kernel
    (Conch, Exllama, or Marlin) for the current platform.

    Args:
        quant_config: The AWQ Marlin quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: IluAWQMarlinConfig) -> None:
        self.quant_config = quant_config
        self.quant_type = scalar_types.uint4
        self.input_dtype = None

        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
            has_zp=self.quant_config.zero_point,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size
        layer.num_groups = num_groups

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        layer.output_size_per_partition = (
            layer.qweight.data.shape[1] * self.quant_config.pack_factor
        )
        align_bits = 64 * 8
        align_size = align_bits // self.quant_config.weight_bits
        if layer.output_size_per_partition % align_size != 0:
            padding_output_size_per_partition = (
                (layer.output_size_per_partition + align_size - 1)
                // align_size
                * align_size
            )
            layer.output_padding_size = (
                padding_output_size_per_partition - layer.output_size_per_partition
            )
            device = layer.qweight.device

            pad_qweight = torch.zeros(
                layer.input_size_per_partition,
                padding_output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device=device,
            )
            pad_qzeros = torch.zeros(
                layer.num_groups,
                padding_output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device=device,
            )
            pad_scales = torch.zeros(
                layer.num_groups,
                padding_output_size_per_partition,
                dtype=layer.scales.data.dtype,
                device=device,
            )
            pad_qweight[
                ..., : layer.output_size_per_partition // self.quant_config.pack_factor
            ] = layer.qweight.data
            pad_qzeros[
                ..., : layer.output_size_per_partition // self.quant_config.pack_factor
            ] = layer.qzeros.data
            pad_scales[..., : layer.output_size_per_partition] = layer.scales.data
            replace_parameter(layer, "qweight", pad_qweight)
            replace_parameter(layer, "qzeros", pad_qzeros)
            replace_parameter(layer, "scales", pad_scales)
        device = layer.qweight.device
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        return
        # TODO(gyf) Marlin format is not support for now..

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = ops.awq_gemm(
            reshaped_x,
            qweight,
            scales,
            qzeros,
            pack_factor,
            group_size=self.quant_config.group_size,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class IluAWQMarlinMoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: IluAWQMarlinConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMarlinMoEMethod only supports 4bit now.")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        self.use_marlin = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full", intermediate_size_per_partition
        )
        self.is_k_full = intermediate_size_per_partition == intermediate_size_full

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size
        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        device = layer.w13_qweight.device
        layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return
        # TODO(gyf) Marlin format is not support for now..

    def awq_marlin_moe_quant_config(
        self,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w1_zp: torch.Tensor | None,
        w2_zp: torch.Tensor | None,
        weight_bits: int,
        group_size: int,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        a1_gscale: torch.Tensor | None = None,
        a2_gscale: torch.Tensor | None = None,
    ) -> FusedMoEQuantConfig:
        """
        Construct a quant config for awq marlin quantization.

        a1_gscale / a2_gscale are optional global scales applied to activation
        quantization scales when Marlin runs with 8-bit activations.
        """
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

        w_shape = None if group_size == -1 else GroupShape(row=1, col=group_size)

        # Activations are NOT quantized for AWQ (fp16/bf16)
        a_shape = w_shape  # Same as weight shape for alignment

        # Determine weight dtype
        if weight_bits == 4:
            weight_dtype = "int4"
        elif weight_bits == 8:
            weight_dtype = torch.int8
        else:
            raise ValueError(f"Unsupported weight_bits: {weight_bits}")

        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(dtype=None, shape=a_shape, alpha_or_gscale=a1_gscale),
            _a2=FusedMoEQuantDesc(dtype=None, shape=a_shape, alpha_or_gscale=a2_gscale),
            _w1=FusedMoEQuantDesc(
                weight_dtype, w_shape, w1_scale, None, w1_zp, w1_bias
            ),
            _w2=FusedMoEQuantDesc(
                weight_dtype, w_shape, w2_scale, None, w2_zp, w2_bias
            ),
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:

        return self.awq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=(
                getattr(layer, "w13_qzeros", None)
                if self.quant_config.zero_point
                else None
            ),
            w2_zp=(
                getattr(layer, "w2_qzeros", None)
                if self.quant_config.zero_point
                else None
            ),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: torch.nn.Module,
    ):
        """
        Select the GEMM implementation for AWQ-Marlin MoE.
        Returns MarlinExperts configured for AWQ quantization.
        This is ONLY used when LoRA is enabled.
        Without LoRA, AWQ uses its own apply() method.
        """
        # Only use modular kernels when LoRA is enabled
        # Without LoRA, AWQ's own apply() method works fine and is more efficient
        if not self.moe.is_lora_enabled:
            raise NotImplementedError(
                "AWQ-Marlin uses its own apply() method when LoRA is not enabled. "
                "Modular kernels are only used for LoRA support."
            )

        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
            MarlinExperts,
        )

        # Ensure quant config is initialized
        assert (
            self.moe_quant_config is not None
        ), "moe_quant_config must be initialized before select_gemm_impl"

        w13_g_idx = getattr(layer, "w13_g_idx", None)
        w2_g_idx = getattr(layer, "w2_g_idx", None)
        w13_g_idx_sort_indices = getattr(layer, "w13_g_idx_sort_indices", None)
        w2_g_idx_sort_indices = getattr(layer, "w2_g_idx_sort_indices", None)

        # Check if using batched expert format (for Expert Parallelism)
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            # For batched format, use BatchedMarlinExperts
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None
            return BatchedMarlinExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )
        else:
            # Standard Marlin experts for AWQ
            return MarlinExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
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
        assert layer.activation.value == "silu", "Only SiLU activation is supported."
        use_ep = layer.expert_map is not None
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata:
            if isinstance(attn_metadata, dict):
                only_decode = use_ep == False and all(
                    t.num_decodes > 0 and t.num_prefills == 0
                    for t in list(attn_metadata.values())
                )
            else:
                only_decode = (
                    use_ep == False
                    and attn_metadata.num_decodes > 0
                    and attn_metadata.num_prefills == 0
                )
        else:
            only_decode = False
        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min(
                (layer.ep_rank + 1) * layer.local_num_experts, layer.global_num_experts
            )
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused Marlin MoE method."
            )

        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts
        if use_ep:
            hidden_size = x.shape[1]
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
                expand_tokens,
            ) = ixfops.moe_compute_token_index_ep(
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_eid,
                end_expert_id=end_eid,
            )
            if expert_sizes_cpu.sum() == 0:
                return torch.zeros(
                    (num_tokens, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            expand_tokens = num_tokens * layer.top_k
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
            ) = ixfops.moe_compute_token_index(
                topk_ids=topk_ids,
                num_experts=num_experts,
            )

        # expand + reorder
        # TODO use kernel
        expand_hidden_states = ixfops.moe_expand_input(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
        )
        if only_decode:
            pt_output_1 = ixfops.moe_w4a16_group_gemv(
                input=expand_hidden_states,
                weight=layer.w13_qweight,
                w_scales=layer.w13_scales,
                quant_type="awq",
                w_zeros=layer.w13_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=None,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )
            pt_output_2 = ixfops.silu_and_mul(pt_output_1)

            pt_output_3 = ixfops.moe_w4a16_group_gemv(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )

            # mul + reduce_sum
            # final_hidden_states: (num_tokens, k)
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=layer.runner_scaling_factor,
                extra_residual=shared_experts_input,
            )
        else:
            expert_sizes_cpu = expert_sizes_gpu.cpu()
            # w4a16 group gemm 1
            # pt_output_1: (expand_tokens, 2n) dtype
            pt_output_1 = ixfops.moe_w4a16_group_gemm(
                input=expand_hidden_states,
                weight=layer.w13_qweight,
                w_scales=layer.w13_scales,
                quant_type="awq",
                tokens_per_experts=expert_sizes_cpu,
                w_zeros=layer.w13_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=None,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )

            # act
            pt_output_2 = ixfops.silu_and_mul(pt_output_1)

            # w4a16 group gemm 2 + reorder
            # pt_output_3: (expand_tokens, k) dtype
            if use_ep:
                pt_output_3 = torch.empty(
                    (num_tokens * layer.top_k, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )

                ixfops.moe_w4a16_group_gemm(
                    input=pt_output_2,
                    weight=layer.w2_qweight,
                    w_scales=layer.w2_scales,
                    quant_type="awq",
                    tokens_per_experts=expert_sizes_cpu,
                    w_zeros=layer.w2_qzeros,
                    group_size=self.quant_config.group_size,
                    dst_to_src=sorted_token_ids,
                    format="NN",
                    output=pt_output_3,
                )

                reduce_mask = src_to_dst == -1
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=layer.runner_scaling_factor,
                    mask=reduce_mask,
                    extra_residual=shared_experts_input,
                )
            else:
                pt_output_3 = ixfops.moe_w4a16_group_gemm(
                    input=pt_output_2,
                    weight=layer.w2_qweight,
                    w_scales=layer.w2_scales,
                    quant_type="awq",
                    tokens_per_experts=expert_sizes_cpu,
                    w_zeros=layer.w2_qzeros,
                    group_size=self.quant_config.group_size,
                    dst_to_src=sorted_token_ids,
                    format="NN",
                    tokens_per_experts_gpu=expert_sizes_gpu,
                )

                # mul + reduce_sum
                # final_hidden_states: (num_tokens, k)
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=layer.runner_scaling_factor,
                    extra_residual=shared_experts_input,
                )
        return final_hidden_states
