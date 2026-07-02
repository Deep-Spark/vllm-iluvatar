# SPDX-License-Identifier: Apache-2.0
"""Iluvatar compressed-tensors MoE method-selection wrapper.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/compressed_tensors_moe.py

Target   : CompressedTensorsMoEMethod.get_moe_method
Baseline : main_93d8f834dd8
Why      : preserve the fork's Iluvatar W4A8/W8A8 MoE method dispatch without
           editing community vLLM source.
How      : ``patch_compressed_tensors_moe`` installs this staticmethod wrapper;
           Iluvatar-only schemes return plugin methods, all other schemes call
           the original community method.
Scope    : late general-plugin patch during ``register_general()``.
Safety   : imports W4A8/W8A8 implementations lazily inside selected branches so
           ordinary compressed-tensors configs keep using community behavior.
Remove   : when community compressed-tensors MoE exposes an official backend
           selection hook for vendor methods.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from compressed_tensors import CompressionFormat

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    WNA16_SUPPORTED_BITS,
)
from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)

_ORIGINAL_GET_MOE_METHOD: Callable | None = None

_SCHEME_DISCOVERY_PREFIX_BASELINE = """\
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")
"""


def _extract_scheme_discovery_prefix(fn: Callable) -> str:
    source_lines = inspect.getsource(fn).splitlines()
    start_marker = "quant_config._add_fused_moe_to_target_scheme_map()"
    end_marker = 'format = scheme_dict.get("format")'

    try:
        start = next(
            idx for idx, line in enumerate(source_lines) if start_marker in line
        )
        end = next(idx for idx, line in enumerate(source_lines) if end_marker in line)
    except StopIteration as exc:
        raise RuntimeError(
            "Unable to locate compressed-tensors MoE scheme discovery prefix "
            "in community get_moe_method; resync the Iluvatar wrapper."
        ) from exc

    return "\n".join(source_lines[start : end + 1])


def check_scheme_discovery_fingerprint(original_get_moe_method: Callable) -> None:
    p.check_fingerprint(
        "CompressedTensorsMoEMethod.get_moe_method scheme discovery prefix",
        _extract_scheme_discovery_prefix(original_get_moe_method),
        _SCHEME_DISCOVERY_PREFIX_BASELINE,
        resync_hint=(
            "Resync IluCompressedTensorsMoEMethod.get_moe_method's copied "
            "scheme discovery prefix before selecting Iluvatar MoE methods."
        ),
    )


def set_original_get_moe_method(fn: Callable) -> None:
    global _ORIGINAL_GET_MOE_METHOD
    _ORIGINAL_GET_MOE_METHOD = fn


def _fallback_get_moe_method(quant_config, layer: torch.nn.Module, layer_name: str):
    if _ORIGINAL_GET_MOE_METHOD is None:
        raise RuntimeError("Original CompressedTensorsMoEMethod.get_moe_method not set")
    return _ORIGINAL_GET_MOE_METHOD(quant_config, layer, layer_name)


def _env_enabled(canonical: str, legacy: str) -> bool:
    return bool(getattr(envs, canonical, False) or getattr(envs, legacy, False))


class IluCompressedTensorsMoEMethod:
    """Wrapper surface installed onto community CompressedTensorsMoEMethod."""

    @staticmethod
    def get_moe_method(
        quant_config,
        layer: torch.nn.Module,
        layer_name: str,
    ):
        # Match the community method's scheme discovery so Iluvatar branches are
        # chosen from the same fused-MoE projection metadata.
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:
            logger.info_once(
                "Using unquantized MoE method for ignored compressed-tensors MoE layer"
            )
            return UnquantizedFusedMoEMethod(layer.moe_config)

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format_ = scheme_dict.get("format")

        if _env_enabled("VLLM_ILU_W8A8_MOE_USE_W4A8", "VLLM_W8A8_MOE_USE_W4A8"):
            from .ilu_compressed_tensors_moe_w4a8_int8 import (
                IluCompressedTensorsW4A8Int8MoEMethod,
            )

            logger.info_once("Using Iluvatar W4A8 compressed-tensors MoE method")
            return IluCompressedTensorsW4A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        if (
            weight_quant is not None
            and weight_quant.num_bits == 4
            and input_quant is not None
            and input_quant.num_bits == 8
        ):
            from .ilu_compressed_tensors_moe_w4a8_int8 import (
                IluCompressedTensorsW4A8Int8MoEMethod,
            )

            logger.info_once("Using Iluvatar W4A8 compressed-tensors MoE method")
            return IluCompressedTensorsW4A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        if _env_enabled("VLLM_ILU_WNA16_MOE_USE_W4A8", "VLLM_WNA16_MOE_USE_W4A8"):
            if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
                group_size = weight_quant.group_size or -1
                valid_format_and_bits = (
                    weight_quant.num_bits in WNA16_SUPPORTED_BITS
                    and format_ == CompressionFormat.pack_quantized.value
                )
                if not valid_format_and_bits:
                    raise ValueError(
                        "For Fused MoE layers, only format: ",
                        f"{CompressionFormat.pack_quantized.value} ",
                        f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                        f"but got format: {format_} "
                        f" and bits: {weight_quant.num_bits}",
                    )

                assert weight_quant.num_bits == 4 and group_size == -1, (
                    "VLLM_ILU_WNA16_MOE_USE_W4A8 only supports channel-wise "
                    "W4A16 -> W4A8 MoE"
                )
                from .ilu_compressed_tensors_moe_w4a8_int8 import (
                    IluCompressedTensorsW4A8Int8MoEMethod,
                )

                logger.info_once(
                    "Using Iluvatar WNA16->W4A8 compressed-tensors MoE method"
                )
                return IluCompressedTensorsW4A8Int8MoEMethod(
                    weight_quant, input_quant, layer.moe_config
                )

        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            from .ilu_compressed_tensors_moe_w8a8_int8 import (
                IluCompressedTensorsW8A8Int8MoEMethod,
            )

            logger.info_once("Using Iluvatar W8A8 compressed-tensors MoE method")
            return IluCompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        return _fallback_get_moe_method(quant_config, layer, layer_name)

