# SPDX-License-Identifier: Apache-2.0
"""Translate the VL checkpoint namespace before vLLM constructs text layers."""

from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import IluFp8Config

PACKED_LINEAR_PROJECTIONS = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
}


class FlashFp8Config(IluFp8Config):
    @classmethod
    def from_text_config(cls, text):
        cfg = cls.from_config(text.quantization_config)
        if (
            not cfg.is_checkpoint_fp8_serialized
            or cfg.store_dtype not in (None, "fp8")
            or cfg.weight_block_size != [128, 128]
            or cfg.activation_scheme != "dynamic"
            or text.quantization_config.get("fmt", "e4m3") != "e4m3"
        ):
            raise ValueError(
                "Flash requires serialized 128x128 block-FP8 and dynamic activations"
            )
        cfg.ignored_layers = [
            name.removeprefix("language_model.model.") for name in cfg.ignored_layers
        ]
        cfg.packed_modules_mapping = PACKED_LINEAR_PROJECTIONS
        return cfg
