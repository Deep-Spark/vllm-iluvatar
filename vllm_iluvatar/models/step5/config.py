"""Step5 text/VL configuration registered out of tree."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from transformers import PretrainedConfig
from vllm.transformers_utils.configs.step3p5 import Step3p5Config


class PerceptionEncoderConfig(PretrainedConfig):
    model_type = "perception_encoder"


class Step5Config(Step3p5Config):
    model_type = "step5"
    architectures = ["Step5ForCausalLM"]

    def __init__(
        self,
        *,
        partial_rotary_factors: list[float] | None = None,
        zero_centered: bool = True,
        fp32_residual_connection: bool = False,
        num_nextn_predict_layers: int = 0,
        layer_types: list[str] | None = None,
        max_position_embeddings: int | None = None,
        max_seq_len: int = 4096,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        quantization_config = kwargs.get("quantization_config")
        if isinstance(quantization_config, dict):
            quantization_config = copy.deepcopy(quantization_config)
            ignored = quantization_config.get("modules_to_not_convert")
            if isinstance(ignored, list):
                mapped: list[str] = []
                for name in ignored:
                    mapped.append(name.replace(".share_expert.", ".moe.share_expert."))
                    if name.endswith(".self_attn.qkv_indexer_proj"):
                        base = name.removesuffix("qkv_indexer_proj")
                        mapped.extend(
                            base + suffix
                            for suffix in (
                                "sparse_indexer_q",
                                "sparse_indexer_k",
                                "sparse_indexer_z",
                                "sparse_indexer_w",
                            )
                        )
                quantization_config["modules_to_not_convert"] = mapped
            kwargs["quantization_config"] = quantization_config
        total_layers = int(kwargs.get("num_hidden_layers", 48))
        serialized = kwargs.pop("layer_types_with_mtp", None)
        if serialized is not None:
            layer_types = list(serialized)
        if layer_types is not None:
            expected = total_layers + int(num_nextn_predict_layers)
            if len(layer_types) < total_layers:
                raise ValueError(
                    f"Step5 layer_types must cover the dense stack: expected at least {total_layers}, got {len(layer_types)}"
                )
            self.layer_types_with_mtp = list(layer_types[:expected])
            dense_layer_types = list(layer_types[:total_layers])
        else:
            self.layer_types_with_mtp = None
            dense_layer_types = None
        if max_position_embeddings is None:
            max_position_embeddings = max_seq_len
        super().__init__(
            layer_types=dense_layer_types,
            num_nextn_predict_layers=num_nextn_predict_layers,
            max_position_embeddings=max_position_embeddings,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        self.partial_rotary_factors = partial_rotary_factors
        self.zero_centered = bool(zero_centered)
        self.fp32_residual_connection = bool(fp32_residual_connection)
        self.tie_word_embeddings = bool(tie_word_embeddings)


class Step5MTPConfig(Step5Config):
    model_type = "step5_mtp"
    architectures = ["Step5MTP"]

    @classmethod
    def from_target(cls, target: PretrainedConfig) -> Step5MTPConfig:
        """Copy the text config without applying checkpoint name remapping twice."""
        text = getattr(target, "text_config", target)
        if not isinstance(text, Step5Config):
            raise TypeError("Step5 MTP requires a Step5 text configuration")
        count = text.num_nextn_predict_layers
        if count < 1:
            raise ValueError("Step5 checkpoint has no MTP layers")
        if (
            text.layer_types_with_mtp is None
            or len(text.layer_types_with_mtp) != text.num_hidden_layers + count
        ):
            raise ValueError("Step5 MTP requires all main and draft layer types")
        result = copy.deepcopy(text)
        result.__class__ = cls
        result.model_type = cls.model_type
        result.architectures = list(cls.architectures)
        result.n_predict = count
        result.quantization_config = None
        return result


class Step5VLConfig(Step5Config):
    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "vision_config": PerceptionEncoderConfig,
        "text_config": Step5Config,
    }

    def __init__(
        self,
        vision_config: dict[str, Any] | PerceptionEncoderConfig | None = None,
        text_config: dict[str, Any] | Step5Config | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(vision_config, dict):
            vision = PerceptionEncoderConfig(**vision_config)
        elif vision_config is None:
            vision = PerceptionEncoderConfig()
        else:
            vision = vision_config
        if isinstance(text_config, dict):
            text = Step5Config(**text_config)
        elif text_config is None:
            text = Step5Config()
        else:
            text = text_config
        text_values = text.to_dict()
        text_values.pop("model_type", None)
        text_values.pop("architectures", None)
        text_values.update(kwargs)
        super().__init__(**text_values)
        self.vision_config = vision
        self.text_config = text
        self.hidden_size = text.hidden_size
        if getattr(self, "quantization_config", None) is None:
            self.quantization_config = getattr(text, "quantization_config", None)


def _register_configs() -> None:
    from transformers import AutoConfig
    from vllm.tokenizers import registry as tokenizer_registry
    from vllm.transformers_utils import config as vllm_config_utils

    for config_cls in (Step5Config, Step5MTPConfig, Step5VLConfig):
        AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)
        vllm_config_utils._CONFIG_REGISTRY[config_cls.model_type] = config_cls
    tokenizer_registry._MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS.update(
        {"step5", "step5_vl", "step5_mtp"}
    )


_register_configs()
__all__ = ["Step5Config", "Step5MTPConfig", "Step5VLConfig"]
