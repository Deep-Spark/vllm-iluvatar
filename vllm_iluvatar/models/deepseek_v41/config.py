# SPDX-License-Identifier: Apache-2.0
"""HF config for DeepSeek-V4.1-Flash (``model_type="deepseek_v41"``)."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

# V4 decoder reads this; V4.1 routes every backbone layer through the gate.
_V41_DEFAULTS: dict[str, Any] = {
    "num_hash_layers": 0,
}

# Official ``ModelArgs`` name -> HF ``text_config`` name.
_OFFICIAL_ALIASES: dict[str, str] = {
    "rope_head_dim": "qk_rope_head_dim",
    "window_size": "sliding_window",
    "norm_eps": "rms_norm_eps",
}

# ``ModelArgs`` name -> key inside ``rope_scaling``
_ROPE_ALIASES: dict[str, str] = {
    "original_seq_len": "original_max_position_embeddings",
    "rope_factor": "factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
}


class DeepseekV41Config(PretrainedConfig):
    """Flattened view of the nested DeepSeek-V4.1 config.

    ``text_config`` is hoisted onto ``self``; ``vision_config`` is kept as a plain
    dict because the vision tower is not implemented (text-only path) and its keys
    (``hidden_size``, ``num_hidden_layers``, ...) would collide with the text ones.
    """

    model_type = "deepseek_v41"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        quantization_config: dict[str, Any] | None = None,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        text = dict(text_config or {})

        max_position_embeddings = text.pop(
            "max_position_embeddings", max_position_embeddings
        )
        rope_scaling = text.pop("rope_scaling", rope_scaling)
        rope_theta = text.pop("rope_theta", rope_theta)

        self.vision_config = vision_config
        if quantization_config is not None:
            self.quantization_config = quantization_config
            self.expert_dtype = quantization_config.get("expert_dtype", "fp4")

        for key, value in text.items():
            setattr(self, key, value)
        for key, value in _V41_DEFAULTS.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters

        self._add_official_aliases(rope_scaling)

        super().__init__(**kwargs)

    def _add_official_aliases(self, rope_scaling: dict[str, Any] | None) -> None:
        """Add ``ModelArgs`` names when the HF spelling is the only one present."""
        for official, hf_name in _OFFICIAL_ALIASES.items():
            if not hasattr(self, official) and hasattr(self, hf_name):
                setattr(self, official, getattr(self, hf_name))

        rope = rope_scaling or {}
        for official, rope_key in _ROPE_ALIASES.items():
            if not hasattr(self, official) and rope_key in rope:
                setattr(self, official, rope[rope_key])


def register_config() -> None:
    """Make ``deepseek_v41`` resolvable by ``AutoConfig`` / vLLM's ``get_config``."""
    from transformers import AutoConfig

    AutoConfig.register("deepseek_v41", DeepseekV41Config, exist_ok=True)


register_config()
