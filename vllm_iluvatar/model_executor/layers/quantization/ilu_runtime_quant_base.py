# SPDX-License-Identifier: Apache-2.0
"""Runtime-quant entry point for checkpoints that carry no quantization config.

Target   : new Iluvatar-only quantization config, registered as ``ilu_runtime``
Baseline : vllm 0.27.1
Why      : ``additional_config.ilu_quant`` is resolved from a *base* quant
           config's ``get_quant_method`` (``ilu_fp8.py:1088``). A native
           BF16/FP16 checkpoint has no ``quantization_config``, so
           ``ModelConfig.quantization`` stays ``None``, vLLM never builds a base
           config, ``get_quant_method`` is never called, and the entire
           ilu_quant payload is ignored without a single log line. Guide 8
           requires fail-fast, so those models need a base config of their own.
How      : ``@register_quantization_config("ilu_runtime")``. The checkpoint
           declares nothing, so every ``LinearBase`` starts out ``ignored`` with
           an ``UnquantizedLinearMethod``; the only thing that can overlay a
           runtime method is a matching ilu_quant group. Non-Linear layers
           return ``None`` and keep community behaviour.
Scope    : native checkpoints quantized at load time, e.g. ``Qwen/Qwen3.8-27B``.
Remove   : if vLLM grows a first-class "no checkpoint quantization, but still
           give me a QuantizationConfig" hook.

Why a new name instead of reusing ``fp8``
-----------------------------------------
Reusing ``fp8`` would require injecting a fake ``quantization_config`` into the
model config, i.e. claiming the checkpoint is FP8 when it is BF16. Guide 8 §2
forbids rewriting the checkpoint's quantization identity, and the lie leaks into
``ModelConfig.quantization`` / ``is_quantized``. ``ilu_runtime`` says exactly
what it is: no checkpoint quantization, conversion described by ilu_quant.
The config is selected explicitly (``--quantization ilu_runtime``) and is never
auto-detected, so no existing checkpoint can land here by accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_config import (
    get_ilu_runtime_quant_config,
    resolve_ilu_runtime_method,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

_QUANT_NAME = "ilu_runtime"


@register_quantization_config(_QUANT_NAME)
class IluRuntimeQuantBaseConfig(QuantizationConfig):
    """Hosts ``additional_config.ilu_quant`` for unquantized checkpoints.

    Holds no checkpoint-derived state: everything that decides which layers are
    converted, and how, lives in the ilu_quant payload.
    """

    def __init__(self) -> None:
        super().__init__()
        # Consumed by method factories through ``copy.copy(base_quant_config)``;
        # ``IluFp8PerChannelOnlineLinearMethod.from_ilu_runtime_config`` assigns
        # both and then reads ``gemm_format`` in ``__init__`` / ``apply``.
        self.gemm_format = "TN"
        self.weight_block_size: list[int] | None = None
        self._payload_checked = False

    def get_name(self) -> QuantizationMethods:
        return _QUANT_NAME  # type: ignore[return-value]

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The payload's method decides the real requirement; this config only
        # dispatches.
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Empty on purpose: ``get_quant_config`` then builds us with
        # ``quant_cls()`` instead of looking for a json in the model directory
        # (``weight_utils.py:360-364``). The checkpoint stays untouched.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluRuntimeQuantBaseConfig:
        if config:
            raise ValueError(
                f"{_QUANT_NAME} takes no checkpoint quantization config, got "
                f"keys {sorted(config)}. Describe the conversion in "
                "additional_config.ilu_quant instead."
            )
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            # Embeddings, attention and everything else keep the behaviour they
            # would have had without any quantization config.
            return None

        self._require_payload()
        original_method = UnquantizedLinearMethod()
        runtime_method = resolve_ilu_runtime_method(
            base_quant_config=self,
            original_method=original_method,
            original_disposition="ignored",
            layer=layer,
            prefix=prefix,
        )
        return runtime_method if runtime_method is not None else original_method

    def _require_payload(self) -> None:
        """Refuse to run as a silent no-op (checked once, then cached)."""
        if self._payload_checked:
            return
        config = get_ilu_runtime_quant_config(self)
        if config is None:
            raise ValueError(
                f"quantization={_QUANT_NAME!r} requires "
                "additional_config.ilu_quant: the checkpoint declares no "
                "quantization, so without a payload every layer would stay in "
                "its native dtype. Pass the runtime quant YAML with --config, "
                "or drop --quantization to serve the checkpoint as-is."
            )
        if not config.enabled:
            raise ValueError(
                f"quantization={_QUANT_NAME!r} was selected but "
                "ilu_quant.enabled is false. Remove --quantization to serve "
                "the checkpoint in its native dtype."
            )
        self._payload_checked = True
        logger.info(
            "ILU runtime quant base config active: source=%s schema=%d "
            "model_type=%s groups=%s",
            config.source,
            config.schema_version,
            config.model_type,
            [group.group_id for group in config.config_groups],
        )
