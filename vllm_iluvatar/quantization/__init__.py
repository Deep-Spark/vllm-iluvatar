"""Importing this package triggers the ``@register_quantization_config``
decorator inside :mod:`vllm_iluvatar.quantization.ilu_deepseek_v4_w4a8`,
which adds ``ilu_deepseek_v4_w4a8`` to vLLM's
``QUANTIZATION_METHODS`` (and ``current_platform.supported_quantization``
if available)."""

from __future__ import annotations

from .ilu_deepseek_v4_w4a8 import IluDeepseekV4W4A8Config

__all__ = ["IluDeepseekV4W4A8Config"]
