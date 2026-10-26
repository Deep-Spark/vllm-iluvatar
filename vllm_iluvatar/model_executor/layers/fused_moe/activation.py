"""Activation configuration shared by unquantized and quantized MoE methods."""

import math


def swiglustep_limit(value: float | None) -> float:
    """Resolve a missing limit; reject values unsupported by native SwiGLUStep.

    Zero is not an alias for missing or for unclamped SiLU. The native
    SwiGLUStep contract requires a finite, strictly positive clamp.
    """
    if value is None:
        return 7.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("SwiGLUStep limit must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("SwiGLUStep limit must be a finite positive number")
    return float(value)
