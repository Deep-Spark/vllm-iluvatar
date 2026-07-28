# SPDX-License-Identifier: Apache-2.0
"""Import hook for the MMEncoderAttention OOT override.

The OOT registry auto-discovers top-level modules in
``vllm_iluvatar.model_executor.layers``. Keep the implementation mirrored under
``layers.attention`` and import it here for registration side effects.
"""

from vllm_iluvatar.model_executor.layers.attention import (
    mm_encoder_attention as _mm_encoder_attention,
)

__all__ = ["_mm_encoder_attention"]
