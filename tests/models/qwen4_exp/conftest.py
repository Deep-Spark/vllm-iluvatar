# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply Iluvatar Triton jit compatibility before Qwen4Exp imports GDN."""

from __future__ import annotations

try:
    from vllm_iluvatar.ilu_patch.general_patches import patch_triton_jit_compat

    patch_triton_jit_compat._apply()
except Exception:
    pass
