# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 test layout.

Quant-specific cases live in ``w4a8/`` and ``mxfp8/``.
Shared (not quant-bound) cases stay in this directory.

ivcore11 collects w4a8 (compressed-tensors) only; ivcore30 collects both.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_util.arch import is_ivcore11

# Do not collect MiniMax mxfp8 cases on ivcore11. w4a8 (compressed-tensors) runs
# on both ivcore11 and ivcore30.
collect_ignore = ["mxfp8"] if is_ivcore11() else []
