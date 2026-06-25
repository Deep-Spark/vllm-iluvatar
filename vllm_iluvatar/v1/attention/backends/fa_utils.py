# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/attention/backends/fa_utils.py

ILU override implementations for the flash-attn capability helpers
``get_flash_attn_version`` and ``flash_attn_supports_sinks``. This module holds
ONLY the override implementations; the wiring (register/_apply + signature guard
+ propagating rebind) lives in the sibling ``fa_utils_patch.py``, which is the
module auto-discovered and applied via ``register()`` during
``register_general``.

Target   : vllm.v1.attention.backends.fa_utils.{get_flash_attn_version,
           flash_attn_supports_sinks}
Baseline : main_e0871ad2259
Why      : IluPlatform is cuda-like (``is_cuda()`` is True), so community
           fa_utils takes the CUDA path whose helpers query upstream
           FlashAttention version dispatch (FA2/FA3/FA4). Iluvatar's ixformer
           flash_attn has no such versioning, so we DELIBERATELY DECOUPLE from
           the upstream logic: report "no FA version" and "sinks supported"
           unconditionally.
How      : full replacement (NOT a wrapper or full-method copy) — the override
           bodies do not reference the community implementation, so there is no
           source-drift fingerprint to track; only a prefix-compatible signature
           is validated (see fa_utils_patch.py / §5.5).
Scope    : all attention backends on Iluvatar that consult these helpers
           (flash_attn / MLA prefill / mm-encoder / turboquant / diffkv ...).
Remove   : when the plugin ships its own attention backend that no longer routes
           through the community fa_utils helpers (converge to a self-owned FA
           stack), or when upstream gates these via the platform abstraction.

Note on ``get_flash_attn_version`` return value: this returns ``None`` (matching
the iluvatar_main_opt_ds_attn_plugin source delta). The standalone fork plugin
instead returns ``2``; both disable upstream FA-version dispatch, but ``None``
keeps every ``== 3`` / ``== 4`` check false without claiming a concrete FA2
path. Keep these aligned if the two plugins are ever reconciled.
"""

from __future__ import annotations


# [ILU override] get_flash_attn_version — delta: ixformer flash_attn has no
# upstream FA version dispatch; always report "no version". IluPlatform is
# cuda-like, so the community xpu/rocm branches are unreachable here and are
# dropped (this is a deliberate full replacement, not upstream-shape tracking).
def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    return None


# [ILU override] flash_attn_supports_sinks — delta: ixformer flash_attn supports
# attention sinks without FA3/FA4 versioning, so report True unconditionally
# (community ties this to ``get_flash_attn_version() in (3, 4)``).
def flash_attn_supports_sinks() -> bool:
    return True
