# SPDX-License-Identifier: Apache-2.0
"""Wiring for top-k/top-p sampler ILU override.

Corresponds to: vllm/v1/sample/ops/topk_topp_sampler.py

Target   : vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p
Baseline : main_6e448d0
Why      : Iluvatar should avoid the upstream Triton top-k/top-p sampler path
           on GPU while keeping upstream CPU behavior intact.
How      : rebind the module-level function so already-imported references are
           updated as well.
Scope    : Iluvatar GPU sampling; CPU behavior is preserved.
Remove   : when upstream Triton sampler is stable on Iluvatar.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["apply_top_k_top_p"]


def register() -> None:
    """Register the top-k/top-p sampler override."""
    p.register_once(
        __name__,
        _apply,
        detail="top-k/top-p sampler PyTorch fallback on Iluvatar GPU",
    )


def _apply() -> None:
    import vllm.v1.sample.ops.topk_topp_sampler as _m

    import vllm_iluvatar.v1.sample.ops.topk_topp_sampler as impl

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        p.rebind_module_func(_m, name, getattr(impl, name))
