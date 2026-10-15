# SPDX-License-Identifier: Apache-2.0
"""Wiring for the GLM sampling override.

Corresponds to: vllm/v1/sample/ops/topk_topp_sampler.py

Target   : TopKTopPSampler.forward_native
Baseline : v0.31.0 (db9527a468)
Why      : VLLM_ILU_GLM_SAMPLING_MODE selects native CQ sorted/radix sampling.
           apply_top_k_top_p is left on upstream.
How      : rebind TopKTopPSampler.forward_native only. The fallback calls the
           original method, which uses upstream apply_top_k_top_p.
Scope    : ivcore30 GLM sampling when the contract matches.
Remove   : when upstream provides this GLM sampling path.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the GLM sampling override."""
    p.register_once(
        __name__,
        _apply,
        detail="GLM CQ sampling on TopKTopPSampler.forward_native",
    )


def _apply() -> None:
    import vllm.v1.sample.ops.topk_topp_sampler as _m

    import vllm_iluvatar.v1.sample.ops.topk_topp_sampler as impl

    p.check_signatures(_m.TopKTopPSampler, impl, ["forward_native"], strict=True)
    _m.TopKTopPSampler.forward_native = impl.forward_native
