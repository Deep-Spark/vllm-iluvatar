# SPDX-License-Identifier: Apache-2.0
"""Wiring for the GLM sampling override.

Corresponds to: vllm/v1/sample/ops/topk_topp_sampler.py

Target   : TopKTopPSampler.forward_native and apply_top_k_top_p
Baseline : v0.31.0 (db9527a468)
Why      : VLLM_ILU_GLM_SAMPLING_MODE selects native CQ sorted/radix sampling.
           TOP_P_RADIX separately enables the native mask for MTP verification.
How      : rebind the method and the free function, including imported aliases
           in V1 rejection and V2 sampling states. Fallback stays upstream.
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
    p.check_signatures(_m, impl, ["apply_top_k_top_p"], strict=True)
    _m.TopKTopPSampler.forward_native = impl.forward_native
    p.rebind_module_func(_m, "apply_top_k_top_p", impl.apply_top_k_top_p)
