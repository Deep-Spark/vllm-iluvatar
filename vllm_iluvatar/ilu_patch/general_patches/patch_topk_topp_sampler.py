# SPDX-License-Identifier: Apache-2.0
"""Wiring for the GLM sampling override.

Corresponds to: vllm/v1/sample/ops/topk_topp_sampler.py

Target   : TopKTopPSampler.forward_native, apply_top_k_top_p
Baseline : v0.31.0 (db9527a468)
Why      : VLLM_ILU_GLM_SAMPLING_MODE selects native CQ sorted/radix sampling;
           supported MR/CQ distributions use the weighted-radix mask.
How      : retain the CQ method override and rebind apply_top_k_top_p,
           including imported V2 references. Disabled/unsupported calls use
           their original mainline implementation.
Scope    : Iluvatar sampling when the selected contract matches.
Remove   : when upstream provides this GLM sampling path.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the GLM sampling override."""
    p.register_once(
        __name__,
        _apply,
        detail="GLM CQ sampler and supported MR/CQ top-p mask",
    )


def _apply() -> None:
    import vllm.v1.sample.ops.topk_topp_sampler as _m

    import vllm_iluvatar.v1.sample.ops.topk_topp_sampler as impl

    p.check_signatures(_m.TopKTopPSampler, impl, ["forward_native"], strict=True)
    _m.TopKTopPSampler.forward_native = impl.forward_native
    p.check_signatures(_m, impl, ["apply_top_k_top_p"], strict=True)
    p.rebind_module_func(_m, "apply_top_k_top_p", impl.apply_top_k_top_p)
