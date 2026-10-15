# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar rejection-sampler override.

Corresponds to: vllm/v1/sample/rejection_sampler.py

Target   : vllm.v1.sample.rejection_sampler.generate_uniform_probs
Baseline : v0.31.0 (db9527a468)
Why      : float64 CUDA uniforms read back as zeros on CoreX and force every
           draft token to be accepted. rejection_sample and the greedy, random,
           and recovered-token kernels stay on upstream.
How      : rebind generate_uniform_probs after a signature check. Also
           fingerprint rejection_sample, because the float32 override only
           applies while upstream calls generate_uniform_probs by name.
Scope    : speculative-decoding rejection sampling on Iluvatar.
Remove   : when float64 CUDA reads are reliable on CoreX, or upstream exposes
           a dtype hook for these uniforms.
Fingerprint: community sources are SHA-checked at register-time; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = [
    "generate_uniform_probs",
]

_BASELINE_SOURCE_SHAS = {
    "rejection_sample": (
        "adadcc804bccf89cc3568b637d14b90480f903d0a5de417719200ad6a284c747"
    ),
    "generate_uniform_probs": (
        "016fdca2217c3f2b2ce40b75d94cac8c49fbea92202180cb57a3f481654fd83f"
    ),
}


def register() -> None:
    """Register the Iluvatar rejection-sampler override."""
    p.register_once(
        __name__,
        _apply,
        detail="float32 rejection-sampler uniforms",
    )


def _source_for_fingerprint(symbol: object) -> str:
    """Return Python source for a function or a Triton JITFunction wrapper."""
    return inspect.getsource(getattr(symbol, "fn", symbol))


def _check_source_sha(symbol_name: str, symbol: object) -> None:
    if hasattr(p, "_enabled") and not p._enabled(p.ENV_CHECK_FINGERPRINT):
        return
    current_sha = p.source_sha(_source_for_fingerprint(symbol))
    expected_sha = _BASELINE_SOURCE_SHAS[symbol_name]
    if current_sha != expected_sha:
        raise RuntimeError(
            "ILU patch: community "
            f"vllm.v1.sample.rejection_sampler.{symbol_name} source changed "
            f"vs baseline v0.31.0. Expected {expected_sha}, got "
            f"{current_sha}. Re-sync vllm_iluvatar/v1/sample/"
            "rejection_sampler.py and patch_rejection_sampler.py."
        )


def _apply() -> None:
    import vllm.v1.sample.rejection_sampler as _m

    import vllm_iluvatar.v1.sample.rejection_sampler as impl

    for name in _BASELINE_SOURCE_SHAS:
        _check_source_sha(name, getattr(_m, name))

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        # [ILU override] rejection sampler - delta: float32 uniforms.
        # rejection_sample and the sampling kernels stay on upstream.
        p.rebind_module_func(_m, name, getattr(impl, name))
