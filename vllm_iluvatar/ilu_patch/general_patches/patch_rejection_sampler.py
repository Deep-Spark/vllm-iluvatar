# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar rejection-sampler overrides.

Corresponds to: vllm/v1/sample/rejection_sampler.py

Target   : vllm.v1.sample.rejection_sampler.rejection_sample
           vllm.v1.sample.rejection_sampler.generate_uniform_probs
           vllm.v1.sample.rejection_sampler.rejection_greedy_sample_kernel
           vllm.v1.sample.rejection_sampler.rejection_random_sample_kernel
Baseline : main_6e448d0
Why      : preserve Iluvatar/CoreX speculative-decoding fixes without editing
           community vLLM source.
How      : rebind module-level functions after signature and source-signature
           checks.
Scope    : speculative-decoding rejection sampling on Iluvatar.
Remove   : when upstream exposes sampler hooks for these backend-specific
           behaviours or Iluvatar Triton/float64 gaps are resolved.
Fingerprint: community sources are SHA-checked at register-time; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = [
    "rejection_sample",
    "generate_uniform_probs",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
]

_BASELINE_SOURCE_SHAS = {
    "rejection_sample": (
        "35d7236c81a06800520691e8eb79014b33653e05ce5e982901fdb68ea003a1ea"
    ),
    "generate_uniform_probs": (
        "f3e8fbcf4f6e8332bbe61141fad916087b61256f4dada65663d274b48e27524b"
    ),
    "rejection_greedy_sample_kernel": (
        "9cd5a8f8fe5a065937797b59a1fede6aadb711391d3288a09fc7ffa65cfc5551"
    ),
    "rejection_random_sample_kernel": (
        "63a939862a8810443cdd667248e16b8aba6c33e0f921812b5b6a34cc597d2bad"
    ),
}

def register() -> None:
    """Register Iluvatar rejection-sampler overrides."""
    p.register_once(
        __name__,
        _apply,
        detail="rejection sampler CoreX compatibility overrides",
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
            f"vs baseline main_6e448d0. Expected {expected_sha}, got "
            f"{current_sha}. Re-sync vllm_iluvatar/v1/sample/"
            "rejection_sampler.py and patch_rejection_sampler.py."
        )


def _apply() -> None:
    import vllm.v1.sample.rejection_sampler as _m

    import vllm_iluvatar.v1.sample.rejection_sampler as impl

    for name in _OVERRIDES:
        _check_source_sha(name, getattr(_m, name))

    impl._community_rejection_sample = _m.rejection_sample
    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        # [ILU override] rejection sampler - delta: CoreX uniform dtype,
        # deterministic-draft fallback, and explicit Triton predicates.
        p.rebind_module_func(_m, name, getattr(impl, name))
