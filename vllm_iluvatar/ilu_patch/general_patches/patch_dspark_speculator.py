# SPDX-License-Identifier: Apache-2.0
"""Wire Iluvatar vocab-sharded DSpark sampling and draft enforce_eager guards.

Corresponds to: vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
                vllm/v1/worker/gpu/spec_decode/dspark/speculator.py

Target   : DFlashSpeculator.init_cudagraph_manager / capture
           DSparkSpeculator._sample_sequential / load_draft_model / propose
Baseline : v0.31.0 (db9527a468)
Why      : Community DFlashSpeculator ignores speculative_config.enforce_eager
           when selecting CUDA graph mode and capturing graphs; also avoid
           full-vocabulary gathers and repeated target-context projection.
How      : wrap init_cudagraph_manager and capture to honor enforce_eager, and
           delegate DSpark methods unless the model advertises local-logits or
           direct-context capability.
Scope    : DFlash / DSpark speculative decoding on Iluvatar.
Remove   : when upstream honors speculative_config.enforce_eager in DFlash and
           supports these model capabilities.
Fingerprint: community target methods are SHA-checked at register-time; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

_COMMUNITY_INIT_CG = (
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator."
    "DFlashSpeculator.init_cudagraph_manager"
)
_COMMUNITY_CAPTURE = (
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator.capture"
)

_COMMUNITY_SOURCE_SHAS = {
    _COMMUNITY_INIT_CG: (
        "a49ae79913d3471f2c6d0bfd8ac1199e9f92513d9ebb2db3989b8970a69a238a"
    ),
    _COMMUNITY_CAPTURE: (
        "74a18dc20b50e5306ccaa920f2d92c4d4bf941a53ed064fb781409378f5a5265"
    ),
}


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSpark vocab-sharded draft logits",
    )


def _check_community_source(symbol_name: str, fn: object) -> None:
    if not p._enabled(p.ENV_CHECK_FINGERPRINT):
        return
    current_sha = p.source_sha(inspect.getsource(inspect.unwrap(fn)))
    expected_sha = _COMMUNITY_SOURCE_SHAS[symbol_name]
    if current_sha != expected_sha:
        raise RuntimeError(
            f"ILU patch: community {symbol_name} source changed vs "
            f"baseline v0.31.0. Expected {expected_sha}, got {current_sha}. "
            "Re-sync vllm_iluvatar/v1/worker/gpu/spec_decode/dspark/speculator.py, "
            "then refresh the baseline SHA in patch_dspark_speculator.py."
        )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import speculator as impl

    orig_init_cg = inspect.unwrap(DFlashSpeculator.init_cudagraph_manager)
    orig_capture = inspect.unwrap(DFlashSpeculator.capture)
    _check_community_source(_COMMUNITY_INIT_CG, orig_init_cg)
    _check_community_source(_COMMUNITY_CAPTURE, orig_capture)

    DFlashSpeculator.init_cudagraph_manager = impl.make_init_cudagraph_manager(
        orig_init_cg
    )
    DFlashSpeculator.capture = impl.make_capture(orig_capture)

    # Keep the community body for top-k / models without local-head APIs.
    DSparkSpeculator._ilu_community_sample_sequential = (
        DSparkSpeculator._sample_sequential
    )
    # [ILU override] _sample_sequential — delta: optional shard-local logits.
    DSparkSpeculator._sample_sequential = impl.sample_sequential
    DSparkSpeculator._ilu_community_load_draft_model = DSparkSpeculator.load_draft_model
    # [ILU override] load_draft_model — delta: bind optional context destination.
    DSparkSpeculator.load_draft_model = impl.load_draft_model
    DSparkSpeculator._ilu_community_propose = DSparkSpeculator.propose
    # [ILU override] propose — delta: reuse the target-projected context buffer.
    DSparkSpeculator.propose = impl.propose
    if hasattr(DSparkSpeculator, "_generate_draft"):
        DSparkSpeculator._generate_draft = impl.make_generate_draft(
            DSparkSpeculator._generate_draft
        )
    if hasattr(DSparkSpeculator, "_build_uniform_attn_metadata"):
        DSparkSpeculator._build_uniform_attn_metadata = (
            impl.make_build_uniform_attn_metadata(
                DSparkSpeculator._build_uniform_attn_metadata
            )
        )
