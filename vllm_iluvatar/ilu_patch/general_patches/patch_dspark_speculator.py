# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/spec_decode/dspark/speculator.py

Target   : DSparkSpeculator._sample_sequential / load_draft_model / propose
Baseline : v0.31.0 (db9527a468)
Why      : avoid full-vocabulary gathers and repeated target-context projection.
How      : delegate to community methods unless the model advertises the
           corresponding local-logits or direct-context capability.
Scope    : DSpark on Iluvatar; direct context requires sequential PP1/CP1,
           no microbatching and no LoRA.
Remove   : when upstream supports these model capabilities.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSpark vocab-sharded draft logits",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import speculator as impl

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
