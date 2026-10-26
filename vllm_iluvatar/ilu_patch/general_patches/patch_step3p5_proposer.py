# SPDX-License-Identifier: Apache-2.0
"""Wiring for Step-3 MTP-3 draft proposer semantics.

Corresponds to: vllm/v1/spec_decode/step3p5.py
                vllm/v1/spec_decode/eagle.py (via Step3p5MTPProposer MRO)

Target   : vllm.v1.spec_decode.step3p5.Step3p5MTPProposer.propose
           vllm.v1.spec_decode.step3p5.Step3p5MTPProposer._create_draft_vllm_config
           vllm.v1.spec_decode.step3p5.Step3p5MTPProposer._sample_draft_tokens_for_step
           vllm.v1.spec_decode.step3p5.Step3p5MTPProposer.build_per_group_and_layer_attn_metadata
Baseline : v0.31.0 (341bf7d4d1)
Why      : community sequential draft rotates MTP layers and forwards under the
           target vllm config; Step-3.5 / 3.7 MTP-3 needs layer 0 and the
           draft vllm config.
How      : wrap only ``Step3p5MTPProposer`` methods. Eagle / DeepSeek / DSpark
           proposers stay on the community path. ``draft_vllm_config`` is
           materialised lazily in propose with draft PP=1.
Scope    : Step3p5MTPProposer (Step-3.5 text tower and Step-3.7 nested text).
Remove   : when upstream restores those two draft semantics.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Step3p5MTPProposer layer-0 draft + draft_vllm_config",
    )


def _apply() -> None:
    from vllm.v1.spec_decode import step3p5 as _step3p5

    from vllm_iluvatar.v1.spec_decode.step3p5 import (
        make_build_per_group_and_layer_attn_metadata,
        make_create_draft_vllm_config,
        make_propose,
        make_sample_draft_tokens_for_step,
    )

    proposer = _step3p5.Step3p5MTPProposer
    # [ILU override] Step3p5MTPProposer — delta: layer-0 draft and draft
    # vllm config. Eagle / DeepSeek / DSpark proposers are not assigned.
    proposer._create_draft_vllm_config = make_create_draft_vllm_config(
        proposer._create_draft_vllm_config
    )
    proposer.propose = make_propose(proposer.propose)
    proposer._sample_draft_tokens_for_step = make_sample_draft_tokens_for_step(
        proposer._sample_draft_tokens_for_step
    )
    proposer.build_per_group_and_layer_attn_metadata = (
        make_build_per_group_and_layer_attn_metadata(
            proposer.build_per_group_and_layer_attn_metadata
        )
    )