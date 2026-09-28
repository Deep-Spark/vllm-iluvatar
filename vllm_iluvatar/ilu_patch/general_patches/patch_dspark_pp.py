# SPDX-License-Identifier: Apache-2.0
"""Let DeepSeek-V4.1 DSpark draft under pipeline parallelism.

Corresponds to: vllm/v1/worker/gpu/spec_decode/dspark/utils.py
                vllm/models/deepseek_v41/nvidia/dspark.py
Target   : maybe_share_target_embed, DSparkDeepseekV4ForCausalLM.load_weights
Baseline : v0.30.0
Why      : 0.30.1 already places the drafter on the last PP rank and refuses
           to alias a ``PPMissingLayer`` embedding, which raises. The draft
           checkpoint still carries the target ``embed.weight``, but the
           community loader skips every non-``mtp.`` tensor.
How      : on PP>1, load ``embed.weight`` into the draft and skip the alias.
           ``lm_head`` sharing on the last rank is left to community.
Scope    : DeepSeek-V4.1 DSpark. PP=1 still aliases the target embedding.
Remove   : when community ``load_dspark_model`` loads a draft-owned embedding
           under pipeline parallelism.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DSpark draft embedding under pipeline parallelism",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark_pp import (
        maybe_share_target_embed_pp,
    )

    # [ILU override] maybe_share_target_embed — delta: no alias when this
    # PP rank has no target embedding. ``load_dspark_model`` imports the
    # name inside the function, so the module attribute is what it calls.
    # The Iluvatar V4.1 draft loads ``embed.weight`` itself under PP>1.
    eagle_utils.maybe_share_target_embed = maybe_share_target_embed_pp(
        eagle_utils.maybe_share_target_embed
    )
