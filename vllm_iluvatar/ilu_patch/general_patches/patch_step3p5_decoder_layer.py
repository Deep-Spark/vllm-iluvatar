# SPDX-License-Identifier: Apache-2.0
"""Wiring for MTP-layer attention config on ``Step3p5DecoderLayer``.

Corresponds to: vllm/model_executor/models/step3p5.py

Target   : vllm.model_executor.models.step3p5.Step3p5DecoderLayer.__init__
Baseline : v0.31.0 (341bf7d4d1)
Why      : MTP draft ``mtp_block`` layers use indices ``>= num_hidden_layers``;
           community code no longer applies ``attention_other_setting`` there.
How      : wrap ``__init__`` to temporarily extend ``layer_types`` for MTP
           indices, then restore the original lists.
Scope    : Step-3.5 / Step-3.7 MTP draft quality on Iluvatar.
Remove   : when upstream extends ``layer_types`` for MTP layer indices.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Step3p5DecoderLayer MTP layer_types fallback",
    )


def _apply() -> None:
    from vllm.model_executor.models import step3p5 as _m

    from vllm_iluvatar.model_executor.models.step3p5_mtp_layer_config import (
        make_decoder_layer_init,
    )

    orig = _m.Step3p5DecoderLayer.__init__
    # [ILU override] Step3p5DecoderLayer.__init__ — delta: extend layer_types
    # for MTP indices, then restore the community lists.
    _m.Step3p5DecoderLayer.__init__ = make_decoder_layer_init(orig)