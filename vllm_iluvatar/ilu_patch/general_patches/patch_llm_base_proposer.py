# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``SpecDecodeBaseProposer.model_returns_tuple`` override.

Corresponds to: vllm/v1/spec_decode/llm_base_proposer.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.v1.spec_decode.llm_base_proposer``; this module only performs
register-time wiring.

Target   : vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.
           model_returns_tuple
Baseline : main_6e448d0
Why      : the ilu GLM-5.2 MTP draft (IluGlmMoeDsaMTPModel) is a DeepSeek-V3.2
           standard MTP whose forward returns (logit_hidden, recycle_hidden);
           the community method only recognises DeepSeekMTPModel, so the tuple
           would otherwise be mis-unpacked for the ilu draft.
How      : wrap the instance method with ``make_model_returns_tuple`` and
           setattr it back onto the class (instances resolve through the class
           at call time).
Scope    : mtp-method speculative decoding on Iluvatar.
Remove   : when upstream recognises the ilu MTP draft arch (or exposes a hook).
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar model_returns_tuple wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="SpecDecodeBaseProposer.model_returns_tuple ilu MTP tuple contract",
    )


def _apply() -> None:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    from vllm_iluvatar.v1.spec_decode.llm_base_proposer import (
        make_model_returns_tuple,
    )

    orig = SpecDecodeBaseProposer.model_returns_tuple
    # [ILU override] model_returns_tuple - delta: treat IluGlmMoeDsaMTPModel as
    # a DeepSeek-family tuple-returning MTP draft; delegate otherwise.
    SpecDecodeBaseProposer.model_returns_tuple = make_model_returns_tuple(orig)
