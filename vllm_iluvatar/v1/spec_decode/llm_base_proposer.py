# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``SpecDecodeBaseProposer.model_returns_tuple`` override.

Corresponds to: vllm/v1/spec_decode/llm_base_proposer.py ::
``SpecDecodeBaseProposer.model_returns_tuple``

The community method returns ``True`` (draft forward yields the
``(logit_hidden, recycle_hidden)`` tuple) only for ``DeepSeekMTPModel``. The
ilu GLM-5.2 MTP draft ``IluGlmMoeDsaMTPModel`` and the plugin DeepSeek V3.2
draft ``DeepseekV32MTPModel`` follow the same tuple contract, so this wrapper
returns ``True`` for them and otherwise delegates to the community implementation
unchanged.

Baseline : main_6e448d0
"""

from __future__ import annotations

import functools
from collections.abc import Callable

# Ilu MTP draft archs that follow the DeepSeek-family (logit, recycle) tuple
# forward contract.
_ILU_TUPLE_MTP_ARCHS = (
    "IluGlmMoeDsaMTPModel",
    # Subclasses IluGlmMoeDsaMTP, so same forward contract.
    "IluGlmMoeDsaFp8MTPModel",
    "DeepseekV32MTPModel",
)


def make_model_returns_tuple(orig: Callable[..., bool]) -> Callable[..., bool]:
    """Wrap the community ``model_returns_tuple`` method.

    Returns ``True`` for ilu MTP draft archs that share the DeepSeek tuple
    contract; otherwise falls back to the community implementation.
    """

    @functools.wraps(orig)
    def model_returns_tuple(self) -> bool:
        if self.method == "mtp":
            architectures = self.draft_model_config.hf_config.architectures or []
            if any(arch in architectures for arch in _ILU_TUPLE_MTP_ARCHS):
                return True
        return orig(self)

    return model_returns_tuple
