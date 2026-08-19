# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT registration for the community DeepSeek MTP draft model.

Corresponds to: vllm/model_executor/models/deepseek_mtp.py ::
``DeepSeekMTP`` PP support marker.

Target   : ``vllm.model_executor.models.deepseek_mtp.DeepSeekMTP``
Baseline : main_6e448d0
Why      : Iluvatar PP+MTP runs need the DeepSeek MTP draft architecture to be
           recognised as pipeline-parallel capable, while the community source
           stays upstream-like.
How      : register the public ``DeepSeekMTPModel`` architecture name to a
           tiny plugin subclass that adds ``SupportsPP`` and otherwise inherits
           the community implementation unchanged.
Scope    : DeepSeek-family MTP draft models on Iluvatar.
Remove   : when upstream marks ``DeepSeekMTP`` as PP-capable.
"""

from __future__ import annotations

from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP as _DeepSeekMTP
from vllm.model_executor.models.interfaces import SupportsPP


class IluDeepSeekMTP(_DeepSeekMTP, SupportsPP):
    """DeepSeek MTP draft model with Iluvatar PP capability marker."""


def register_model() -> None:
    """Register the community DeepSeek MTP draft arch to the Iluvatar wrapper."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_iluvatar.model_executor.models.deepseek_mtp:IluDeepSeekMTP",
    )


__all__ = ["IluDeepSeekMTP", "register_model"]
