# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Step3.7 VL model override.

Corresponds to:
    vllm/model_executor/models/step3p7.py

Target:
    vllm.model_executor.models.step3p7.Step3p7ForConditionalGeneration
    (arch "Step3p7ForConditionalGeneration").
Baseline:
    main_6e448d0.
Why:
    Step-3.7 nests ``Step3p7TextConfig`` (model_type step3p5) without
    ``tie_word_embeddings``. Community ``Step3p5Model`` reads that field when
    building PP embeddings. The language tower is constructed with
    ``hf_config=text_config``, so the nested object must carry the default.
How:
    ``ModelRegistry.register_model`` points the community arch at this subclass.
    Before community construction, fill ``tie_word_embeddings=False`` on the
    nested text config when the attribute is missing. The language model still
    resolves to the existing Step3.5 OOT class. vLLM has no per-method model
    hook, so whole-class re-registration is the only official extension point.
Scope:
    Step-3.7 Flash serving on Iluvatar.
Remove:
    When upstream ``Step3p7TextConfig`` defines ``tie_word_embeddings``.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.models.step3p7 import (
    Step3p7ForConditionalGeneration as _Step3p7ForConditionalGeneration,
)


def ensure_step3p7_text_config(vllm_config: VllmConfig) -> None:
    """Fill missing nested ``tie_word_embeddings`` for Step-3.7 text configs."""
    config = vllm_config.model_config.hf_config
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    if not hasattr(text_config, "tie_word_embeddings"):
        text_config.tie_word_embeddings = False


class Step3p7ForConditionalGeneration(_Step3p7ForConditionalGeneration):
    """Step-3.7 VL model with a Step3.5-compatible nested text config."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        ensure_step3p7_text_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)


def register_model() -> None:
    """Point Step-3.7 architecture resolution at the plugin subclass."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Step3p7ForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.step3p7:"
        "Step3p7ForConditionalGeneration",
    )
