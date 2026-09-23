# SPDX-License-Identifier: Apache-2.0
"""Teach vLLM that ``model_type = "hy_v4"`` is an MLA architecture.

Corresponds to: vllm/transformers_utils/model_arch_config_convertor.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : ``ModelArchConfigConvertorBase.is_deepseek_mla``
Baseline : the method answers from a hardcoded ``model_type`` tuple
           ("deepseek_v2", "deepseek_v32", "glm_moe_dsa", ...). ``hy_v4`` is
           absent, and every other entry point derives from this one answer.
Why      : ``is_deepseek_mla`` is what makes ``ModelConfig`` report the MLA KV
           cache geometry -- one ``kv_lora_rank + qk_rope_head_dim`` latent row
           per token instead of ``num_kv_heads * head_dim`` K plus V. Answering
           False for HY V4 sizes the cache as if it were MHA, so ``get_head_size``
           and the KV cache spec disagree with what ``MLAAttention`` actually
           writes. There is no config field that can flip the answer: the check
           is purely the model_type allow-list, so this cannot be solved from
           the model package or the HF config.
How      : wrap the bound method after the module is imported. The wrapper
           returns True for ``hy_v4`` and delegates verbatim otherwise, so no
           other architecture changes behaviour and no method body is copied.
Scope    : HY V4 (Hy4-preview) only.
Remove   : when the Iluvatar vLLM baseline lists ``hy_v4`` itself (upstream
           #54160 adds it).

Timing   : ``ModelConfig`` is built in ``EngineArgs.create_engine_config``, long
           after the platform plugin runs ``register_general`` ->
           ``apply_general_patches()`` at ``import vllm`` time, so the wrapper is
           always in place before the first call.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_HY_V4_MODEL_TYPE = "hy_v4"


def register() -> None:
    """Register the HY V4 MLA-config wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail=f"is_deepseek_mla -> True for model_type={_HY_V4_MODEL_TYPE}",
    )


def _apply() -> None:
    import importlib

    module = importlib.import_module(
        "vllm.transformers_utils.model_arch_config_convertor"
    )
    convertor_cls = module.ModelArchConfigConvertorBase

    orig = convertor_cls.is_deepseek_mla
    if getattr(orig, "_ilu_hy_v4_wrapped", False):
        return

    # [ILU override] ModelArchConfigConvertorBase.is_deepseek_mla — delta: add
    # ``hy_v4`` to the set of MLA model types. HY V4 is MLA with DSA on top
    # (kv_lora_rank + qk_rope_head_dim latent KV), so the DeepSeek geometry is
    # the correct one; the allow-list simply predates the architecture.
    def is_deepseek_mla(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        if model_type == _HY_V4_MODEL_TYPE:
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        return orig(self)

    is_deepseek_mla._ilu_hy_v4_wrapped = True
    convertor_cls.is_deepseek_mla = is_deepseek_mla
