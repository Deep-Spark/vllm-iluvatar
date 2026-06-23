# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT DeepSeek-V4 W4A8 model package.

Modules here are auto-imported by ``vllm_iluvatar.ilu_register`` (the package is
listed in ``_OOT_MODEL_PACKAGES``). ``model.register_model()`` registers both
the causal-LM arch (``IluDeepseekV4ForCausalLM``) and the MTP draft arch
(``IluDeepSeekV4MTPModel``) via ``ModelRegistry.register_model``; the tokenizer
mode is set by ``IluPlatform.check_and_update_config``. No community-source
edits to the vLLM model registry / deepseek_v4 package are required.
"""
