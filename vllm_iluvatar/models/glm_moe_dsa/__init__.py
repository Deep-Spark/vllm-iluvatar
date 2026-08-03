# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM-5.2 (GlmMoeDsa) W4A8 model package.

Modules here are auto-imported by ``vllm_iluvatar.ilu_register`` (the package is
listed in ``_OOT_MODEL_PACKAGES``). ``model.register_model()`` registers the
causal-LM arch (``IluGlmMoeDsaForCausalLM``) via ``ModelRegistry.register_model``.

Layout mirrors ``deepseek_v4``: the attention *algorithm* chain is fully
self-contained in ``attention.py`` (no community
``MultiHeadLatentAttentionWrapper`` / ``MLAAttention`` / ``SparseMLAAttentionImpl``
in the runtime path); ``model.py`` holds the model graph + W4A8 weight loading.
"""
