# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM-5.2 (GlmMoeDsa) model package.

Modules here are auto-imported by ``vllm_iluvatar.ilu_register`` (the package is
listed in ``_OOT_MODEL_PACKAGES``) and each module's ``register_model()`` hook
is called. Two checkpoint flavours are served:

* ``model.py`` -- the converted W4A8 weights, arch ``IluGlmMoeDsaForCausalLM``.
* ``fp8_model.py`` -- the official ZhipuAI FP8 weights, registered under the
  community arch ``GlmMoeDsaForCausalLM`` to override vLLM's own
  implementation. It subclasses the W4A8 graph and only overrides what the FP8
  tensor set forces (absorbed up-proj, fused ``indexer.wk``).

Layout mirrors ``deepseek_v4``: the attention *algorithm* chain is fully
self-contained in ``attention.py`` (no community
``MultiHeadLatentAttentionWrapper`` / ``MLAAttention`` / ``SparseMLAAttentionImpl``
in the runtime path); ``model.py`` holds the model graph + weight loading.
"""
