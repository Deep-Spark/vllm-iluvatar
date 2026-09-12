# SPDX-License-Identifier: Apache-2.0
"""Iluvatar Qwen3.8-Flash-Next (``qwen4_exp``) model package.

``vllm_iluvatar.ilu_register`` imports this package's top-level modules at
plugin init (the package is listed in ``_OOT_MODEL_PACKAGES``): ``config.py``
registers the HF config classes on import, and ``model.py`` exposes the
``register_model()`` hook. Deliberately no imports here, so neither the model
nor torch is pulled in until a checkpoint asks for it.

Layout:

* ``model.py`` -- model graph and the single official registry hook.
* ``attention.py`` / ``indexer.py`` / ``qsa_cache.py`` -- QSA ownership.
* ``ple.py`` / ``model_state.py`` -- PLE and request-state integration.
* ``hyperconnection.py`` / ``hc.py`` -- HC modules and ixformer CUDA adapter.

This model is **V2-model-runner only** (``VLLM_USE_V2_MODEL_RUNNER=1``): the PLE
conv carry, the PLE n-gram context and the QSA group accumulator are per-request
state with no KV-cache home, and ``Qwen4ExpModelState`` is what owns them.
"""
