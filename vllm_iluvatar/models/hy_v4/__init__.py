# SPDX-License-Identifier: Apache-2.0
"""Iluvatar HY V4 (Hy4-preview) model package.

Layout mirrors ``models/deepseek_v4``:

    config.py          HYV4Config + _CONFIG_REGISTRY / AutoConfig registration
    model.py           model graph, weight loading, register_model() hook
    attention.py       MLA + lightning indexer + learnable sink + gated output
    flashmla_sparse.py sink-capable Iluvatar sparse-MLA backend
    moe.py             dense FFN and MoE blocks
    hc.py              iHC layers (ixformer fused, eager fallback)
    weight_utils.py    MXFP8 indexer-wk dequant

Modules are imported by ``ilu_register`` during plugin init; the model class
itself is resolved lazily by ``ModelRegistry`` from the path string in
``model.register_model()``.
"""
