# SPDX-License-Identifier: Apache-2.0
"""Architecture split of the raw-FP8 DeepSeek-V4-Flash attention stack.

ivcore11 has no fp8 sparse-MLA kernels -- ``ds4_sparse_mla_fp8_decode`` and
friends register their fast backend for ivcore30 only, and fall back to
ixformer's ``torch`` reference there -- so the layer builds the bf16-KV
attention instead, which keeps the compressor one level deeper.
"""

from __future__ import annotations

import pytest

from vllm_iluvatar.models.deepseek_v4 import model as v4_model

_CKPT_COMPRESSOR = "layers.3.attn.compressor.wkv.weight"
_CKPT_INDEXER_COMPRESSOR = "layers.3.attn.indexer.compressor.wkv.weight"


@pytest.mark.parametrize(
    ("ivcore11", "expected"),
    [
        (True, "model.layers.3.attn.mla_attn.compressor.wkv.weight"),
        (False, "model.layers.3.attn.compressor.wkv.weight"),
    ],
)
def test_main_compressor_follows_the_attention_stack(monkeypatch, ivcore11, expected):
    monkeypatch.setattr(v4_model, "is_ivcore11", lambda: ivcore11)
    mapper = v4_model._make_v4_fp8_weights_mapper()

    assert mapper.map_name(_CKPT_COMPRESSOR) == expected


@pytest.mark.parametrize("ivcore11", [True, False])
def test_indexer_compressor_never_moves(monkeypatch, ivcore11):
    # The indexer owns its compressor in both stacks, so the ``.attn.compressor.``
    # key must stay anchored tightly enough to miss this one.
    monkeypatch.setattr(v4_model, "is_ivcore11", lambda: ivcore11)
    mapper = v4_model._make_v4_fp8_weights_mapper()

    assert (
        mapper.map_name(_CKPT_INDEXER_COMPRESSOR)
        == "model.layers.3.attn.indexer.compressor.wkv.weight"
    )


def test_scale_rewrite_survives_the_added_rule(monkeypatch):
    # The compressor rule is added to ``orig_to_new_substr``; the regex mapping
    # that turns ``.scale`` into ``weight_scale_inv`` is separate and must still
    # fire.
    monkeypatch.setattr(v4_model, "is_ivcore11", lambda: True)
    mapper = v4_model._make_v4_fp8_weights_mapper()

    assert (
        mapper.map_name("layers.3.attn.compressor.wkv.scale")
        == "model.layers.3.attn.mla_attn.compressor.wkv.weight_scale_inv"
    )
