# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for sharing the DSA top-k global-index conversion.

``_global_topk`` maps request-local indices onto global KV slots. Every layer
used to redo it even though only ``full`` layers rewrite the buffer it reads,
so the result is reused until something invalidates it. What matters here is
that the invalidation covers every write: a ``full`` layer's indexer, and a PP
stage copying a buffer in from upstream (whose indices belong to another
stage's KV cache and must be re-converted against this stage's block_table).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_iluvatar.models.glm_moe_dsa import attention as attn_mod
from vllm_iluvatar.models.glm_moe_dsa.attention import GlmDsaTopkConvertCache

NUM_TOKENS = 4
TOPK = 8


def _value(tag: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.full((NUM_TOKENS, 1, TOPK), tag), torch.tensor([tag]))


class TestCacheSemantics:
    def test_starts_empty(self):
        assert GlmDsaTopkConvertCache().get(NUM_TOKENS) is None

    def test_roundtrip(self):
        cache = GlmDsaTopkConvertCache()
        value = _value(1.0)
        cache.put(NUM_TOKENS, value)
        assert cache.get(NUM_TOKENS) is value

    def test_invalidate_drops_the_entry(self):
        cache = GlmDsaTopkConvertCache()
        cache.put(NUM_TOKENS, _value(1.0))
        cache.invalidate()
        assert cache.get(NUM_TOKENS) is None

    def test_other_token_count_misses(self):
        """A differently shaped capture must not pick up this one's tensors."""
        cache = GlmDsaTopkConvertCache()
        cache.put(NUM_TOKENS, _value(1.0))
        assert cache.get(NUM_TOKENS + 1) is None

    def test_reput_after_invalidate_is_visible(self):
        cache = GlmDsaTopkConvertCache()
        cache.put(NUM_TOKENS, _value(1.0))
        cache.invalidate()
        fresh = _value(2.0)
        cache.put(NUM_TOKENS, fresh)
        assert cache.get(NUM_TOKENS) is fresh


def _make_mla(cache):
    """Bare object exposing just what ``_global_topk`` touches."""
    mla = SimpleNamespace(
        topk_indices_buffer=torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32),
        topk_convert_cache=cache,
    )
    mla._global_topk = (
        attn_mod.IluGlmMoeDsaSparseMLA._global_topk.__get__(mla)  # noqa: SLF001
    )
    return mla


def _metadata():
    return SimpleNamespace(
        req_id_per_token=torch.zeros(NUM_TOKENS, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        block_size=64,
    )


class TestGlobalTopkReuse:
    @pytest.fixture
    def convert(self):
        with patch.object(
            attn_mod,
            "triton_convert_req_index_to_global_index",
            side_effect=lambda *a, **k: (
                torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32),
                torch.zeros(NUM_TOKENS, dtype=torch.int32),
            ),
        ) as m:
            yield m

    def test_converts_once_then_reuses(self, convert):
        mla = _make_mla(GlmDsaTopkConvertCache())
        md = _metadata()

        first = mla._global_topk(NUM_TOKENS, md)
        for _ in range(3):
            assert mla._global_topk(NUM_TOKENS, md) is first
        assert convert.call_count == 1

    def test_reconverts_after_invalidate(self, convert):
        """What a ``full`` layer's indexer, or a PP recv, has to trigger."""
        cache = GlmDsaTopkConvertCache()
        mla = _make_mla(cache)
        md = _metadata()

        mla._global_topk(NUM_TOKENS, md)
        cache.invalidate()
        mla._global_topk(NUM_TOKENS, md)
        assert convert.call_count == 2

    def test_layers_share_one_cache(self, convert):
        """The buffer is shared per PP-rank model, so the conversion is too."""
        cache = GlmDsaTopkConvertCache()
        full, shared_a, shared_b = (_make_mla(cache) for _ in range(3))
        md = _metadata()

        out = full._global_topk(NUM_TOKENS, md)
        assert shared_a._global_topk(NUM_TOKENS, md) is out
        assert shared_b._global_topk(NUM_TOKENS, md) is out
        assert convert.call_count == 1

    def test_disabled_cache_converts_every_call(self, convert):
        mla = _make_mla(None)
        md = _metadata()

        for _ in range(3):
            mla._global_topk(NUM_TOKENS, md)
        assert convert.call_count == 3
