# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-caching admission checks and the compressed-K page declaration.

The compressed-K page is an exact 1/(2 * compress_ratio) of the main KV page,
so ``unify_kv_cache_spec_page_size`` would scale its ``block_size`` up instead
of padding it, and hits -- which round down to the LCM of the group block sizes
-- would round to zero for every request shorter than the scaled block. These
tests pin both halves of that: the padded page keeps the block sizes equal, and
the configurations whose per-request state cannot be restored are rejected.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm_iluvatar.models.qwen4_exp.model import (
    _verify_prefix_caching_support,
)
from vllm_iluvatar.models.qwen4_exp.qsa_cache import QSACompressedKeyCache

BLOCK_SIZE = 1024
COMPRESS_RATIO = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 128
INDEX_HEAD_DIM = 128


def _vllm_config(*, enable_prefix_caching: bool) -> SimpleNamespace:
    cache_config = CacheConfig(block_size=BLOCK_SIZE)
    cache_config.enable_prefix_caching = enable_prefix_caching
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel_config: NUM_KV_HEADS,
        get_head_size=lambda: HEAD_SIZE,
        hf_text_config=SimpleNamespace(ple_layer_ids=[]),
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        kv_transfer_config=None,
    )


def _compressed_key_cache(vllm_config, prefix: str) -> QSACompressedKeyCache:
    return QSACompressedKeyCache(
        head_size=INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        cache_config=vllm_config.cache_config,
        prefix=prefix,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=COMPRESS_RATIO,
    )


def _main_kv_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )


def test_compressed_key_page_is_unpadded_without_prefix_caching() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=False)
    spec = _compressed_key_cache(vllm_config, "qsa.0").get_kv_cache_spec(vllm_config)

    assert spec.page_size_padded is None
    # One row per completed group, bf16.
    assert spec.page_size_bytes == (BLOCK_SIZE // COMPRESS_RATIO * INDEX_HEAD_DIM * 2)


def test_compressed_key_page_matches_main_kv_page_with_prefix_caching() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=True)
    spec = _compressed_key_cache(vllm_config, "qsa.0").get_kv_cache_spec(vllm_config)

    main_spec = _main_kv_spec()
    assert spec.page_size_bytes == main_spec.page_size_bytes
    assert spec.block_size == main_spec.block_size


def test_padded_compressed_page_keeps_the_hit_alignment_at_block_size() -> None:
    """The whole point: unify must not scale the compressed group's block_size."""
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4, 64), (2, 16, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )
    main_spec = _main_kv_spec()
    padded = _compressed_key_cache(
        _vllm_config(enable_prefix_caching=True), "qsa.0"
    ).get_kv_cache_spec(_vllm_config(enable_prefix_caching=True))
    unpadded = _compressed_key_cache(
        _vllm_config(enable_prefix_caching=False), "qsa.1"
    ).get_kv_cache_spec(_vllm_config(enable_prefix_caching=False))

    # MambaSpec pages are already aligned to the main KV page by the platform.
    aligned_mamba = MambaSpec(
        block_size=mamba_spec.block_size,
        shapes=mamba_spec.shapes,
        dtypes=mamba_spec.dtypes,
        mamba_cache_mode="align",
        page_size_padded=main_spec.page_size_bytes,
    )

    with_padding = unify_kv_cache_spec_page_size(
        {"attn": main_spec, "mamba": aligned_mamba, "compressed": padded}
    )
    assert {spec.block_size for spec in with_padding.values()} == {BLOCK_SIZE}

    without_padding = unify_kv_cache_spec_page_size(
        {"attn": main_spec, "mamba": aligned_mamba, "compressed": unpadded}
    )
    scaled = without_padding["compressed"].block_size
    page_ratio = main_spec.page_size_bytes // unpadded.page_size_bytes
    assert scaled == BLOCK_SIZE * page_ratio, (
        "expected unify to scale the unpadded compressed group instead of "
        "padding it; this test's premise is what the padding avoids"
    )


def test_prefix_caching_accepts_ple() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=True)
    vllm_config.model_config.hf_text_config.ple_layer_ids = [2]
    _verify_prefix_caching_support(vllm_config)


def test_prefix_caching_is_rejected_with_kv_connector() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=True)
    vllm_config.kv_transfer_config = object()
    with pytest.raises(NotImplementedError, match="KV"):
        _verify_prefix_caching_support(vllm_config)


def test_prefix_caching_is_rejected_with_prefix_match_unit() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=True)
    vllm_config.cache_config.prefix_match_unit = 64
    with pytest.raises(NotImplementedError, match="prefix-match-unit"):
        _verify_prefix_caching_support(vllm_config)


def test_checks_are_skipped_without_prefix_caching() -> None:
    vllm_config = _vllm_config(enable_prefix_caching=False)
    vllm_config.model_config.hf_text_config.ple_layer_ids = [2]
    vllm_config.kv_transfer_config = object()
    _verify_prefix_caching_support(vllm_config)
