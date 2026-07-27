# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests (T5) for the DeepSeek-V4 FP8 KV cache specs and page algebra.

Covers design §5.1 ("KV cache / page 结构" 取值表) for the four spec flavours that
Stage 1 must expose:

* main MLA        -> ``IluDeepseekV4Fp8Attention.get_kv_cache_spec()``
* SWA             -> community ``DeepseekV4SWACache``
* indexer K cache -> community ``DeepseekV4IndexerCache``
* compressor state-> community ``CompressorStateCache``

Every expected number below is hard-coded from design §5.1 on purpose: reading it
back from the code under test would make the assertion vacuous.

The three cache holders are community classes reused verbatim (design §4.2), so
they are constructed directly here. The main MLA spec is exercised through the
plugin class' ``get_kv_cache_spec`` bound to a minimal duck-typed ``self`` — the
method only reads ``compress_ratio`` / ``head_dim`` / ``kv_cache_dtype`` /
``kv_cache_torch_dtype`` (community ``attention.py:605-625``), and building a
whole attention layer is Task 1.2's job, not this file's.

Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6, 4.8, 4.9
"""

from __future__ import annotations

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Expected values, hard-coded from design §5.1 (one constant per table row).
# ---------------------------------------------------------------------------

# "主 MLA per-token 字节" / "SWA per-token 字节"
#   = 448B FP8 NoPE + 128B bf16 RoPE + 8B UE8M0 scale
EXPECTED_PER_TOKEN_BYTES = 584
# "alignment（主 MLA / SWA / indexer / compressor state）"
EXPECTED_ALIGNMENT = 576
# "model_version" — design §5.1 sources it from attention.py:619 (main MLA) and
# sparse_swa.py:102 (SWA) only. The indexer / compressor-state specs deliberately
# leave it unset: SlidingWindowMLASpec.real_page_size_bytes switches to the 584B
# per-token formula as soon as model_version == "deepseek_v4", which would be
# wrong for a float32 compressor state page. Hence model_version is asserted for
# the two specs that set it, and alignment (576) for all four.
EXPECTED_MODEL_VERSION = "deepseek_v4"
# "cache_config.cache_dtype" / "KV cache tensor dtype"
EXPECTED_CACHE_DTYPE_STR = "fp8_ds_mla"
EXPECTED_KV_TENSOR_DTYPE = torch.uint8
# "MLAAttentionSpec.head_size（主 MLA）" — semantic head_dim
EXPECTED_MAIN_HEAD_SIZE = 512
# "SWA block_size"
EXPECTED_SWA_BLOCK_SIZE = 64
# "kernel block size / preferred block"
EXPECTED_KERNEL_BLOCK_SIZES = [256]
# "indexer K cache head_dim" = 128 + 128 // 128 * 4
EXPECTED_INDEXER_HEAD_DIM = 132
# "compressor state dtype"
EXPECTED_COMPRESSOR_STATE_DTYPE = torch.float32
# "compressor block_size": cr=4 -> 4; cr=128 -> 8
EXPECTED_COMPRESSOR_BLOCK_SIZE = {4: 4, 128: 8}
# "compressor sliding_window" = coff * cr: cr=4 -> 8; cr=128 -> 128
EXPECTED_COMPRESSOR_SLIDING_WINDOW = {4: 8, 128: 128}

# config.json: sliding_window = 128, index_head_dim = 128; the indexer's fp8
# quant block is 128 (community ``attention.py`` DeepseekV4Indexer).
SLIDING_WINDOW = 128
INDEX_HEAD_DIM = 128
INDEXER_QUANT_BLOCK = 128

# KV cache block size used by the DSV4 sparse MLA kernel.
KV_BLOCK_SIZE = 256

FP8_ATTENTION_MODULE = (
    "vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vllm_config(
    *,
    block_size: int = KV_BLOCK_SIZE,
    cache_dtype: str = EXPECTED_CACHE_DTYPE_STR,
):
    """A fresh VllmConfig carrying only the cache knobs the specs read."""
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheConfig

    return VllmConfig(
        cache_config=CacheConfig(block_size=block_size, cache_dtype=cache_dtype)
    )


@contextlib.contextmanager
def _config_ctx(**kwargs):
    """Activate a fresh VllmConfig (cache holders call get_current_vllm_config)."""
    from vllm.config import set_current_vllm_config

    vllm_config = _vllm_config(**kwargs)
    with set_current_vllm_config(vllm_config):
        yield vllm_config


def _per_token_bytes(spec) -> int:
    """Page bytes divided by the number of tokens a page holds."""
    return spec.real_page_size_bytes // spec.storage_block_size


def _main_mla_spec(vllm_config, *, compress_ratio: int):
    """Call the plugin's ``get_kv_cache_spec`` with a minimal duck-typed self."""
    fp8_attention = importlib.import_module(FP8_ATTENTION_MODULE)

    attn = SimpleNamespace(
        compress_ratio=compress_ratio,
        head_dim=EXPECTED_MAIN_HEAD_SIZE,
        kv_cache_dtype=EXPECTED_CACHE_DTYPE_STR,
        kv_cache_torch_dtype=EXPECTED_KV_TENSOR_DTYPE,
    )
    return fp8_attention.IluDeepseekV4Fp8Attention.get_kv_cache_spec(
        attn, vllm_config
    )


def _swa_cache(vllm_config, *, prefix: str = "layers.0.attn.swa_cache"):
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    return DeepseekV4SWACache(
        head_dim=EXPECTED_MAIN_HEAD_SIZE,
        window_size=SLIDING_WINDOW,
        dtype=EXPECTED_KV_TENSOR_DTYPE,
        prefix=prefix,
        cache_config=vllm_config.cache_config,
    )


def _indexer_k_cache(
    vllm_config,
    *,
    compress_ratio: int = 4,
    prefix: str = "layers.0.attn.indexer.k_cache",
):
    from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache

    # Community derivation (DeepseekV4Indexer.__init__):
    #   head_dim + head_dim // quant_block_size * 4
    head_dim = INDEX_HEAD_DIM + INDEX_HEAD_DIM // INDEXER_QUANT_BLOCK * 4
    return head_dim, DeepseekV4IndexerCache(
        head_dim=head_dim,
        dtype=EXPECTED_KV_TENSOR_DTYPE,
        prefix=prefix,
        cache_config=vllm_config.cache_config,
        compress_ratio=compress_ratio,
    )


def _compressor_state_cache(
    *,
    compress_ratio: int,
    prefix: str = "layers.0.attn.compressor.state_cache",
):
    from vllm.models.deepseek_v4.compressor import CompressorStateCache

    # design §5.1: coff = 1 + (compress_ratio == 4); state_dim = 2 * coff * head_dim
    coff = 1 + (compress_ratio == 4)
    state_dim = 2 * coff * EXPECTED_MAIN_HEAD_SIZE
    return CompressorStateCache(
        state_dim=state_dim,
        dtype=EXPECTED_COMPRESSOR_STATE_DTYPE,
        compress_ratio=compress_ratio,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# main MLA spec (Requirements 4.2, 4.3, 4.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_main_mla_spec_page_algebra(compress_ratio):
    with _config_ctx() as vllm_config:
        spec = _main_mla_spec(vllm_config, compress_ratio=compress_ratio)

    assert _per_token_bytes(spec) == EXPECTED_PER_TOKEN_BYTES
    assert spec.alignment == EXPECTED_ALIGNMENT
    assert spec.model_version == EXPECTED_MODEL_VERSION
    assert spec.cache_dtype_str == EXPECTED_CACHE_DTYPE_STR
    assert spec.dtype == EXPECTED_KV_TENSOR_DTYPE
    assert spec.head_size == EXPECTED_MAIN_HEAD_SIZE
    assert spec.num_kv_heads == 1
    assert spec.block_size == KV_BLOCK_SIZE
    assert spec.compress_ratio == compress_ratio


@pytest.mark.parametrize("compress_ratio", [0, 1])
def test_main_mla_spec_is_none_for_swa_only_layers(compress_ratio):
    """compress_ratio <= 1 layers are covered by DeepseekV4SWACache instead."""
    with _config_ctx() as vllm_config:
        assert _main_mla_spec(vllm_config, compress_ratio=compress_ratio) is None


# ---------------------------------------------------------------------------
# SWA spec (Requirements 4.2, 4.3, 4.6)
# ---------------------------------------------------------------------------


def test_swa_cache_spec_page_algebra():
    with _config_ctx() as vllm_config:
        swa_cache = _swa_cache(vllm_config)
        spec = swa_cache.get_kv_cache_spec(vllm_config)

    assert swa_cache.block_size == EXPECTED_SWA_BLOCK_SIZE
    assert spec.block_size == EXPECTED_SWA_BLOCK_SIZE
    assert _per_token_bytes(spec) == EXPECTED_PER_TOKEN_BYTES
    assert spec.alignment == EXPECTED_ALIGNMENT
    assert spec.model_version == EXPECTED_MODEL_VERSION
    assert spec.cache_dtype_str == EXPECTED_CACHE_DTYPE_STR
    assert spec.dtype == EXPECTED_KV_TENSOR_DTYPE
    assert spec.head_size == EXPECTED_MAIN_HEAD_SIZE
    assert spec.sliding_window == SLIDING_WINDOW


def test_dsv4_sparse_backend_supported_kernel_block_sizes():
    from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
        IluDeepseekV4FlashMLASparseBackend,
    )

    fp8_attention = importlib.import_module(FP8_ATTENTION_MODULE)
    backend_cls = fp8_attention.IluDeepseekV4Fp8Attention.backend_cls

    assert backend_cls is IluDeepseekV4FlashMLASparseBackend
    assert (
        backend_cls.get_supported_kernel_block_sizes()
        == EXPECTED_KERNEL_BLOCK_SIZES
    )


# ---------------------------------------------------------------------------
# indexer K cache spec (Requirements 4.3, 4.4)
# ---------------------------------------------------------------------------


def test_indexer_k_cache_spec_head_dim_and_dtype():
    with _config_ctx() as vllm_config:
        head_dim, k_cache = _indexer_k_cache(vllm_config)
        spec = k_cache.get_kv_cache_spec(vllm_config)

    assert head_dim == EXPECTED_INDEXER_HEAD_DIM
    assert spec.head_size == EXPECTED_INDEXER_HEAD_DIM
    assert spec.dtype == EXPECTED_KV_TENSOR_DTYPE
    assert spec.alignment == EXPECTED_ALIGNMENT
    assert spec.num_kv_heads == 1
    assert spec.compress_ratio == 4


# ---------------------------------------------------------------------------
# compressor state spec (Requirements 4.3, 4.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compressor_state_cache_spec(compress_ratio):
    with _config_ctx() as vllm_config:
        state_cache = _compressor_state_cache(compress_ratio=compress_ratio)
        spec = state_cache.get_kv_cache_spec(vllm_config)

    assert state_cache.dtype == EXPECTED_COMPRESSOR_STATE_DTYPE
    assert spec.dtype == EXPECTED_COMPRESSOR_STATE_DTYPE
    assert state_cache.block_size == EXPECTED_COMPRESSOR_BLOCK_SIZE[compress_ratio]
    assert spec.block_size == EXPECTED_COMPRESSOR_BLOCK_SIZE[compress_ratio]
    assert (
        state_cache.sliding_window
        == EXPECTED_COMPRESSOR_SLIDING_WINDOW[compress_ratio]
    )
    assert (
        spec.sliding_window == EXPECTED_COMPRESSOR_SLIDING_WINDOW[compress_ratio]
    )
    assert spec.alignment == EXPECTED_ALIGNMENT
    assert spec.num_kv_heads == 1


def test_compressor_state_dim_matches_design_table():
    """state_dim = 2 * coff * head_dim -> cr=4: 2048; cr=128: 1024 (design §5.1)."""
    with _config_ctx():
        c4 = _compressor_state_cache(
            compress_ratio=4, prefix="layers.0.attn.compressor.state_cache"
        )
        c128 = _compressor_state_cache(
            compress_ratio=128, prefix="layers.1.attn.compressor.state_cache"
        )

    assert c4.state_dim == 2048
    assert c128.state_dim == 1024
