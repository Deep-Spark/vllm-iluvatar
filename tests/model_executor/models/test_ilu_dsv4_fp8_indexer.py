# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the fp8 lightning indexer (模块3.2).

Covers the grouped fp8 indexer refactor added to
``vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention``:

* ``IluDeepseekV4Fp8IndexerCache`` -> uint8, ``head_size = 132``; per block a
  contiguous ``[block_size, 128]`` e4m3 K region followed by a contiguous
  ``[block_size]`` fp32 ue8m0 scale region (K rows 16-aligned), ilu indexer
  backend, no 576 alignment.
* the ue8m0 scale helper the cache layout is written against.

The three compute steps of the indexer (q rope+fp8-quant, K insert, paged
mqa-logits) all run on ixformer ops now, so their parity tests live in the
ixformer suite -- see the per-step notes below. Runs on CPU; only ``torch`` and
the plugin module are needed.
"""

from __future__ import annotations

import contextlib
import importlib
import math

import pytest
import torch

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

FP8_ATTENTION_MODULE = (
    "vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention"
)

KV_BLOCK_SIZE = 256
INDEX_HEAD_DIM = 128
EXPECTED_INDEXER_HEAD_DIM = 132


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(FP8_ATTENTION_MODULE)


# ---------------------------------------------------------------------------
# gold helpers (independent of the code under test)
# ---------------------------------------------------------------------------


def _gold_ue8m0(amax: float) -> float:
    scale_raw = max(float(amax), 1e-10) / FP8_MAX
    return 2.0 ** math.ceil(math.log2(scale_raw))


# ---------------------------------------------------------------------------
# A1: ue8m0 scale
# ---------------------------------------------------------------------------


def test_ue8m0_scale_is_power_of_two_and_covers_amax(mod):
    amax = torch.tensor([0.0, 1e-9, 0.3, 7.0, 448.0, 5000.0])
    scale = mod._ue8m0_quant_scale(amax)
    # power of two: log2(scale) is integral
    log2 = torch.log2(scale)
    assert torch.allclose(log2, torch.round(log2), atol=1e-5)
    # dequantisable: amax / scale <= fp8_max (with the 1e-10 floor for zeros)
    assert torch.all(torch.clamp(amax, min=1e-10) / scale <= FP8_MAX + 1e-3)
    # matches the scalar gold
    for a in amax.tolist():
        assert mod._ue8m0_quant_scale(torch.tensor(a)).item() == pytest.approx(
            _gold_ue8m0(a), rel=1e-6
        )


# ---------------------------------------------------------------------------
# B1: q rope + fp8 quant (q_scale folded into weights)
#
# NOTE: the q-side rope+fp8-quant torch equivalent (``_indexer_q_rope_fp8_quant``
# / NV ``fused_indexer_q_rope_quant``) has been moved into the ixformer op
# ``ds4_fused_indexer_q_rope_fp8_quant`` (deepseek/fp8_indexer.py). Its native
# kernel vs torch-reference vs gold parity now lives in the ixformer test
# ``tests/python/inference/functions/test_ds4_fused_indexer_q_rope_fp8_quant.py``.
# The ue8m0 scale helper is still tested above (A1) since the K-insert path (B2)
# reuses it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# B3: fp8 mqa logits
#
# NOTE: the fp8 paged mqa-logits torch equivalent (``_indexer_fp8_mqa_logits_torch``
# / NV DeepGEMM ``fp8_fp4_mqa_logits``) has been replaced by the ivcore30-native
# ixformer op ``dsa_indexer_mqa_logits_with_blocks_fp8`` (deepseek/fp8_indexer.py).
# Its native kernel vs torch-reference vs fp32-gold parity now lives in the
# ixformer test
# ``tests/python/inference/functions/test_dsa_indexer_mqa_logits_with_blocks_fp8.py``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# B2: pool + norm + rope + grouped fp8 K insert
#
# NOTE: the K-insert torch equivalent (``_indexer_pool_norm_rope_fp8_insert_torch``
# / NV ``indexer_k_quant_and_cache``) has been replaced by the ixformer op
# ``ds4_indexer_compressor_pool_norm_rope_fp8_kv_insert``
# (deepseek/fp8_compressor.py), so the whole B1/B2/B3 indexer pipeline now runs on
# ixformer kernels. Parity for this op lives in the ixformer test suite.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A2: grouped fp8 indexer cache spec (head_size = 132)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _config_ctx():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.cache import CacheConfig

    vllm_config = VllmConfig(
        cache_config=CacheConfig(block_size=KV_BLOCK_SIZE, cache_dtype="fp8_ds_mla")
    )
    with set_current_vllm_config(vllm_config):
        yield vllm_config


def test_fp8_indexer_cache_spec(mod):
    from vllm.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV4IndexerBackend,
    )

    with _config_ctx() as vllm_config:
        cache = mod.IluDeepseekV4Fp8IndexerCache(
            head_dim=INDEX_HEAD_DIM,
            prefix="layers.0.attn.indexer.k_cache",
            cache_config=vllm_config.cache_config,
            compress_ratio=4,
        )
        spec = cache.get_kv_cache_spec(vllm_config)

    assert cache.head_dim == EXPECTED_INDEXER_HEAD_DIM
    assert cache.dtype == torch.uint8
    assert spec.head_size == EXPECTED_INDEXER_HEAD_DIM
    assert spec.dtype == torch.uint8
    assert spec.num_kv_heads == 1
    assert spec.compress_ratio == 4
    assert spec.block_size == KV_BLOCK_SIZE
    assert cache.get_attn_backend() is IluDeepseekV4IndexerBackend
