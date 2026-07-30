# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the fp8 lightning indexer (模块3.2).

Covers the grouped fp8 indexer refactor added to
``vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention``:

* ``IluDeepseekV4Fp8IndexerCache`` -> uint8, ``head_size = 132``; per block a
  contiguous ``[block_size, 128]`` e4m3 K region followed by a contiguous
  ``[block_size]`` fp32 ue8m0 scale region (K rows 16-aligned), ilu indexer
  backend, no 576 alignment.
* the torch equivalents (TODO(ilu_fp8_align_indexer_fp8)):
  - ``_indexer_pool_norm_rope_fp8_insert_torch`` (NV ``indexer_k_quant_and_cache``)
  (the q rope+fp8-quant step is now the ixformer op
  ``ds4_fused_indexer_q_rope_fp8_quant`` and the fp8 paged mqa-logits step is
  ``dsa_indexer_mqa_logits_with_blocks_fp8``; both parity tests live in ixformer.)

Each numeric test derives an independent fp32 gold (a simple per-token loop) and
checks the vectorised implementation against it, plus the ue8m0 / fp8 layout
invariants. Runs on CPU; only ``torch`` and the plugin module are needed.
"""

from __future__ import annotations

import contextlib
import importlib
import math

import pytest
import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
SCALE_BYTES = 4

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


def _fp8_round(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 -> fp32 (bit-exact with the impl's ``.to(fp8)``)."""
    return torch.clamp(x, -FP8_MAX, FP8_MAX).to(FP8_DTYPE).float()


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
# ---------------------------------------------------------------------------


def _gold_insert_one(
    *, state_cache, block_table, req, pos, win, cr, overlap, head_dim,
    state_width, sbs, norm_w, cos_sin_cache, rope_head_dim, eps,
):
    kv = torch.zeros(win, head_dim)
    score = torch.full((win, head_dim), float("-inf"))
    for w in range(win):
        src = pos - win + 1 + w
        head_off = head_dim if (overlap and w >= cr) else 0
        if src >= 0:
            bn = int(block_table[req, src // sbs])
            ib = src % sbs
            row = state_cache[bn, ib].float()
            kv[w] = row[head_off:head_off + head_dim]
            score[w] = row[state_width + head_off: state_width + head_off + head_dim]
    soft = torch.softmax(score, dim=0)
    pooled = (soft * kv).sum(dim=0)
    var = (pooled * pooled).mean()
    normed = pooled * torch.rsqrt(var + eps) * norm_w.float()

    half = rope_head_dim // 2
    nope = head_dim - rope_head_dim
    cp = (pos // cr) * cr
    cos = cos_sin_cache[cp, :half].float()
    sin = cos_sin_cache[cp, half:rope_head_dim].float()
    pairs = normed.view(head_dim // 2, 2)
    even, odd = pairs[:, 0].clone(), pairs[:, 1].clone()
    np2 = nope // 2
    e, o = even[np2:].clone(), odd[np2:].clone()
    even[np2:] = e * cos - o * sin
    odd[np2:] = e * sin + o * cos
    rotated = torch.empty(head_dim)
    rotated[0::2] = even
    rotated[1::2] = odd
    return rotated


def test_pool_norm_rope_fp8_insert_matches_gold(mod):
    torch.manual_seed(2)
    head_dim, rope_head_dim = 8, 4
    cr = 2
    overlap = 1  # coff = 2 -> state_width = 2 * head_dim
    state_width = (1 + overlap) * head_dim
    sbs = 16  # state block size
    kbs = KV_BLOCK_SIZE
    eps = 1e-6

    win = (1 + overlap) * cr
    nreq = 1
    n_state_blocks = 2
    state_cache = torch.randn(n_state_blocks, sbs, 2 * state_width)
    block_table = torch.arange(nreq * 4, dtype=torch.int32).reshape(nreq, 4)
    norm_w = torch.randn(head_dim)
    max_pos = 32
    cos_sin_cache = torch.randn(max_pos, rope_head_dim)

    # 3 tokens for req 0 at positions 4,5,6; window-end is (pos+1)%cr==0 -> pos=5
    positions = torch.tensor([4, 5, 6])
    token_to_req = torch.tensor([0, 0, 0])
    slot_mapping = torch.tensor([10, 11, 12])  # >=0 -> valid state slots
    kv_slot_mapping = torch.tensor([20, 21, 22])

    k_cache = torch.zeros(1, kbs, head_dim + SCALE_BYTES, dtype=torch.uint8)

    mod._indexer_pool_norm_rope_fp8_insert_torch(
        state_cache=state_cache,
        token_to_req_indices=token_to_req,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        rms_norm_weight=norm_w,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        compress_ratio=cr,
        rope_head_dim=rope_head_dim,
        rms_norm_eps=eps,
        head_dim=head_dim,
    )

    # Grouped block layout: read the K region (token p at [p*head_dim:...]) and
    # scale region (token p at [kbs*head_dim + p*4:...]) via a flat block view.
    kc = k_cache.view(1, kbs * (head_dim + SCALE_BYTES))
    s_base = kbs * head_dim

    def _k_slot(p: int) -> torch.Tensor:
        return kc[0, p * head_dim : (p + 1) * head_dim]

    def _scale_slot(p: int) -> torch.Tensor:
        off = s_base + p * SCALE_BYTES
        return kc[0, off : off + SCALE_BYTES]

    # only pos=5 (token idx 1) is a window end -> kv_slot 21
    written_k = _k_slot(21)
    written_scale = _scale_slot(21)
    for p in (20, 22):
        assert torch.all(_k_slot(p) == 0), "non-window-end K must not write"
        assert torch.all(_scale_slot(p) == 0), "non-window-end scale must not write"
    assert torch.any(written_k != 0), "window-end token must write"

    rotated = _gold_insert_one(
        state_cache=state_cache, block_table=block_table, req=0, pos=5, win=win,
        cr=cr, overlap=overlap, head_dim=head_dim, state_width=state_width,
        sbs=sbs, norm_w=norm_w, cos_sin_cache=cos_sin_cache,
        rope_head_dim=rope_head_dim, eps=eps,
    )
    scale = _gold_ue8m0(float(rotated.abs().max()))
    gold_fp8 = _fp8_round(rotated / scale)

    got_fp8 = written_k.contiguous().view(FP8_DTYPE).float()
    got_scale = written_scale.contiguous().view(torch.float32).item()
    assert got_scale == pytest.approx(scale, rel=1e-6)
    assert torch.allclose(got_fp8, gold_fp8, rtol=0, atol=0)


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
