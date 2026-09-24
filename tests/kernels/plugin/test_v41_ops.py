# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 compressor / indexer plugin kernels.

Each op is checked against a pure-torch transcription of the reference Triton
(``off/vllm/vllm/models/deepseek_v41/common/ops`` and
``.../deepseek_v4/common/ops/fused_indexer_q.py``). The packed fp8_ds_mla insert
additionally cross-checks the ixformer kernel it replaces: the two must agree
byte for byte, which is what says dropping the block table changed nothing.

Tolerances: the kernels contract the rotation into FMAs where torch does not, so
the fp32 intermediates differ in the last bits. Every output here is bf16 or
e4m3, whose steps are far wider than that, so the comparisons are on the stored
values rather than on bits -- except the scales and the ixformer cross-check,
which must be exact.
"""

from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401  (registers torch.ops._C)

DEVICE = "cuda"
LATENT_DIM = 512
INDEX_HEAD_DIM = 128
ROPE_DIM = 64
FP8_MAX = 448.0
# The packed fp8_ds_mla record: 448 e4m3 + 64 bf16 rope + 7 ue8m0 + 1 pad.
PACKED_DATA_BYTES = 576
PACKED_NOPE_BYTES = 448
PACKED_SCALE_BYTES = 8
PACKED_NUM_SCALES = 7
PACKED_ROW_BYTES = PACKED_DATA_BYTES + PACKED_SCALE_BYTES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _make_cos_sin(max_pos: int, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """[max_pos, rope_dim] fp32, cos half then sin half."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    angles = torch.arange(max_pos).float()[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).to(DEVICE)


def _rope_rows(
    rows: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J forward rotation of the trailing ROPE_DIM dims, in fp32."""
    out = rows.clone()
    half = ROPE_DIM // 2
    cs = cos_sin[positions.to(torch.long)]
    cos, sin = cs[:, :half], cs[:, half:ROPE_DIM]
    tail = rows[:, -ROPE_DIM:]
    even, odd = tail[:, 0::2], tail[:, 1::2]
    new_tail = torch.empty_like(tail)
    new_tail[:, 0::2] = even * cos - odd * sin
    new_tail[:, 1::2] = odd * cos + even * sin
    out[:, -ROPE_DIM:] = new_tail
    return out


def _bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _paged_layout(num_tokens: int, rows_per_page: int, compress_ratio: int, seed: int):
    """Positions, a block table, and the compressed slots they imply.

    Mirrors ``get_compressed_slot_mapping``: the slot is the global compressed
    row index ``page * rows_per_page + row``. Page 0 is left unused so the
    ixformer kernel, which treats it as the null page, agrees with ours.

    Tokens are dealt round-robin to requests so every ``(request, position)``
    pair -- and therefore every slot -- is distinct. Two tokens sharing a slot
    would make the last writer win and the per-token comparison meaningless.
    """
    del seed
    num_reqs = 3
    token_to_req = (torch.arange(num_tokens) % num_reqs).to(torch.int32)
    positions = (torch.arange(num_tokens) // num_reqs).to(torch.int64)
    max_col = (int(positions.max()) // compress_ratio) // rows_per_page + 1
    block_table = torch.arange(1, num_reqs * max_col + 1, dtype=torch.int32).reshape(
        num_reqs, max_col
    )

    groups = positions // compress_ratio
    cols = groups // rows_per_page
    pages = block_table[token_to_req.long(), cols]
    slots = pages.to(torch.int64) * rows_per_page + (groups % rows_per_page)
    # A quarter of the tokens carry no slot, as CUDA-graph padding does.
    slots[::4] = -1
    num_pages = int(block_table.max()) + 1
    return (
        positions.to(DEVICE),
        slots.to(DEVICE),
        block_table.to(DEVICE),
        token_to_req.to(DEVICE),
        num_pages,
    )


def _publishes(positions: torch.Tensor, slots: torch.Tensor, ratio: int):
    return (slots >= 0) & ((positions + 1) % ratio == 0)


# ---------------------------------------------------------------------------
# B1: ratio-1 latent RMSNorm
# ---------------------------------------------------------------------------


def test_compress_norm_cr1():
    torch.manual_seed(0)
    num_tokens = 71
    eps = 1e-6
    kv_score = torch.randn(num_tokens, LATENT_DIM, device=DEVICE)
    weight = torch.randn(LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    slots = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    slots[::5] = -1

    latent = torch.zeros(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_compress_norm_cr1(kv_score, slots, weight, eps, latent)

    # Reference: the weight multiply happens in fp32, before the bf16 store
    # (``_store_latent``), not after it.
    row = kv_score.float()
    rms = torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + eps)
    expected = ((row * rms) * weight.float()).to(torch.bfloat16)

    live = slots >= 0
    torch.testing.assert_close(
        latent[live].float(), expected[live].float(), rtol=8e-3, atol=8e-3
    )
    # Slotless rows are left alone, exactly as the reference leaves them.
    assert torch.all(latent[~live] == 0)


# ---------------------------------------------------------------------------
# B2: RoPE + compressed-cache insert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_rope_insert_bf16(compress_ratio: int):
    torch.manual_seed(1)
    num_tokens, rows_per_page = 64, 8
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=1
    )
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(
        num_pages,
        rows_per_page,
        LATENT_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, cache, slots, compress_ratio, None
    )

    rope_pos = (positions // compress_ratio) * compress_ratio
    expected = _rope_rows(latent.float(), rope_pos, cos_sin).to(torch.bfloat16)
    ok = _publishes(positions, slots, compress_ratio)

    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        got = cache[slot // rows_per_page, slot % rows_per_page]
        torch.testing.assert_close(
            got.float(), expected[t].float(), rtol=8e-3, atol=8e-3
        )
    # Nothing was published for the skipped tokens.
    written = {
        (int(slots[t]) // rows_per_page, int(slots[t]) % rows_per_page)
        for t in torch.nonzero(ok).flatten().tolist()
    }
    for page in range(num_pages):
        for row in range(rows_per_page):
            if (page, row) not in written:
                assert torch.all(cache[page, row] == 0)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_rope_insert_packed_matches_ixformer(compress_ratio: int):
    """The plugin insert and the ixformer one must write identical records.

    The plugin derives page and row from the compressed slot; ixformer walks the
    block table. Byte equality is the evidence that the two addressing agree
    and that dropping the block table is a no-op.
    """
    ixfops = pytest.importorskip("ixformer.inference.functions")
    if not hasattr(ixfops, "v41_rope_fp8_insert_by_token"):
        pytest.skip("this ixformer build has no v41_rope_fp8_insert_by_token")

    torch.manual_seed(2)
    num_tokens, rows_per_page = 64, 8
    positions, slots, block_table, token_to_req, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=2
    )
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)

    ours = torch.zeros(
        num_pages,
        rows_per_page,
        PACKED_ROW_BYTES,
        device=DEVICE,
        dtype=torch.uint8,
    )
    theirs = torch.zeros_like(ours)

    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, ours, slots, compress_ratio, None
    )
    ixfops.v41_rope_fp8_insert_by_token(
        theirs,
        latent,
        positions,
        slots,
        block_table,
        token_to_req,
        cos_sin,
        rows_per_page * compress_ratio,
        compress_ratio,
    )
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_rope_insert_packed_record(compress_ratio: int):
    """Decode the record back and check it against the reference rotation."""
    torch.manual_seed(3)
    num_tokens, rows_per_page = 48, 8
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=3
    )
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(
        num_pages,
        rows_per_page,
        PACKED_ROW_BYTES,
        device=DEVICE,
        dtype=torch.uint8,
    )
    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, cache, slots, compress_ratio, None
    )

    rope_pos = (positions // compress_ratio) * compress_ratio
    rotated = _rope_rows(latent.float(), rope_pos, cos_sin)
    nope = rotated[:, :PACKED_NOPE_BYTES].reshape(num_tokens, PACKED_NUM_SCALES, 64)
    amax = nope.abs().amax(-1)
    exponent = torch.where(
        amax == 0, torch.zeros_like(amax), torch.ceil(torch.log2(amax / FP8_MAX))
    )

    flat = cache.reshape(num_pages, -1)
    ok = _publishes(positions, slots, compress_ratio)
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        page, row = slot // rows_per_page, slot % rows_per_page
        base = row * PACKED_DATA_BYTES
        sbase = rows_per_page * PACKED_DATA_BYTES + row * PACKED_SCALE_BYTES

        scale_bits = flat[page, sbase : sbase + PACKED_NUM_SCALES].float() - 127.0
        torch.testing.assert_close(scale_bits, exponent[t], rtol=0, atol=0)
        assert int(flat[page, sbase + PACKED_NUM_SCALES]) == 0

        scales = torch.exp2(scale_bits).repeat_interleave(64)
        got_nope = (
            flat[page, base : base + PACKED_NOPE_BYTES]
            .view(torch.float8_e4m3fn)
            .float()
            * scales
        )
        # One e4m3 step is 2^-3 relative, so half a step is ~6%.
        torch.testing.assert_close(got_nope, nope[t].reshape(-1), rtol=7e-2, atol=1e-3)

        got_rope = (
            flat[page, base + PACKED_NOPE_BYTES : base + PACKED_DATA_BYTES]
            .view(torch.bfloat16)
            .float()
        )
        torch.testing.assert_close(
            got_rope, rotated[t, PACKED_NOPE_BYTES:], rtol=8e-3, atol=8e-3
        )


def test_rope_insert_plain_fp8():
    torch.manual_seed(4)
    num_tokens, rows_per_page, compress_ratio = 32, 8, 1
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=4
    )
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.tensor([0.5], device=DEVICE, dtype=torch.float32)
    cache = torch.zeros(
        num_pages,
        rows_per_page,
        LATENT_DIM,
        device=DEVICE,
        dtype=torch.float8_e4m3fn,
    )
    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, cache, slots, compress_ratio, scale
    )

    rotated = _bf16_round(_rope_rows(latent.float(), positions, cos_sin))
    expected = (rotated / scale).clamp(-FP8_MAX, FP8_MAX)
    ok = _publishes(positions, slots, compress_ratio)
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        got = cache[slot // rows_per_page, slot % rows_per_page].float()
        torch.testing.assert_close(got, expected[t], rtol=7e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# A1 / A2: indexer K store
# ---------------------------------------------------------------------------


def _ref_indexer_k(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compress_ratio: int,
) -> torch.Tensor:
    """k_norm then RoPE, with the reference's two bf16 round-trips."""
    k = k_pre.float()
    variance = k.pow(2).mean(-1, keepdim=True)
    k = _bf16_round(k * torch.rsqrt(variance + eps) * weight.float())
    rope_pos = (positions // compress_ratio) * compress_ratio
    return _bf16_round(_rope_rows(k, rope_pos, cos_sin))


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_indexer_k_store_bf16(compress_ratio: int):
    torch.manual_seed(5)
    num_tokens, rows_per_page, eps = 64, 8, 1e-6
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=5
    )
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(
        num_pages,
        rows_per_page,
        INDEX_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        cache,
        slots,
        compress_ratio,
        False,
    )

    expected = _ref_indexer_k(k_pre, positions, cos_sin, weight, eps, compress_ratio)
    ok = _publishes(positions, slots, compress_ratio)
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        got = cache[slot // rows_per_page, slot % rows_per_page].float()
        torch.testing.assert_close(got, expected[t], rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_indexer_k_store_fp8(compress_ratio: int):
    torch.manual_seed(6)
    num_tokens, rows_per_page, eps = 64, 8, 1e-6
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=6
    )
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    row_bytes = INDEX_HEAD_DIM + 4
    cache = torch.zeros(
        num_pages, rows_per_page, row_bytes, device=DEVICE, dtype=torch.uint8
    )

    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        cache,
        slots,
        compress_ratio,
        True,
    )

    expected = _ref_indexer_k(k_pre, positions, cos_sin, weight, eps, compress_ratio)
    absmax = expected.abs().amax(-1).clamp(min=1e-4)
    exp_scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_MAX)))

    flat = cache.reshape(num_pages, -1)
    ok = _publishes(positions, slots, compress_ratio)
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        page, row = slot // rows_per_page, slot % rows_per_page
        got_scale = flat[
            page,
            rows_per_page * INDEX_HEAD_DIM + row * 4 : rows_per_page * INDEX_HEAD_DIM
            + row * 4
            + 4,
        ].view(torch.float32)
        torch.testing.assert_close(got_scale[0], exp_scale[t], rtol=0, atol=0)

        values = (
            flat[page, row * INDEX_HEAD_DIM : (row + 1) * INDEX_HEAD_DIM]
            .view(torch.float8_e4m3fn)
            .float()
            * got_scale[0]
        )
        torch.testing.assert_close(values, expected[t], rtol=7e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# A3: indexer Q RoPE
# ---------------------------------------------------------------------------


def test_indexer_q_rope_bf16():
    torch.manual_seed(7)
    num_tokens, num_heads = 33, 8
    scale = 0.037
    positions = (torch.arange(num_tokens, dtype=torch.int64) % 97).to(DEVICE)
    cos_sin = _make_cos_sin(128)
    q = torch.randn(
        num_tokens,
        num_heads,
        INDEX_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(num_tokens, num_heads, device=DEVICE, dtype=torch.bfloat16)
    q_ref, w_ref = q.clone(), weights.clone()

    torch.ops._C.v41_indexer_q_rope(q, positions, cos_sin, weights, weights, scale)

    rows = q_ref.float().reshape(-1, INDEX_HEAD_DIM)
    rope_pos = positions.repeat_interleave(num_heads)
    expected = _rope_rows(rows, rope_pos, cos_sin).reshape(
        num_tokens, num_heads, INDEX_HEAD_DIM
    )
    torch.testing.assert_close(q.float(), expected, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(
        weights.float(),
        (w_ref.float() * scale).to(torch.bfloat16).float(),
        rtol=8e-3,
        atol=8e-3,
    )


def test_indexer_q_rope_fp8():
    torch.manual_seed(8)
    num_tokens, num_heads = 33, 8
    scale = 0.037
    positions = (torch.arange(num_tokens, dtype=torch.int64) % 97).to(DEVICE)
    cos_sin = _make_cos_sin(128)
    q = torch.randn(
        num_tokens,
        num_heads,
        INDEX_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(num_tokens, num_heads, device=DEVICE, dtype=torch.bfloat16)

    q_fp8, weights_out = torch.ops._C.v41_indexer_q_rope_fp8(
        q, positions, cos_sin, weights, scale
    )
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert weights_out.dtype == torch.float32

    rows = q.float().reshape(-1, INDEX_HEAD_DIM)
    rope_pos = positions.repeat_interleave(num_heads)
    rotated = _rope_rows(rows, rope_pos, cos_sin)
    # Only the rotated tail is re-rounded; the NoPE half is already bf16.
    rotated[:, -ROPE_DIM:] = _bf16_round(rotated[:, -ROPE_DIM:])
    amax = rotated.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    q_scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))

    got = q_fp8.float().reshape(-1, INDEX_HEAD_DIM) * q_scale
    torch.testing.assert_close(got, rotated, rtol=7e-2, atol=1e-3)
    torch.testing.assert_close(
        weights_out,
        weights.float() * q_scale.reshape(num_tokens, num_heads) * scale,
        rtol=8e-3,
        atol=1e-6,
    )


def test_empty_batches_are_noops():
    cos_sin = _make_cos_sin(8)
    empty_i64 = torch.empty(0, device=DEVICE, dtype=torch.int64)
    latent = torch.empty(0, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(1, 4, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_rope_insert(latent, empty_i64, cos_sin, cache, empty_i64, 1, None)
    assert torch.all(cache == 0)

    kv_score = torch.empty(0, LATENT_DIM, device=DEVICE)
    weight = torch.ones(LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_compress_norm_cr1(kv_score, empty_i64, weight, 1e-6, latent)
