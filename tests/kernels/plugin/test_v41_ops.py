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
INT8_MAX = 127.0
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
        latent, positions, cos_sin, cache, slots, compress_ratio
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
        latent, positions, cos_sin, ours, slots, compress_ratio
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
        latent, positions, cos_sin, cache, slots, compress_ratio
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


# The four records this platform stores, and nothing else. Anything outside
# the table has no reader here, so the ops reject it rather than writing bytes
# that would later be decoded as something they are not.
_MAIN_RECORDS = [(PACKED_ROW_BYTES, torch.uint8), (LATENT_DIM, torch.bfloat16)]
_INDEX_RECORDS = ["fp8", "int8", "bf16"]


def _indexer_record_row(record: str) -> tuple[int, torch.dtype]:
    """Row width and dtype of an index-key record. fp8 and int8 share both."""
    if record == "bf16":
        return INDEX_HEAD_DIM, torch.bfloat16
    return INDEX_HEAD_DIM + 4, torch.uint8


@pytest.mark.parametrize("width,dtype", _MAIN_RECORDS)
def test_the_main_record_geometries_round_trip(width: int, dtype: torch.dtype):
    """fp8 is 584 uint8 and bf16 is 512 bf16, on both the write and the move."""
    torch.manual_seed(13)
    num_tokens, rows_per_page, compress_ratio = 32, 8, 1
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=13
    )
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(num_pages, rows_per_page, width, device=DEVICE, dtype=dtype)
    rows = torch.zeros(num_tokens, width, device=DEVICE, dtype=dtype)
    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, cache, slots, compress_ratio, rows
    )
    # Guard against a vacuous round trip: an op that wrote nothing at all
    # would replay nothing and still compare equal.
    assert bool((cache.view(torch.uint8) != 0).any())
    replayed = torch.zeros_like(cache)
    torch.ops._C.v41_move_rows_to_paged_cache(
        rows, replayed, slots, positions, compress_ratio
    )
    assert torch.equal(replayed, cache)


def test_the_main_store_rejects_a_record_it_has_no_reader_for():
    """Per-tensor fp8 rows are official's third option and unread here."""
    positions = torch.zeros(4, device=DEVICE, dtype=torch.int64)
    slots = torch.arange(4, device=DEVICE, dtype=torch.int64)
    cos_sin = _make_cos_sin(8)
    latent = torch.randn(4, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(1, 4, LATENT_DIM, device=DEVICE, dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="packed uint8 record or a plain bf16"):
        torch.ops._C.v41_rope_insert(latent, positions, cos_sin, cache, slots, 1)


@pytest.mark.parametrize("record", _INDEX_RECORDS)
def test_the_index_record_geometries_round_trip(record: str):
    """fp8/int8 are 132 uint8 and bf16 is 128 bf16, on the write and the move."""
    torch.manual_seed(14)
    num_tokens, rows_per_page, eps, compress_ratio = 32, 8, 1e-6, 1
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=14
    )
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    width, dtype = _indexer_record_row(record)
    cache = torch.zeros(num_pages, rows_per_page, width, device=DEVICE, dtype=dtype)
    rows = torch.zeros(num_tokens, width, device=DEVICE, dtype=dtype)
    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        cache,
        slots,
        compress_ratio,
        record,
        rows,
    )
    replayed = torch.zeros_like(cache)
    torch.ops._C.v41_move_rows_to_paged_cache(
        rows, replayed, slots, positions, compress_ratio
    )
    assert torch.equal(replayed, cache)


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
    round_bf16: bool = True,
) -> torch.Tensor:
    """k_norm then RoPE.

    The bf16 and fp8 references round back through bf16 after the norm and
    again after the rotation. ixformer's int8 reference does neither: it
    carries fp32 from the norm straight into the quantizer, so that record
    passes ``round_bf16=False``.
    """
    maybe_round = _bf16_round if round_bf16 else (lambda x: x)
    k = k_pre.float()
    variance = k.pow(2).mean(-1, keepdim=True)
    k = maybe_round(k * torch.rsqrt(variance + eps) * weight.float())
    rope_pos = (positions // compress_ratio) * compress_ratio
    return maybe_round(_rope_rows(k, rope_pos, cos_sin))


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
        "bf16",
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
        "fp8",
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


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_indexer_k_store_int8(compress_ratio: int):
    """The scheme-A key record: 128 int8 bytes and one ``amax / 127`` scale.

    Same 132-byte page as fp8, so only the quantizer is under test here. The
    bytes are compared as integers against the reference's own rounding, which
    pins the scale and the round-nearest together; a tolerance on the
    dequantized value would pass even if the scale were off by a step.
    """
    torch.manual_seed(6)
    num_tokens, rows_per_page, eps = 64, 8, 1e-6
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=6
    )
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(
        num_pages, rows_per_page, INDEX_HEAD_DIM + 4, device=DEVICE, dtype=torch.uint8
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
        "int8",
    )

    expected = _ref_indexer_k(
        k_pre, positions, cos_sin, weight, eps, compress_ratio, round_bf16=False
    )
    absmax = expected.abs().amax(-1).clamp(min=1e-10)
    exp_scale = absmax / INT8_MAX
    exp_bytes = (
        (expected * (INT8_MAX / absmax).unsqueeze(-1))
        .round()
        .clamp(-INT8_MAX, INT8_MAX)
    )

    flat = cache.reshape(num_pages, -1)
    ok = _publishes(positions, slots, compress_ratio)
    assert bool(ok.any())
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        page, row = slot // rows_per_page, slot % rows_per_page
        base = rows_per_page * INDEX_HEAD_DIM + row * 4
        got_scale = flat[page, base : base + 4].view(torch.float32)
        torch.testing.assert_close(got_scale[0], exp_scale[t], rtol=1e-6, atol=0)

        values = (
            flat[page, row * INDEX_HEAD_DIM : (row + 1) * INDEX_HEAD_DIM]
            .view(torch.int8)
            .float()
        )
        assert (values - exp_bytes[t]).abs().max() <= 1


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


def test_indexer_q_rope_int8():
    """ixformer's scheme A: one ``amax / 127`` per row, folded into the weights.

    The quantized bytes are compared as integers with a one-step slack rather
    than through a tolerance on the dequantized value. A step here is
    ``amax / 127`` wide, so a tolerance loose enough to absorb the rotation's
    last-bit disagreement would also absorb a wrong byte; "off by at most one
    level, and only where the product sits near a rounding boundary" is the
    actual claim.
    """
    torch.manual_seed(9)
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

    q_int8, weights_out = torch.ops._C.v41_indexer_q_rope_int8(
        q, positions, cos_sin, weights, scale
    )
    assert q_int8.dtype == torch.int8
    assert weights_out.dtype == torch.float32

    rows = q.float().reshape(-1, INDEX_HEAD_DIM)
    rope_pos = positions.repeat_interleave(num_heads)
    # No bf16 round-trip after the rotation: unlike the fp8 path, the reference
    # carries fp32 from the rotation straight into the quantizer.
    rotated = _rope_rows(rows, rope_pos, cos_sin)
    amax = rotated.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    q_scale = amax / INT8_MAX

    expected = (rotated * (INT8_MAX / amax)).round().clamp(-INT8_MAX, INT8_MAX)
    got = q_int8.float().reshape(-1, INDEX_HEAD_DIM)
    assert (got - expected).abs().max() <= 1

    torch.testing.assert_close(
        weights_out,
        weights.float() * q_scale.reshape(num_tokens, num_heads) * scale,
        rtol=1e-4,
        atol=1e-8,
    )


def test_indexer_q_rope_int8_saturates_without_overflowing():
    """A row whose amax lands on the clamp must reach 127, never wrap to -128.

    The kernel rounds into an int and clamps afterwards, so this is the case
    that would have shown a clamp applied in the wrong order.
    """
    cos_sin = _make_cos_sin(8)
    positions = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    q = torch.full((1, 1, INDEX_HEAD_DIM), 2.0, device=DEVICE, dtype=torch.bfloat16)
    q[0, 0, 0] = -2.0
    weights = torch.ones(1, 1, device=DEVICE, dtype=torch.bfloat16)

    q_int8, _ = torch.ops._C.v41_indexer_q_rope_int8(
        q, positions, cos_sin, weights, 1.0
    )

    # Position 0 rotates by the identity, so every lane holds +-amax exactly.
    assert int(q_int8[0, 0, 0]) == -127
    assert q_int8[0, 0, 1:].min().item() == 127


def test_indexer_q_rope_int8_zero_row_does_not_divide_by_zero():
    """An all-zero query row: the eps floor has to keep the scale finite."""
    cos_sin = _make_cos_sin(8)
    positions = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    q = torch.zeros(2, 4, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.ones(2, 4, device=DEVICE, dtype=torch.bfloat16)

    q_int8, weights_out = torch.ops._C.v41_indexer_q_rope_int8(
        q, positions, cos_sin, weights, 1.0
    )

    assert int(q_int8.abs().max()) == 0
    assert torch.isfinite(weights_out).all()


def test_indexer_q_rope_int8_empty_batch():
    cos_sin = _make_cos_sin(8)
    q = torch.empty(0, 4, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.empty(0, 4, device=DEVICE, dtype=torch.bfloat16)
    positions = torch.empty(0, dtype=torch.int64, device=DEVICE)

    q_int8, weights_out = torch.ops._C.v41_indexer_q_rope_int8(
        q, positions, cos_sin, weights, 1.0
    )

    assert q_int8.shape == (0, 4, INDEX_HEAD_DIM)
    assert q_int8.dtype == torch.int8
    assert weights_out.shape == (0, 4)


# ---------------------------------------------------------------------------
# Pipeline transfer: the dense mirror, and replaying it into another cache
# ---------------------------------------------------------------------------


def _packed_record_from_page(
    cache: torch.Tensor,
    slot: int,
    rows_per_page: int,
    value_bytes: int,
    scale_bytes: int,
) -> torch.Tensor:
    """The contiguous [values | scales] record a segregated page holds."""
    page, row = slot // rows_per_page, slot % rows_per_page
    flat = cache.reshape(cache.size(0), -1)
    values = flat[page, row * value_bytes : (row + 1) * value_bytes]
    sbase = rows_per_page * value_bytes + row * scale_bytes
    return torch.cat([values, flat[page, sbase : sbase + scale_bytes]])


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_rope_insert_row_out_mirrors_the_page(compress_ratio: int):
    """What crosses the pipeline must be what the page holds, byte for byte.

    The receiving rank replays these bytes instead of recomputing, so any
    divergence here would put a different record on each side of the cut.
    """
    torch.manual_seed(9)
    num_tokens, rows_per_page = 48, 8
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=9
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
    rows = torch.zeros(num_tokens, PACKED_ROW_BYTES, device=DEVICE, dtype=torch.uint8)

    torch.ops._C.v41_rope_insert(
        latent, positions, cos_sin, cache, slots, compress_ratio, rows
    )

    ok = _publishes(positions, slots, compress_ratio)
    assert bool(ok.any())
    for t in torch.nonzero(ok).flatten().tolist():
        expected = _packed_record_from_page(
            cache,
            int(slots[t]),
            rows_per_page,
            PACKED_DATA_BYTES,
            PACKED_SCALE_BYTES,
        )
        assert torch.equal(rows[t], expected)
    # A token that published nothing leaves its mirror row alone; the receiver
    # reapplies the same test, so the staleness is never read.
    for t in torch.nonzero(~ok).flatten().tolist():
        assert torch.all(rows[t] == 0)


@pytest.mark.parametrize("record", ["bf16", "fp8", "int8"])
def test_indexer_k_store_row_out_mirrors_the_page(record: str):
    torch.manual_seed(10)
    num_tokens, rows_per_page, eps, compress_ratio = 48, 8, 1e-6, 2
    positions, slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=10
    )
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    row_bytes, dtype = _indexer_record_row(record)
    cache = torch.zeros(num_pages, rows_per_page, row_bytes, device=DEVICE, dtype=dtype)
    rows = torch.zeros(num_tokens, row_bytes, device=DEVICE, dtype=dtype)

    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        cache,
        slots,
        compress_ratio,
        record,
        rows,
    )

    ok = _publishes(positions, slots, compress_ratio)
    assert bool(ok.any())
    for t in torch.nonzero(ok).flatten().tolist():
        slot = int(slots[t])
        if record == "bf16":
            expected = cache[slot // rows_per_page, slot % rows_per_page]
        else:
            expected = _packed_record_from_page(
                cache, slot, rows_per_page, INDEX_HEAD_DIM, 4
            )
        assert torch.equal(rows[t], expected)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("packed", [True, False])
def test_move_rows_replays_into_a_different_slot_mapping(
    compress_ratio: int, packed: bool
):
    """The receiving rank's cache must end up holding the sender's records.

    The two ranks page the same tokens differently, which is the whole reason
    the transfer is dense rows and a local slot mapping rather than a cache
    copy. Here the receiver's block table is shifted, so the slots genuinely
    differ and a kernel that ignored them would fail.
    """
    torch.manual_seed(11)
    num_tokens, rows_per_page = 48, 8
    positions, send_slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=11
    )
    # The receiver's pages for the same tokens, one page further along.
    recv_slots = torch.where(send_slots < 0, send_slots, send_slots + rows_per_page)
    recv_pages = num_pages + 1

    width = PACKED_ROW_BYTES if packed else LATENT_DIM
    dtype = torch.uint8 if packed else torch.bfloat16
    cos_sin = _make_cos_sin(128)
    latent = torch.randn(num_tokens, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    sender_cache = torch.zeros(
        num_pages, rows_per_page, width, device=DEVICE, dtype=dtype
    )
    rows = torch.zeros(num_tokens, width, device=DEVICE, dtype=dtype)
    torch.ops._C.v41_rope_insert(
        latent,
        positions,
        cos_sin,
        sender_cache,
        send_slots,
        compress_ratio,
        rows,
    )

    recv_cache = torch.zeros(
        recv_pages, rows_per_page, width, device=DEVICE, dtype=dtype
    )
    torch.ops._C.v41_move_rows_to_paged_cache(
        rows, recv_cache, recv_slots, positions, compress_ratio
    )

    ok = _publishes(positions, send_slots, compress_ratio)
    assert bool(ok.any())
    for t in torch.nonzero(ok).flatten().tolist():
        send_slot, recv_slot = int(send_slots[t]), int(recv_slots[t])
        if packed:
            sent = _packed_record_from_page(
                sender_cache,
                send_slot,
                rows_per_page,
                PACKED_DATA_BYTES,
                PACKED_SCALE_BYTES,
            )
            got = _packed_record_from_page(
                recv_cache,
                recv_slot,
                rows_per_page,
                PACKED_DATA_BYTES,
                PACKED_SCALE_BYTES,
            )
        else:
            sent = sender_cache[send_slot // rows_per_page, send_slot % rows_per_page]
            got = recv_cache[recv_slot // rows_per_page, recv_slot % rows_per_page]
        assert torch.equal(got, sent)

    # Slots the sender skipped stay untouched on the receiver, even though the
    # mirror row for those tokens holds whatever was there before. A packed
    # page is segregated, so "untouched" is about the record, not about the
    # 584-byte window a plain tensor index would give.
    written = {int(recv_slots[t]) for t in torch.nonzero(ok).flatten().tolist()}
    for slot in range(recv_pages * rows_per_page):
        if slot in written:
            continue
        if packed:
            record = _packed_record_from_page(
                recv_cache,
                slot,
                rows_per_page,
                PACKED_DATA_BYTES,
                PACKED_SCALE_BYTES,
            )
        else:
            record = recv_cache[slot // rows_per_page, slot % rows_per_page]
        assert torch.all(record == 0)


def test_move_rows_replays_the_indexer_record():
    """The indexer key takes the same route, at its own record width."""
    torch.manual_seed(12)
    num_tokens, rows_per_page, eps, compress_ratio = 32, 8, 1e-6, 1
    positions, send_slots, _, _, num_pages = _paged_layout(
        num_tokens, rows_per_page, compress_ratio, seed=12
    )
    recv_slots = torch.where(send_slots < 0, send_slots, send_slots + rows_per_page)
    cos_sin = _make_cos_sin(128)
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    sender_cache = torch.zeros(
        num_pages,
        rows_per_page,
        INDEX_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    rows = torch.zeros(num_tokens, INDEX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        sender_cache,
        send_slots,
        compress_ratio,
        "bf16",
        rows,
    )
    recv_cache = torch.zeros(
        num_pages + 1,
        rows_per_page,
        INDEX_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    torch.ops._C.v41_move_rows_to_paged_cache(
        rows, recv_cache, recv_slots, positions, compress_ratio
    )

    ok = _publishes(positions, send_slots, compress_ratio)
    for t in torch.nonzero(ok).flatten().tolist():
        send_slot, recv_slot = int(send_slots[t]), int(recv_slots[t])
        assert torch.equal(
            recv_cache[recv_slot // rows_per_page, recv_slot % rows_per_page],
            sender_cache[send_slot // rows_per_page, send_slot % rows_per_page],
        )


def test_empty_batches_are_noops():
    cos_sin = _make_cos_sin(8)
    empty_i64 = torch.empty(0, device=DEVICE, dtype=torch.int64)
    latent = torch.empty(0, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(1, 4, LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_rope_insert(latent, empty_i64, cos_sin, cache, empty_i64, 1)
    assert torch.all(cache == 0)

    kv_score = torch.empty(0, LATENT_DIM, device=DEVICE)
    weight = torch.ones(LATENT_DIM, device=DEVICE, dtype=torch.bfloat16)
    torch.ops._C.v41_compress_norm_cr1(kv_score, empty_i64, weight, 1e-6, latent)
