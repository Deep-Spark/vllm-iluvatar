# SPDX-License-Identifier: Apache-2.0
"""The BF16 sliding-window insert and the short-context top-k fill.

``v41_qkv_rope_insert`` is the hottest plugin kernel pair in a decode step --
it runs once per layer per step on both the query and the latent KV row -- and
``v41_fill_short_context_topk`` is the bypass the indexer takes when the whole
compressed context fits in ``topk``. Neither had a test, and both have since
been rewritten for width and occupancy, so what is pinned here is the contract
the rewrites had to preserve: which bytes move, which tokens are skipped, and
that a row published to the window cache is the same row left behind in ``kv``.
"""

from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401  (registers torch.ops._C)

DEVICE = "cuda"
ROW_DIM = 512
ROPE_DIM = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _make_cos_sin(max_pos: int) -> torch.Tensor:
    """[max_pos, ROPE_DIM] fp32, cos half then sin half."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    angles = torch.arange(max_pos).float()[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).to(DEVICE)


def _rope_tail(rows: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
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


def _build(num_tokens: int, num_heads: int, rows_per_page: int, seed: int = 0):
    torch.manual_seed(seed)
    num_pages = (num_tokens + rows_per_page - 1) // rows_per_page + 2
    q = torch.randn(num_tokens, num_heads, ROW_DIM, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, ROW_DIM, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.zeros(
        num_pages, rows_per_page, ROW_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    # Page 0 row 0 is never a real target here, and a quarter of the tokens
    # carry no slot at all, as CUDA-graph padding does.
    slots = torch.arange(
        rows_per_page, rows_per_page + num_tokens, dtype=torch.int64, device=DEVICE
    )
    slots[::4] = -1
    cos_sin = _make_cos_sin(num_tokens + 8)
    return q, kv, cache, positions, slots, cos_sin


@pytest.mark.parametrize("num_heads", [1, 16, 32])
def test_qkv_rope_insert_rotates_and_publishes(num_heads: int):
    num_tokens, rows_per_page = 37, 8
    q, kv, cache, positions, slots, cos_sin = _build(
        num_tokens, num_heads, rows_per_page
    )
    q_ref, kv_ref = q.clone(), kv.clone()

    torch.ops._C.v41_qkv_rope_insert(q, cache, kv, slots, positions, cos_sin)

    # Every head of every token is rotated, whether or not the token publishes.
    for h in range(num_heads):
        expected = _rope_tail(q_ref[:, h].float(), positions, cos_sin)
        torch.testing.assert_close(
            q[:, h].float(), expected.to(torch.bfloat16).float(), rtol=8e-3, atol=8e-3
        )

    expected_kv = _rope_tail(kv_ref.float(), positions, cos_sin).to(torch.bfloat16)
    torch.testing.assert_close(kv.float(), expected_kv.float(), rtol=8e-3, atol=8e-3)

    # A published row is exactly the row left behind in ``kv``, bit for bit:
    # the kernel writes what it rotated, it does not rotate twice.
    written = set()
    for t in range(num_tokens):
        slot = int(slots[t])
        if slot < 0:
            continue
        page, row = slot // rows_per_page, slot % rows_per_page
        written.add((page, row))
        assert torch.equal(cache[page, row], kv[t]), f"token {t} slot {slot}"
    for page in range(cache.size(0)):
        for row in range(rows_per_page):
            if (page, row) not in written:
                assert torch.all(cache[page, row] == 0)


def test_qkv_rope_insert_without_a_query():
    """The DSpark proposer publishes context KV with no query of its own."""
    num_tokens, rows_per_page = 20, 8
    q, kv, cache, positions, slots, cos_sin = _build(num_tokens, 0, rows_per_page)
    kv_ref = kv.clone()

    torch.ops._C.v41_qkv_rope_insert(q, cache, kv, slots, positions, cos_sin)

    expected = _rope_tail(kv_ref.float(), positions, cos_sin).to(torch.bfloat16)
    torch.testing.assert_close(kv.float(), expected.float(), rtol=8e-3, atol=8e-3)


def test_qkv_rope_insert_drops_slots_past_the_cache():
    """A slot naming a page the cache does not have publishes nothing."""
    num_tokens, rows_per_page = 8, 4
    q, kv, cache, positions, slots, cos_sin = _build(num_tokens, 2, rows_per_page)
    slots.fill_(cache.size(0) * rows_per_page + 1)

    torch.ops._C.v41_qkv_rope_insert(q, cache, kv, slots, positions, cos_sin)

    assert torch.all(cache == 0)


def test_qkv_rope_insert_empty_batch():
    q, kv, cache, positions, slots, cos_sin = _build(0, 4, 8)
    torch.ops._C.v41_qkv_rope_insert(q, cache, kv, slots, positions, cos_sin)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("topk", [16, 512])
def test_fill_short_context_topk(compress_ratio: int, topk: int):
    """Identity selection up to the causal group count, -1 past it."""
    num_tokens = 71
    idxs = torch.full((num_tokens, topk), 7, dtype=torch.int32, device=DEVICE)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    # Graph-capture padding carries a negative position and yields an empty row.
    positions[::9] = -1

    torch.ops._C.v41_fill_short_context_topk(idxs, lens, positions, compress_ratio)

    groups = torch.where(
        positions < 0,
        torch.zeros_like(positions),
        (positions + 1) // compress_ratio,
    )
    col = torch.arange(topk, device=DEVICE)
    expected = torch.where(col[None, :] < groups[:, None], col[None, :], -1)
    assert torch.equal(idxs, expected.to(torch.int32))
    assert torch.equal(lens, groups.clamp(max=topk).to(torch.int32))
