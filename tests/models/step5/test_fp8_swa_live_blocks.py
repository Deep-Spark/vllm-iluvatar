# SPDX-License-Identifier: Apache-2.0
"""An FP8 SWA read must not load values from a retired physical page."""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions import flash_attn_unpad_with_kvcache_fp8
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVQuantMode, SlidingWindowSpec

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
    IluFlashAttentionMetadataBuilder,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_ivcore30(),
    reason="requires ivcore30 native FP8 attention; fallback is not graph-capturable",
)
@pytest.mark.parametrize("backend", ["ixattn", "ixinfer"])
@pytest.mark.parametrize(
    ("query_lengths", "base_k"),
    [
        pytest.param([4] * 32, 65539, id="mtp-decode"),
        pytest.param([4, 4, 257, 113], 65539, id="mixed-prefill-decode"),
        pytest.param([4] * 4, 32, id="short-context"),
        pytest.param([1, 1, 1, 0], 512, id="window-boundary-and-padding"),
        pytest.param([4] * 3, 262144, id="long-page-aligned-context"),
        pytest.param([513, 1024], 1024, id="prefill-longer-than-window"),
    ],
)
def test_fp8_swa_excludes_reused_bf16_pages(query_lengths, base_k, backend):
    device = torch.device("cuda")
    page_size, window_size, head_dim, heads = 16, 512, 192, 16
    max_k = 270400
    requests = len(query_lengths)
    max_q = max(query_lengths)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: heads,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: head_dim,
        ),
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=1, decode_context_parallel_size=1
        ),
        cache_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=128,
        ),
        attention_config=None,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=requests),
    )
    builder = IluFlashAttentionMetadataBuilder(
        SlidingWindowSpec(
            block_size=page_size,
            num_kv_heads=1,
            head_size=head_dim,
            head_size_v=head_dim,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            sliding_window=window_size,
        ),
        ["attention"],
        config,
        device,
    )
    assert not builder.supports_update_block_table
    rng = torch.Generator(device=device).manual_seed(49)
    dtype = torch.float8_e4m3fn
    per_request_pages = max(
        53, (window_size + max_q + 2 * page_size - 3) // page_size + 1
    )
    num_pages = 1 + requests * per_request_pages
    raw = torch.zeros(num_pages * 16384, dtype=torch.uint8, device=device)
    cache = raw.as_strided(
        (num_pages, 1, page_size, 2 * head_dim),
        (16384, page_size * 2 * head_dim, 2 * head_dim, 1),
    )
    cache.copy_(
        (torch.randn(cache.shape, device=device, generator=rng) * 0.3)
        .to(dtype)
        .view(torch.uint8)
    )
    cache[0].zero_()
    key, value = cache.view(dtype).split(head_dim, dim=-1)
    query = (
        torch.randn(sum(query_lengths), heads, head_dim, device=device, generator=rng)
        * 0.5
    ).to(dtype)
    output = torch.empty_like(query, dtype=torch.bfloat16)
    scale = torch.ones(1, device=device)
    table = torch.zeros(requests, max_k // page_size, dtype=torch.int32, device=device)
    q_lens_cpu = torch.tensor(query_lengths, dtype=torch.int32)
    starts_cpu = torch.cat(
        (q_lens_cpu.new_zeros(1), q_lens_cpu.cumsum(0, dtype=torch.int32))
    )
    q_lens, starts = q_lens_cpu.to(device), starts_cpu.to(device)
    rows = torch.arange(requests, device=device)
    lens = (base_k + rows * (page_size + 1)).int()
    phys_ids = 1 + torch.arange(
        requests * per_request_pages, device=device, dtype=torch.int32
    ).view(requests, per_request_pages)
    columns = torch.arange(per_request_pages, device=device)

    def remap():
        table.zero_()
        last = (lens.long() - 1) // page_size
        logical = last[:, None] - columns
        valid_rows, valid_cols = torch.where(logical >= 0)
        table[valid_rows, logical[valid_rows, valid_cols]] = phys_ids[
            valid_rows, valid_cols
        ]
        first_live = (lens - q_lens - window_size + 1).clamp_min(0) // page_size
        active = (first_live > 0) & (q_lens > 0)
        return table[rows[active], (first_live[active] - 1).long()].long()

    def flash(start, end, cache_table, cu_q, cu_k, max_q, max_k):
        flash_attn_unpad_with_kvcache_fp8(
            q=query[start:end],
            key_cache=key,
            value_cache=value,
            q_scale=scale,
            k_scale=scale,
            v_scale=scale,
            block_table=cache_table,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            is_causal=True,
            window_left=window_size - 1,
            window_right=0,
            atten_scale=head_dim**-0.5,
            out=output[start:end],
            output_dtype=output.dtype,
            backend=backend,
        )

    slot_mapping = starts.new_full((sum(query_lengths),), -1)

    def build():
        return builder.build(
            0,
            CommonAttentionMetadata(
                query_start_loc=starts,
                query_start_loc_cpu=starts_cpu,
                seq_lens=lens,
                num_reqs=requests,
                num_actual_tokens=sum(query_lengths),
                max_query_len=max_q,
                max_seq_len=max_k,
                block_table_tensor=table,
                slot_mapping=slot_mapping,
            ),
        )

    def forward(meta):
        if (decode := meta.decode) is not None:
            flash(
                0,
                meta.num_decode_tokens,
                decode.block_table,
                decode.cu_seqlens_q,
                decode.cu_seqlens_k,
                decode.max_query_len,
                decode.max_decode_seq_len,
            )
        if (prefill := meta.prefill) is not None:
            flash(
                meta.num_decode_tokens,
                None,
                prefill.block_table,
                prefill.query_start_loc,
                prefill.key_start_loc,
                prefill.max_query_len,
                prefill.max_seq_len or meta.max_seq_len,
            )

    # A recycled page can contain finite BF16 data whose low bytes represent
    # NaN in E4M3. The page is outside the window of every scheduled query.
    retired = remap()
    recycled_value = torch.full(
        (retired.numel(), page_size, head_dim // 2),
        0.99609375,
        dtype=torch.bfloat16,
        device=device,
    ).view(torch.uint8)
    flash(
        0,
        None,
        table,
        starts,
        torch.cat((starts[:1], lens.cumsum(0, dtype=torch.int32))),
        max_q,
        max_k,
    )
    expected = output.clone()
    cache[retired, 0, :, head_dim:] = recycled_value
    original_lens = lens.clone()
    original_table = table.clone()
    meta = build()
    torch.testing.assert_close(lens, original_lens)
    torch.testing.assert_close(table, original_table)
    torch.testing.assert_close(
        meta.key_start_loc,
        torch.cat((starts[:1], lens.cumsum(0, dtype=torch.int32))),
    )
    assert meta.slot_mapping.data_ptr() == slot_mapping.data_ptr()
    forward(meta)
    torch.testing.assert_close(output, expected, atol=0.002, rtol=0)
    if max_q <= 4:
        previous_compact_table = meta.decode.block_table.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(meta)
        cache[retired, 0, :, head_dim:] = 0
        # Cross a logical page boundary AND change physical IDs. Moving only
        # the logical tail can leave every compact ID unchanged in this fixture.
        lens.add_(page_size + 7)
        phys_ids = phys_ids.roll(7, dims=1)
        retired = remap()
        flash(
            0,
            None,
            table,
            starts,
            torch.cat((starts[:1], lens.cumsum(0, dtype=torch.int32))),
            max_q,
            max_k,
        )
        expected = output.clone()
        recycled_value = torch.full(
            (retired.numel(), page_size, head_dim // 2),
            0.99609375,
            dtype=torch.bfloat16,
            device=device,
        ).view(torch.uint8)
        cache[retired, 0, :, head_dim:] = recycled_value
        updated = build()
        assert (
            updated.decode.block_table.data_ptr() == meta.decode.block_table.data_ptr()
        )
        assert not torch.equal(updated.decode.block_table, previous_compact_table)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0)
