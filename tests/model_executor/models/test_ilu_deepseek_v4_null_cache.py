# SPDX-License-Identifier: Apache-2.0
"""V4 metadata + native state writer lifecycle on an aliased cache allocation."""

from types import SimpleNamespace

import ixformer.inference.functions as ixf
import pytest
import torch
from vllm.models.deepseek_v4.compressor import CompressorMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata


@pytest.mark.parametrize("dim,ratio,block", [(512, 4, 4), (128, 4, 4), (512, 128, 8)])
@torch.inference_mode()
def test_pad_state_writer_preserves_shared_null_through_replay(dim, ratio, block):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    width = dim * (2 if ratio == 4 else 1)
    stride_bytes = 206848
    backing = torch.zeros(3 * stride_bytes, dtype=torch.uint8, device="cuda")
    state = torch.as_strided(
        backing.view(torch.float32),
        (3, block, width * 2),
        (stride_bytes // 4, width * 2, 1),
    )
    fp8_view = torch.as_strided(
        backing.view(torch.float8_e4m3fn), (3, 64, 512), (stride_bytes, 512, 1)
    )
    kv = torch.full((4, width), 1.9921875, device="cuda")
    score = kv.clone()
    ape = torch.zeros(ratio, width, device="cuda")
    positions = torch.arange(4, dtype=torch.int64, device="cuda")
    slots = torch.full((4,), -1, dtype=torch.int64, device="cuda")
    tables = torch.zeros(2, 1, dtype=torch.int32, device="cuda")
    starts_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts_cpu.cuda(),
        query_start_loc_cpu=starts_cpu,
        seq_lens=torch.tensor([2, 2], dtype=torch.int32, device="cuda"),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=tables,
        slot_mapping=slots,
    )
    builder = object.__new__(CompressorMetadataBuilder)
    builder.block_size = block
    builder.token_to_req_indices = torch.zeros(4, dtype=torch.int32, device="cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )

    def write():
        metadata = builder.build(0, common)
        assert metadata.slot_mapping.data_ptr() == slots.data_ptr()
        ixf.save_partial_states(
            kv,
            score,
            ape,
            positions,
            metadata.slot_mapping,
            state,
            metadata.block_size,
            ratio,
        )

    write()
    assert not backing.count_nonzero(), "Dummy PAD must not write any shared bytes"
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        write()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    addresses = (state.data_ptr(), slots.data_ptr(), backing.data_ptr())
    # One live request + one dummy, then reversed, then all dummy and live again.
    for values in (
        [block, block + 1, -1, -1],
        [-1, -1, 2 * block, 2 * block + 1],
        [-1, -1, -1, -1],
        [block + 2, block + 3, -1, -1],
    ):
        slots.copy_(torch.tensor(values, device="cuda"))
        tables.copy_(
            torch.tensor(
                [[1 if values[0] >= 0 else 0], [2 if values[2] >= 0 else 0]],
                device="cuda",
            )
        )
        kv.add_(0.015625)
        previous = backing.clone()
        graph.replay()
        assert addresses == (state.data_ptr(), slots.data_ptr(), backing.data_ptr())
        assert not backing[:stride_bytes].count_nonzero(), "Null page changed"
        expected = previous.view(torch.float32).clone()
        expected_state = torch.as_strided(expected, state.shape, state.stride())
        for token, slot in enumerate(values):
            if slot >= 0:
                page, row = divmod(slot, block)
                expected_state[page, row, :width] = kv[token]
                expected_state[page, row, width:] = score[token]
        assert torch.equal(backing, expected.view(torch.uint8)), (
            "Unexpected writer bytes"
        )
    assert torch.isfinite(state).all()
    # Finite FP32 owner bytes may look like FP8 NaNs: whole-view scans are invalid.
    assert torch.isnan(fp8_view[1].float()).any()
    # Negative control: writer itself accepts slot zero; metadata ownership matters.
    slots.fill_(-1)
    slots[0] = 0
    write()
    assert backing[:stride_bytes].count_nonzero()
