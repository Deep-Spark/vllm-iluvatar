# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GDN attention backend metadata."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from vllm_iluvatar.v1.attention.backends import ilu_gdn_attn

torch = importlib.import_module("torch")


def test_ilu_gdn_backend_uses_community_enum_name():
    assert ilu_gdn_attn.IluGDNAttentionBackend.get_name() == "GDN_ATTN"
    assert ilu_gdn_attn.IluGDNAttentionBackend.is_ssm() is True
    assert (
        ilu_gdn_attn.IluGDNAttentionBackend.get_builder_cls()
        is ilu_gdn_attn.IluGDNAttentionMetadataBuilder
    )


def test_metadata_builder_creates_output_index_for_mixed_spec_decode(monkeypatch):
    monkeypatch.setattr(
        ilu_gdn_attn,
        "async_tensor_h2d",
        lambda tensor, device: tensor.to(device),
    )
    monkeypatch.setattr(
        ilu_gdn_attn,
        "mamba_get_block_table_tensor",
        lambda *args, **kwargs: torch.tensor(
            [[10, 11], [20, 21], [30, 31]], dtype=torch.int32, device="cpu"
        ),
    )
    monkeypatch.setattr(
        ilu_gdn_attn,
        "compute_causal_conv1d_metadata",
        lambda *args, **kwargs: ({"nums": "dict"}, "batch-ptr", "offset-ptr"),
    )

    builder = object.__new__(ilu_gdn_attn.IluGDNAttentionMetadataBuilder)
    builder.kv_cache_spec = object()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="default")
    )
    builder.use_spec_decode = True
    builder.num_spec = 1
    builder.use_full_cuda_graph = False
    builder.decode_cudagraph_max_bs = 0

    query_start_loc = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="cpu")
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        compute_num_computed_tokens=lambda: torch.tensor(
            [5, 0, 7], dtype=torch.int32, device="cpu"
        ),
        block_table_tensor=object(),
        seq_lens=[2, 1, 2],
        num_reqs=3,
        num_actual_tokens=5,
    )

    with torch.device("cpu"):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
            num_accepted_tokens=torch.tensor([2, 1, 2], dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([1, -1, 1], dtype=torch.int32),
        )

    assert isinstance(metadata, ilu_gdn_attn.IluGDNAttentionMetadata)
    assert metadata.num_prefills == 1
    assert metadata.num_decodes == 0
    assert metadata.num_spec_decodes == 2
    assert torch.equal(
        metadata.non_spec_token_indx, torch.tensor([2], dtype=torch.int32, device="cpu")
    )
    assert torch.equal(
        metadata.spec_token_indx,
        torch.tensor([0, 1, 3, 4], dtype=torch.int32, device="cpu"),
    )
    assert torch.equal(
        metadata.output_index,
        torch.tensor([1, 2, 0, 3, 4], dtype=torch.int32, device="cpu"),
    )
    assert metadata.non_spec_state_indices_tensor is not None
    assert metadata.non_spec_state_indices_tensor.is_contiguous()
    assert metadata.non_spec_state_indices_tensor.dtype == torch.int32
    assert torch.equal(
        metadata.spec_conv_state_indices,
        torch.tensor([10, 30], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.spec_token_starts,
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.non_spec_token_starts,
        torch.tensor([2], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.chunk_offsets,
        torch.tensor([0, 1], dtype=torch.int32, device="cpu"),
    )
    assert metadata.n_chunk_total == 1


def test_pack_int32_1d_column_view_uses_graph_buffer():
    table = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    col = table[:, 0]
    assert not col.is_contiguous()
    buf = torch.empty(8, dtype=torch.int32)
    packed = ilu_gdn_attn._pack_int32_1d(col, buf)
    assert packed is not None
    assert packed.is_contiguous()
    assert packed.dtype == torch.int32
    assert packed.data_ptr() == buf.data_ptr()
    assert torch.equal(packed, torch.tensor([0, 4, 8], dtype=torch.int32))


def test_pack_int32_1d_contiguous_is_noop():
    src = torch.tensor([0, 1, 3], dtype=torch.int32)
    buf = torch.empty(8, dtype=torch.int32)
    packed = ilu_gdn_attn._pack_int32_1d(src, buf)
    assert packed is not None
    assert packed.data_ptr() == src.data_ptr()


def test_pack_int32_1d_graph_padded_view_stays():
    """FULL graph writes a padded contiguous slice; packing must not shrink it."""
    buf = torch.empty(8, dtype=torch.int32)
    buf[:3].copy_(torch.tensor([7, 8, 9], dtype=torch.int32))
    buf[3:].fill_(0)
    padded = buf[:5]
    assert padded.is_contiguous()
    packed = ilu_gdn_attn._pack_int32_1d(padded, buf)
    assert packed is not None
    assert packed.data_ptr() == padded.data_ptr()
    assert packed.numel() == 5


def test_builder_packs_strided_block_table_column(monkeypatch):
    table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)
    assert not table[:, 0].is_contiguous()
    monkeypatch.setattr(
        ilu_gdn_attn,
        "mamba_get_block_table_tensor",
        lambda *args, **kwargs: table,
    )
    monkeypatch.setattr(
        ilu_gdn_attn,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (2, 0, 2, 0),
    )

    builder = object.__new__(ilu_gdn_attn.IluGDNAttentionMetadataBuilder)
    builder.kv_cache_spec = object()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="all")
    )
    builder.use_spec_decode = False
    builder.num_spec = 0
    builder.use_full_cuda_graph = False
    builder.decode_cudagraph_max_bs = 8
    builder.non_spec_state_indices_tensor = torch.empty(8, dtype=torch.int32)
    builder.non_spec_query_start_loc = torch.empty(9, dtype=torch.int32)

    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        compute_num_computed_tokens=lambda: torch.tensor([4, 5], dtype=torch.int32),
        block_table_tensor=object(),
        seq_lens=[5, 6],
        num_reqs=2,
        num_actual_tokens=2,
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )
    idx = metadata.non_spec_state_indices_tensor
    assert idx is not None
    assert idx.is_contiguous()
    assert idx.dtype == torch.int32
    assert torch.equal(idx, torch.tensor([10, 20], dtype=torch.int32))
    assert idx.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    cu = metadata.non_spec_query_start_loc
    assert cu is not None
    assert cu.data_ptr() == query_start_loc.data_ptr()
