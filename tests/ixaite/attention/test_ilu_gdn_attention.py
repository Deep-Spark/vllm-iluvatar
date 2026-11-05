# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GDN attention backend metadata."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from vllm_iluvatar.backends.ixaite.attention import gdn as ilu_gdn_attn

torch = importlib.import_module("torch")


# Check the registry actually consulted by the runner, including its Mamba table.
def test_selected_gdn_backend_creates_its_own_metadata(monkeypatch):
    from vllm.v1.attention.backends import registry

    from vllm_iluvatar.backends.ixaite.attention.register import (
        register_attention_backends,
    )

    monkeypatch.setattr(registry, "_ATTN_OVERRIDES", {})
    monkeypatch.setattr(registry, "_MAMBA_ATTN_OVERRIDES", {})
    register_attention_backends()
    backend = registry.MambaAttentionBackendEnum.GDN_ATTN.get_class()
    assert backend.get_builder_cls() is ilu_gdn_attn.IluGDNAttentionMetadataBuilder


# Keep the community name while publishing the independent state builder.
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
    builder.apc_all = False
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
        metadata.spec_conv_state_indices_tensor,
        torch.tensor([10, 30], dtype=torch.int32, device="cpu"),
    )
    assert metadata.spec_conv_state_indices_tensor.stride() == (1,)
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


# Densify singleton/multi-request columns into the existing graph buffer.
@pytest.mark.parametrize("batch_size", [1, 2])
def test_builder_packs_strided_block_table_column(monkeypatch, batch_size):
    table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)[
        :batch_size
    ]
    assert table[:, 0].stride() == (4,)
    monkeypatch.setattr(
        ilu_gdn_attn,
        "mamba_get_block_table_tensor",
        lambda *args, **kwargs: table,
    )
    monkeypatch.setattr(
        ilu_gdn_attn,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (batch_size, 0, batch_size, 0),
    )

    builder = object.__new__(ilu_gdn_attn.IluGDNAttentionMetadataBuilder)
    builder.apc_all = False
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

    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        compute_num_computed_tokens=lambda: torch.tensor([4, 5], dtype=torch.int32)[
            :batch_size
        ],
        block_table_tensor=object(),
        seq_lens=[5, 6][:batch_size],
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )
    idx = metadata.non_spec_state_indices_tensor
    assert idx is not None
    assert idx.is_contiguous()
    assert idx.dtype == torch.int32
    assert torch.equal(idx, torch.tensor([10, 20], dtype=torch.int32)[:batch_size])
    assert idx.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    cu = metadata.non_spec_query_start_loc
    assert cu is not None
    assert cu.data_ptr() == query_start_loc.data_ptr()


def test_all_mode_uses_actual_block_size_and_keeps_full_table(monkeypatch):
    """A shared completed source has separate destinations; partial continuations stay in their block."""
    monkeypatch.setattr(ilu_gdn_attn, "async_tensor_h2d", lambda tensor, device: tensor)
    monkeypatch.setattr(
        ilu_gdn_attn, "mamba_get_block_table_tensor", lambda table, *args: table
    )
    monkeypatch.setattr(
        ilu_gdn_attn, "split_decodes_and_prefills", lambda *args, **kw: (0, 3, 0, 19)
    )
    monkeypatch.setattr(
        ilu_gdn_attn,
        "compute_causal_conv1d_metadata",
        lambda *args, **kw: (None, None, None),
    )
    builder = object.__new__(ilu_gdn_attn.IluGDNAttentionMetadataBuilder)
    builder.apc_all = True
    builder.use_spec_decode = False
    builder.use_full_cuda_graph = False
    builder.kv_cache_spec = SimpleNamespace(block_size=256)
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="all")
    )
    table = torch.tensor([[3, 8, 9], [3, 10, 11], [4, 12, 13]], dtype=torch.int32)
    computed = torch.tensor([256, 256, 259], dtype=torch.int32)
    cu = torch.tensor([0, 7, 12, 19], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        compute_num_computed_tokens=lambda: computed,
        num_computed_tokens_cpu=computed,
        block_table_tensor=table,
        seq_lens=computed + cu[1:] - cu[:-1],
        num_reqs=3,
        num_actual_tokens=19,
    )
    result = builder.build(0, metadata)
    assert result.state_block_table is table
    assert result.cache_block_size == 256
    assert result.non_spec_state_indices_tensor.tolist() == [3, 3, 12]
    assert result.output_state_indices.tolist() == [8, 10, 12]
    assert result.block_idx_first_scheduled_token.tolist() == [1, 1, 1]
