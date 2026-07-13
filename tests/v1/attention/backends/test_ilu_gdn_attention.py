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
