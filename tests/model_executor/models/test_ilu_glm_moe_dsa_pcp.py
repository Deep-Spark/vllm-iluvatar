# SPDX-License-Identifier: Apache-2.0
"""PCP cache writes must retain the runner's rank-major padded layout."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.ops import pcp

from vllm_iluvatar.models.glm_moe_dsa import attention as attn
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV4IndexerBackend,
)


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize(
    "rank_tokens,decodes",
    [
        ([[0, 1, 6], [2, 3, 4, 5]], 0),  # uneven DualChunkSwap
        ([[0, 1, 6], [0, 2, 3, 4]], 1),  # replicated decode + prefill
        ([[0], []], 0),  # empty rank must still gather
        ([[0], [0]], 1),  # decode-only must not gather
    ],
)
def test_pcp_mla_cache_matches_unpartitioned_tokens(
    monkeypatch, use_fp8, rank_tokens, decodes
):
    width, rope = 4, 2
    padded = max(map(len, rank_tokens))
    full_kv = torch.arange(7 * width, dtype=torch.float32).reshape(7, width) + 1
    full_pe = torch.arange(7 * rope, dtype=torch.float32).reshape(7, 1, rope) + 101
    local_kv, local_pe, slots = [], [], []
    for rank, tokens in enumerate(rank_tokens):
        n = len(tokens)
        kv = torch.zeros(padded, width)
        pe = torch.zeros(padded, 1, rope)
        kv[:n], pe[:n] = full_kv[tokens], full_pe[tokens]
        local_kv.append(kv)
        local_pe.append(pe)
        slots.extend(
            [-1 if rank > 0 and i < decodes else t for i, t in enumerate(tokens)]
        )
        slots.extend([-1] * (padded - n))
    slots = torch.tensor(slots)
    expected = torch.full((8, width + rope), -99.0)
    ids = sorted(set(sum(rank_tokens, [])))
    expected[ids] = torch.cat((full_kv[ids], full_pe[ids, 0]), -1)

    for rank, tokens in enumerate(rank_tokens):
        calls = []

        def gather(tensor, dim, rank=rank, calls=calls):
            family = (
                local_kv
                if tensor.shape[-1] == width
                else [p.flatten(1) for p in local_pe]
            )
            torch.testing.assert_close(tensor, family[rank][decodes:])
            calls.append(tensor.shape)
            return torch.cat([t[decodes:] for t in family], dim=dim)

        monkeypatch.setattr(
            pcp,
            "get_pcp_group",
            lambda: SimpleNamespace(world_size=2, all_gather=gather),
        )
        layer = attn.IluGlmMoeDsaSparseMLA.__new__(attn.IluGlmMoeDsaSparseMLA)
        torch.nn.Module.__init__(layer)
        layer.prefix = "attn"
        layer.num_heads = 1
        layer.qk_nope_head_dim = 2
        layer.qk_rope_head_dim = rope
        layer.kv_lora_rank = width
        layer.v_head_dim = width
        layer.kv_cache_dtype = "bfloat16"
        layer.use_fp8_attn = use_fp8
        layer._decode_prepare_enabled = False
        layer._fp8_kv_scale_frozen = False
        layer.use_rms_rope_fp8_kv_fusion = True
        layer._k_scale = torch.ones(1)
        layer.kv_cache = torch.full((1, 8, width + rope), -99.0)
        layer.rotary_emb = lambda positions, q, k: (q, k)
        layer._k_up_proj = lambda q: q.new_zeros((len(q), 1, width))
        layer._v_up_proj = lambda x, outputs=None: (
            x if outputs is None else outputs.copy_(x)
        )
        layer._maybe_init_fp8_kv_scale = lambda kv, slots: None
        layer._sparse_mla_kernel = lambda q, *args: q.new_zeros((len(q), 1, width))
        layer._sparse_mla_kernel_fp8 = lambda q, *args: q.new_zeros((len(q), 1, width))
        meta = SimpleNamespace(
            num_actual_tokens=len(tokens),
            slot_mapping=slots,
            pcp_world_size=2,
            num_decode_tokens=decodes,
            block_size=8,
        )
        monkeypatch.setattr(
            attn,
            "get_forward_context",
            lambda meta=meta: SimpleNamespace(attn_metadata={"attn": meta}),
        )

        def insert(kv, pe, cache, mapping, *args, **kwargs):
            assert len(kv) == len(mapping)
            mask = mapping >= 0
            cache.view(8, -1)[mapping[mask]] = torch.cat((kv[mask], pe[mask, 0]), -1)

        monkeypatch.setattr(attn.ops, "concat_and_cache_mla", insert)
        monkeypatch.setattr(attn.ixf_ops, "glm_dsa_fp8_kv_insert", insert)
        monkeypatch.setattr(attn.ixf_ops, "rms_norm", lambda x, *args: x)
        result = layer.forward(
            torch.zeros(padded, 1, 4),
            local_kv[rank],
            local_pe[rank][:, 0],
            torch.arange(padded),
            kv_rms_weight=torch.ones(width),
            kv_rms_eps=1e-6,
        )
        torch.testing.assert_close(layer.kv_cache[0], expected)
        assert len(calls) == (0 if decodes == padded else 2)
        assert result.shape == (padded, width)
        assert torch.count_nonzero(result[len(tokens) :]) == 0


def test_pcp_capability_is_scoped_to_glm_and_uncompressed_indexer():
    assert attn.IluGlmMoeDsaPCPBackend.supports_pcp()
    assert attn.IluDeepseekV32IndexerBackend.supports_pcp()
    assert not IluFlashMLASparseBackend.supports_pcp()
    assert not IluDeepseekV4IndexerBackend.supports_pcp()


@pytest.mark.parametrize("empty", [False, True])
def test_indexer_gathers_padding_and_skips_empty_queries(monkeypatch, empty):
    from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as mod
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadata,
    )

    local = torch.zeros(2, 128) if empty else torch.ones(2, 128)
    remote = torch.full((2, 128), 2.0)
    gathered = []

    def gather(tensor, dim):
        gathered.append(tensor)
        return torch.cat((tensor, remote), dim=dim)

    monkeypatch.setattr(
        pcp, "get_pcp_group", lambda: SimpleNamespace(world_size=2, all_gather=gather)
    )
    meta = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([8]),
        num_reqs=1,
        max_query_len=0 if empty else 2,
        max_seq_len=8,
        num_actual_tokens=0 if empty else 2,
        query_start_loc=torch.tensor([0, 0 if empty else 2]),
        slot_mapping=torch.tensor([-1, -1, 6, 7] if empty else [4, 5, 6, 7]),
        head_dim=128,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=0 if empty else 2,
        pcp_world_size=2,
        prefill=SimpleNamespace(
            chunks=[SimpleNamespace(max_context_len=8, token_start=0, token_end=0)]
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"indexer": meta}),
    )
    cache = torch.zeros(8, 128)

    def insert(k, cache, slots):
        assert k.shape[0] == slots.numel() == 4
        valid = slots >= 0
        cache[slots[valid]] = k[valid]

    monkeypatch.setattr(mod.ops, "indexer_k_cache", insert)
    result = mod.ilu_sparse_attn_indexer(
        local,
        "indexer",
        cache,
        local[:, None],
        local,
        torch.ones(2, 1),
        4,
        128,
        16,
        16,
        torch.empty(2, 4, dtype=torch.int32),
    )
    assert len(gathered) == 1
    torch.testing.assert_close(cache[6:8], remote)
    torch.testing.assert_close(cache[4:6], local)
    assert result.shape == (2, 4)
