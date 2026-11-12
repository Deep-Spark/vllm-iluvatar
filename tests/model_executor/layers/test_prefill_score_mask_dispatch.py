"""Producer masking is opt-in and confined to supported prefill chunks."""

from unittest.mock import Mock

import pytest
import torch

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as indexer


@pytest.mark.parametrize(
    "prefill,enabled,writer,heads,width,expected",
    [
        (True, True, False, 32, 24576, True),
        (False, True, False, 32, 24576, False),
        (True, False, False, 32, 24576, False),
        (True, True, True, 32, 24576, False),
        (True, True, False, 64, 24576, False),
        (True, True, False, 32, 16384, False),
        (True, True, False, 32, 32769, False),
    ],
)
def test_score_mask_dispatch(
    monkeypatch, prefill, enabled, writer, heads, width, expected
):
    ilu_envs.apply()
    monkeypatch.setenv("VLLM_ILU_V41_PREFILL_SCORE_MASK", "1" if enabled else "0")
    rows = 256
    fake_logits = torch.empty(rows, 1)
    masked = Mock(return_value=fake_logits)
    original = Mock(return_value=fake_logits)
    filtering, topk = Mock(), Mock()
    monkeypatch.setattr(indexer, "cuda_arch", "ivcore30")
    monkeypatch.setattr(
        indexer.ixfops,
        "dsa_indexer_mqa_logits_with_candidate_mask_fp8",
        masked,
        raising=False,
    )
    monkeypatch.setattr(indexer, "_mqa_logits_for_candidates", original)
    monkeypatch.setattr(indexer, "_two_level_filter", filtering)
    monkeypatch.setattr(
        indexer.ixfops, "dsa_update_topk_indices", topk, raising=False
    )
    indexer._indexer_logits_and_topk(
        candidate_blocks=torch.empty(rows, 2048, dtype=torch.int32),
        candidate_block_size=8,
        candidate_write=writer,
        q=torch.empty(rows, heads, 128, dtype=torch.float8_e4m3fn),
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, width], dtype=torch.int32),
        kv_cache=torch.empty(1, 128, 132, dtype=torch.uint8),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        weights=torch.empty(rows, heads),
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
        cu_seqlen_ke=torch.full((rows,), width, dtype=torch.int32),
        topk_tokens=512,
        topk_indices_buffer=torch.empty(rows, 512, dtype=torch.int32),
        max_q_len=rows,
        max_kv_len=width,
        max_context_len=width,
        valid_lens=torch.full((rows,), width, dtype=torch.int32),
        token_start=0,
        token_end=rows,
        is_prefill=prefill,
    )
    assert masked.call_count == int(expected)
    assert original.call_count == int(not expected)
    assert filtering.call_count == int(not expected)
    assert topk.call_count == 1
