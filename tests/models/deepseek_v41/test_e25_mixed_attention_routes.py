"""Actual frozen prefill-as-decode branch counts for the E25 suffix."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention


@pytest.mark.parametrize("old_tokens", [0, 2, 10])
def test_e25_suffix_uses_separate_prefill_and_old_decode_calls(old_tokens, monkeypatch):
    new_tokens = 128
    tokens = old_tokens + new_tokens
    owner = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(owner)
    owner.kv_cache = torch.empty(3, 64, 512, dtype=torch.bfloat16)
    index_sources = (20, 24, 28, 32, 36)
    layers = []
    metadata = {
        "owner20": SimpleNamespace(
            block_size=64, block_table=torch.zeros(1, 1, dtype=torch.int32)
        )
    }
    calls, maps = [], []
    module = __import__(
        IluDeepseekV41Attention.__module__, fromlist=["get_forward_context"]
    )
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_OPT", False)
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_AS_DECODE", True)

    def mapper(
        topk,
        reqs,
        block_table,
        page_rows,
        valid,
        positions,
        ratio,
        slot_limit,
        pad_walk,
    ):
        maps.append(topk.shape[0])
        assert page_rows == 64 and ratio == 1 and not pad_walk
        assert reqs.numel() == valid.numel() == positions.numel() == topk.shape[0]
        return topk.clone(), torch.full(
            (topk.shape[0],), topk.shape[1], dtype=torch.int32
        )

    monkeypatch.setattr(torch.ops._C, "v41_global_topk_indices_and_lens", mapper)
    for layer_id in range(21, 40):
        layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio, layer.compressed_kv_format = 1, None
        layer.is_kv_source = layer.is_kv_forwarder = False
        layer.compressed_cache_prefix = "owner20"
        layer._static_forward_context = {"owner20": owner}
        layer.index_source_layer_id = max(
            source for source in index_sources if source <= layer_id
        )
        layer.topk_indices_buffer = torch.zeros(tokens, 4, dtype=torch.int32)
        prefix = f"swa{layer_id}"
        layer.swa_cache_layer = SimpleNamespace(prefix=prefix, kv_cache=torch.empty(0))
        metadata[prefix] = SimpleNamespace(
            num_decodes=bool(old_tokens),
            num_prefills=1,
            num_decode_tokens=old_tokens,
            num_prefill_tokens=new_tokens,
            decode_swa_indices=torch.zeros(old_tokens, 1, 2, dtype=torch.int32),
            decode_swa_lens=torch.full((old_tokens,), 2, dtype=torch.int32),
            prefill_swa_indices=torch.zeros(new_tokens, 1, 2, dtype=torch.int32),
            prefill_swa_lens=torch.full((new_tokens,), 2, dtype=torch.int32),
            is_valid_token=torch.ones(tokens, dtype=torch.bool),
            token_to_req_indices=torch.zeros(tokens, dtype=torch.int32),
        )

        def reader(
            q,
            swa_indices,
            swa_lens,
            cache,
            main_indices,
            main_lens,
            output,
            *,
            positions,
            prefill=False,
            layer_id=layer_id,
        ):
            calls.append((layer_id, "prefill" if prefill else "decode", q.shape[0]))
            assert cache is owner.kv_cache
            assert swa_indices.shape[0] == swa_lens.numel() == q.shape[0]
            assert main_indices.shape[0] == main_lens.numel() == q.shape[0]
            assert positions.shape[0] == q.shape[0]
            output.fill_(1 if prefill else 2)

        layer._sparse_mla = reader
        layers.append(layer)
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    q = torch.zeros(tokens, 1, 512, dtype=torch.bfloat16)
    positions = torch.arange(tokens)
    for layer in layers:
        output = torch.empty_like(q)
        layer.forward_mqa(q, torch.empty(0), positions, output)
        assert (output[:old_tokens] == 2).all()
        assert (output[old_tokens:] == 1).all()
    expected_order = []
    for layer_id in range(21, 40):
        expected_order.append((layer_id, "prefill", new_tokens))
        if old_tokens:
            expected_order.append((layer_id, "decode", old_tokens))
    assert calls == expected_order
    assert len(calls) == (38 if old_tokens else 19)
    assert maps == [call[2] for call in calls]
    assert Counter(layer.index_source_layer_id for layer in layers) == {
        20: 3,
        24: 4,
        28: 4,
        32: 4,
        36: 4,
    }
