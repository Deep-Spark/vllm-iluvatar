"""Window reads must not include discarded pre-replay pages."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention


@pytest.mark.parametrize("causal", [True, False])
def test_window_excludes_poisoned_pages_before_replay(causal):
    # The first page contains stale KV. The valid replay starts inside it;
    # even a present page is not evidence that its earlier rows may be read.
    layer = SimpleNamespace(
        window_size=128,
        _window_offsets=torch.arange(128),
        _draft_window_offsets=torch.arange(256),
        _window_idx=torch.empty(2, 256, dtype=torch.int64),
        _window_lens=torch.empty(2, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0]),
        is_valid_token=torch.tensor([True, False]),
        block_table=torch.tensor([[9, 10, 11, 12]]),
        block_size=32,
        replay_start=torch.tensor([17]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([22]),
        causal=causal,
    )
    indices, lengths = IluDeepseekV41Attention._window_indices(
        layer, metadata, 0, 2, torch.tensor([20, 21])
    )
    last = 20 if causal else 21
    assert indices[0, : last - 16].tolist() == list(
        range(9 * 32 + 17, 9 * 32 + last + 1)
    )
    assert lengths.tolist() == [last - 16, 0]
    assert torch.all(indices[0, last - 16 :] == -1)
    assert torch.all(indices[1] == -1)


def test_community_state_gathers_original_history_in_batch_order(monkeypatch):
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    config = MagicMock()
    config.model_config.enable_prompt_embeds = False
    config.model_config.uses_mrope = False
    config.model_config.is_multimodal_model = False
    config.scheduler_config.max_num_seqs = 8
    config.scheduler_config.max_num_batched_tokens = 64
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    state = DeepseekV41ModelState(
        config, SimpleNamespace(token_lookback_depth=3), None, torch.device("cuda")
    )
    batch = SimpleNamespace(idx_mapping=torch.tensor([2, 0], device="cuda"))
    history = torch.arange(60, dtype=torch.int32, device="cuda").view(3, 20)
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=history),
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([5, 0, 17], dtype=torch.int32, device="cuda")
        ),
    )
    result = state.prepare_inputs(batch, req_states)
    window = result["lookback_token_ids"]
    assert window is state.lookback_token_ids
    assert window[:2].tolist() == [[56, 55, 54], [4, 3, 2]]
    assert torch.all(window[2:] == -1)
    # Reusing the request slot with a short prompt must overwrite old history.
    req_states.num_computed_tokens.gpu[2] = 1
    state.prepare_inputs(batch, req_states)
    assert window[0].tolist() == [40, -1, -1]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_window_reference_reorder_and_reuse(causal, block_size):
    # Independent scalar reference checks the complete persistent outputs,
    # including padded tokens and stale tails after nonzero -> zero replay.
    count = 7
    layer = SimpleNamespace(
        window_size=128,
        _window_offsets=torch.arange(128),
        _draft_window_offsets=torch.arange(256),
        _window_idx=torch.full((count, 256), 123456, dtype=torch.int64),
        _window_lens=torch.full((count,), 123456, dtype=torch.int32),
    )
    table = torch.arange(1, 1 + 3 * 32).reshape(3, 32)
    req = torch.tensor([2, 2, 0, 0, 1, 1, 2])
    positions = torch.tensor([127, 128, 257, 258, 511, 512, 129])
    valid = torch.tensor([True, True, True, False, True, True, False])
    # Query rows are logically ordered 0,1,2; token gather order differs.
    qsl = torch.tensor([0, 2, 4, 7])
    seq = torch.tensor([259, 513, 130])
    for starts in ([257, 500, 117], [0, 0, 0], [258, 512, 128]):
        metadata = SimpleNamespace(
            token_to_req_indices=req,
            is_valid_token=valid,
            block_table=table,
            block_size=block_size,
            replay_start=torch.tensor(starts),
            query_start_loc=qsl,
            seq_lens=seq,
            causal=causal,
        )
        indices, lengths = IluDeepseekV41Attention._window_indices(
            layer, metadata, 0, count, positions
        )
        expected = torch.full_like(indices, -1)
        expected_lengths = torch.zeros_like(lengths)
        for row in range(count):
            r, pos = int(req[row]), int(positions[row])
            if causal:
                lo, hi, width = max(0, pos + 1 - 128), pos, 128
            else:
                lo = max(0, int(seq[r]) - int(qsl[r + 1] - qsl[r]) - 128)
                hi, width = int(seq[r]) - 1, 256
            lo = max(lo, starts[r])
            for column, logical in enumerate(range(lo, lo + width)):
                page_column = logical // block_size
                if valid[row] and logical <= hi and page_column < table.shape[1]:
                    page = int(table[r, page_column])
                    expected[row, column] = page * block_size + logical % block_size
                    expected_lengths[row] += 1
        assert torch.equal(indices, expected)
        assert torch.equal(lengths, expected_lengths)
        assert indices.data_ptr() == layer._window_idx.data_ptr()
        assert lengths.data_ptr() == layer._window_lens.data_ptr()
