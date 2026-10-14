from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_iluvatar.models.deepseek_v41.engram import NgramHashState


def test_hash_cache_uses_bound_page_size():
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.swa_cache_module = SimpleNamespace(kv_cache=torch.empty(3, 128, 8))
    state.block_size = 16
    state._cache = None
    state._kv_cache_ref = None
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        assert state.ensure_cache()
        assert state.block_size == 128
        assert state._cache.shape == (384,)
        assert torch.all(state._cache == -1)
        old_cache = state._cache
        assert state.ensure_cache()
        assert state._cache is old_cache
        state.swa_cache_module.kv_cache = torch.empty(4, 64, 8)
        assert state.ensure_cache()
        assert state.block_size == 64
        assert state._cache.shape == (256,)
