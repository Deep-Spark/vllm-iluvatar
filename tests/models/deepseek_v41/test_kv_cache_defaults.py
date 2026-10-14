"""V4.1 uses plain FP8 by default and retains explicit KV choices."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig

from vllm_iluvatar.models.deepseek_v41.attention import (
    _resolve_dsv41_kv_cache_dtype,
    _resolve_swa_bounded_replay,
)


@pytest.mark.parametrize(
    "requested,expected,dtype",
    [
        ("auto", "fp8_e4m3", torch.float8_e4m3fn),
        ("fp8", "fp8_e4m3", torch.float8_e4m3fn),
        ("fp8_e4m3", "fp8_e4m3", torch.float8_e4m3fn),
        ("fp8_ds_mla", "fp8_ds_mla", torch.uint8),
        ("bfloat16", "bfloat16", torch.bfloat16),
    ],
)
def test_cache_choice(requested, expected, dtype):
    config = CacheConfig(cache_dtype=requested)
    assert _resolve_dsv41_kv_cache_dtype(config) == (expected, dtype)
    assert config.cache_dtype == expected
    assert _resolve_dsv41_kv_cache_dtype(config) == (expected, dtype)


@pytest.mark.parametrize(
    "enabled,v2,pcp,expected",
    [
        (True, True, 1, True),
        (False, True, 1, False),
        (True, False, 1, False),
        (True, True, 2, False),
    ],
)
def test_v41_replay_matches_community_capability_constraints(
    enabled, v2, pcp, expected
):
    config = CacheConfig()
    config.swa_bounded_replay = enabled
    vllm_config = SimpleNamespace(
        use_v2_model_runner=v2,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )
    assert _resolve_swa_bounded_replay(config, vllm_config) is expected


def test_v41_retains_community_model_state_contract():
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    from vllm_iluvatar.models.deepseek_v41.model import (
        IluDeepseekV41ForCausalLM,
    )

    assert IluDeepseekV41ForCausalLM.get_model_state_cls() is DeepseekV41ModelState


@pytest.mark.parametrize("enabled", [False, True])
def test_v41_swa_spec_excludes_replayed_window_from_prefix_lookup(enabled):
    from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
        IluDeepseekV4SWACache,
    )

    layer = object.__new__(IluDeepseekV4SWACache)
    torch.nn.Module.__init__(layer)
    layer.cache_config = CacheConfig(cache_dtype="fp8_e4m3")
    layer.bounded_replay = enabled
    layer._block_size = 32
    layer.head_dim = 512
    layer.window_size = 128
    layer.dtype = torch.float8_e4m3fn
    spec = layer.get_kv_cache_spec(SimpleNamespace())
    assert spec.prefix_cacheable is (not enabled)
    assert spec.prefix_replay_tokens == (128 if enabled else 0)


def test_v41_metadata_uses_formal_operators():
    from ixformer.inference.functions import deepseekv41 as ops

    from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
        IluDeepseekV41SparseSWAMetadataBuilder,
    )

    assert (
        IluDeepseekV41SparseSWAMetadataBuilder._compute_swa_decode_indices_and_lens
        is ops.dsv41_compute_swa_decode_indices_and_lens
    )
    assert (
        IluDeepseekV41SparseSWAMetadataBuilder._compute_swa_prefill_gather_lens
        is ops.dsv41_compute_swa_prefill_gather_lens
    )


def test_bf16_attention_uses_supported_native_backend(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41 import attention as attention_mod

    calls = []
    monkeypatch.setattr(
        attention_mod.ixfops,
        "dsv41_mla_with_kvcache",
        lambda **kwargs: calls.append(kwargs),
    )
    attn = object.__new__(attention_mod.IluDeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.kv_cache_dtype = "bfloat16"
    attn.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty(2, 64, 512, dtype=torch.bfloat16)
    )
    attn.scale = 512**-0.5
    attn.attn_sink = torch.zeros(2)
    q = torch.empty(1, 2, 512, dtype=torch.bfloat16)
    indices = torch.zeros(1, 4, dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    attn._sparse_mla(q, indices, lengths, None, None, None, torch.empty_like(q))
    assert len(calls) == 1
    assert calls[0]["backend"] == "ixformer_native"
