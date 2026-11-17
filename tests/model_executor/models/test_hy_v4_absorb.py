# SPDX-License-Identifier: Apache-2.0
"""HY4 all-sparse-MQA absorb routing, including graph padding and warmup."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import IluFp8Config
from vllm_iluvatar.models.hy_v4.attention import (
    HYV4MLAAttentionLayer,
    indexer_fused_quant_config,
)


@pytest.mark.parametrize("context_bound", [0, 85889])
def test_fp8_indexer_passes_prefill_cpu_context_bound(monkeypatch, context_bound):
    from vllm_iluvatar.models.hy_v4 import fp8_indexer as mod
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadata,
    )

    m, heads, dim = 21, 32, 128
    positions = torch.arange(m)
    chunk = SimpleNamespace(
        max_context_len=85909,
        token_start=0,
        token_end=m,
        cu_seqlens_q=torch.tensor([0, m], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 85909], dtype=torch.int32),
        block_table=torch.zeros(1, 1343, dtype=torch.int32),
        max_q_len=m,
        max_kv_len=85909,
        valid_lens=torch.arange(85889, 85910, dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(m, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(85889, 85910, dtype=torch.int32),
        context_lower_bound=context_bound,
    )
    metadata = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([85909], dtype=torch.int32),
        max_seq_len=85909,
        slot_mapping=torch.arange(m),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=m,
        prefill=SimpleNamespace(chunks=[chunk]),
    )
    indexer = SimpleNamespace(
        n_head=heads,
        head_dim=dim,
        rope_dim=64,
        softmax_scale=dim**-0.5,
        n_head_scale=heads**-0.5,
        wq_b=Mock(return_value=(torch.zeros(m, heads * dim), None)),
        wk_weights_proj=Mock(return_value=(torch.zeros(m, dim + heads), None)),
        topk_tokens=2048,
        topk_indices_buffer=torch.full((m, 2048), -1, dtype=torch.int32),
        k_cache=SimpleNamespace(
            prefix="test.indexer", kv_cache=torch.zeros(1, 64, 132)
        ),
        k_norm=SimpleNamespace(weight=torch.ones(dim), bias=torch.zeros(dim), eps=1e-6),
    )
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_INDEXER_K", "1")
    with (
        patch.object(
            mod, "_indexer_rope_table", return_value=(positions, torch.zeros(m, dim))
        ),
        patch.object(
            mod,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"test.indexer": metadata}),
        ),
        patch.object(
            mod.ixfops,
            "fused_indexer_q_rope_quant",
            return_value=(torch.zeros(m, heads, dim), torch.ones(m, heads)),
        ),
        patch.object(mod.ixfops, "hy4_indexer_k_norm_rope_fp8_insert"),
        patch.object(
            mod.ixfops,
            "dsa_indexer_mqa_logits_with_kvcache_fp8",
            return_value=torch.zeros(m, 85909),
        ) as score,
        patch.object(mod.ixfops, "dsa_update_topk_indices"),
    ):
        result = mod.hy4_fp8_indexer_forward(
            indexer, torch.zeros(m, 8), torch.zeros(m, 8), positions, None
        )
    assert result is indexer.topk_indices_buffer
    assert score.call_count == 1
    # The facade registry owns the production native default.
    assert "backend" not in score.call_args.kwargs
    assert score.call_args.kwargs["context_lower_bound"] == context_bound
    assert score.call_args.kwargs["valid_lens"] is chunk.valid_lens


@pytest.mark.parametrize("legacy_env", [None, "0", "1"])
@pytest.mark.parametrize(
    "sparse,dcp,pcp,force_mqa,expected",
    [
        (True, 1, False, True, True),
        (False, 1, False, True, False),
        (True, 2, False, True, False),
        (True, 1, True, True, False),
        (True, 1, False, False, False),
    ],
)
def test_k_up_selection_follows_execution_mode(
    monkeypatch, legacy_env, sparse, dcp, pcp, force_mqa, expected
):
    if legacy_env is None:
        monkeypatch.delenv("VLLM_ILU_HYV4_FP8_K_UP", raising=False)
    else:
        monkeypatch.setenv("VLLM_ILU_HYV4_FP8_K_UP", legacy_env)
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_KV_UP", "0")

    def init_base(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.impl = SimpleNamespace(is_sparse=sparse, dcp_world_size=dcp)
        self.use_pcp = pcp
        self._vllm_config = SimpleNamespace(
            attention_config=SimpleNamespace(sparse_mla_force_mqa=force_mqa)
        )

    with patch.object(MLAAttention, "__init__", init_base):
        layer = HYV4MLAAttentionLayer()
    assert layer.fp8_k_up is expected


def make_layer():
    layer = HYV4MLAAttentionLayer.__new__(HYV4MLAAttentionLayer)
    torch.nn.Module.__init__(layer)
    layer.fp8_k_up = True
    layer.qk_nope_head_dim = 6
    layer.qk_rope_head_dim = 2
    layer._k_up_proj = Mock(side_effect=lambda q: q[..., :4] * 2)
    layer.impl = SimpleNamespace(
        forward_mqa=Mock(side_effect=lambda q, cache, meta, self: (q[0] + 3, None))
    )
    layer._v_up_proj = Mock(side_effect=lambda x, out: out.copy_(x.flatten(1)))
    return layer


@pytest.mark.parametrize("n", [0, 1, 3])
def test_fp8_absorb_uses_only_real_rows_and_zeros_padding(n):
    layer = make_layer()
    q = torch.arange(5 * 2 * 8).view(5, 2, 8).float()
    output = torch.full((5, 8), float("nan"))
    meta = SimpleNamespace(num_actual_tokens=n)
    result = layer.forward_impl(q, None, None, None, meta, output)
    assert result is output
    torch.testing.assert_close(output[:n], (q[:n, :, :4] * 2 + 3).flatten(1))
    assert (output[n:] == 0).all()
    assert layer._k_up_proj.call_count == int(n > 0)
    if n:
        assert layer._k_up_proj.call_args.args[0].shape == (n, 2, 6)
        torch.testing.assert_close(
            layer.impl.forward_mqa.call_args.args[0][1], q[:n, :, 6:]
        )


def test_fp8_absorb_warmup_has_no_quant_or_cache_access():
    layer = make_layer()
    out = torch.full((4, 8), float("nan"))
    layer.forward_impl(torch.empty(4, 2, 8), None, None, None, None, out)
    assert (out == 0).all()
    layer._k_up_proj.assert_not_called()
    layer.impl.forward_mqa.assert_not_called()


def test_unsupported_fp8_absorb_mode_delegates_original_contract():
    layer = make_layer()
    layer.fp8_k_up = False
    args = [torch.empty(1)] * 6
    with patch.object(MLAAttention, "forward_impl", return_value="baseline") as base:
        assert layer.forward_impl(*args) == "baseline"
    assert base.call_args.args[:6] == tuple(args)


def test_fp8_absorb_rejects_output_quant_fusion():
    layer = make_layer()
    with pytest.raises(ValueError, match="unquantized"):
        layer.forward_impl(
            torch.empty(1),
            None,
            None,
            None,
            None,
            torch.empty(1),
            output_scale=torch.ones(1),
        )


def test_indexer_fused_dispatch_preserves_mixed_checkpoint_and_shared_config():
    prefix = "model.layers.78.self_attn.indexer"
    base = IluFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[prefix + ".weights_proj"],
        weight_block_size=[128, 128],
    )
    base.packed_modules_mapping = {"wk_weights_proj": ["wk", "weights_proj"]}
    fused = indexer_fused_quant_config(base, prefix)
    assert fused is not base
    assert base.ignored_layers == [prefix + ".weights_proj"]
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method, disposition = fused._get_original_method_and_disposition(
        layer, prefix + ".wk_weights_proj"
    )
    assert isinstance(method, UnquantizedLinearMethod) and disposition == "ignored"
    assert not fused._is_layer_skipped(prefix + ".wq_b")
    assert indexer_fused_quant_config(None, prefix) is None
