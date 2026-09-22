# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ixformer adapters next to MiniMax-M3 call sites."""

from __future__ import annotations

import inspect

import pytest
import torch

from vllm_iluvatar.models.minimax_m3.mxfp8 import indexer as indexer
from vllm_iluvatar.models.minimax_m3.mxfp8 import sparse_attention as sparse_attention
from vllm_iluvatar.models.minimax_m3.mxfp8.attention import (
    MiniMAXGemmaRMSNorm,
    fused_minimax_m3_qknorm_rope_kv_insert,
)


def test_local_model_uses_ixformer_ops_and_nhd_kv_layout():
    backend = sparse_attention.MiniMaxM3SparseBackend
    assert backend.get_kv_cache_shape(3, 128, 2, 128) == (3, 2, 128, 2, 128)
    assert backend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
    assert backend.supported_kv_cache_dtypes == ["bfloat16"]
    assert inspect.getsource(MiniMAXGemmaRMSNorm.forward).count("gemma_rms_norm") == 1


def test_fused_insert_is_not_on_custom_ops():
    """The ixformer fused insert is not a ``vllm._custom_ops`` override.

    Community ``models/minimax_m3/nvidia`` calls the community name against a
    packed KV layout; this adapter assumes the plugin's K/V-split NHD cache.
    """
    from vllm_iluvatar import _custom_ops as ilu_ops

    name = "fused_minimax_m3_qknorm_rope_kv_insert"
    assert name not in ilu_ops._ILU_OVERRIDES
    assert name not in ilu_ops._ILU_ADDED
    assert not hasattr(ilu_ops, name)
    assert fused_minimax_m3_qknorm_rope_kv_insert.__module__.endswith(
        "models.minimax_m3.mxfp8.attention"
    )


def test_local_impls_own_their_dispatch_without_patching_community():
    """The impls extend the abstract bases, not the community Triton impls.

    Borrowing a community ``forward`` would mean rebinding the kernel names in
    the community module, i.e. a process-wide patch of a module we don't own.
    """
    from vllm.models.minimax_m3.common import indexer as common_indexer
    from vllm.models.minimax_m3.common import sparse_attention as common_sparse
    from vllm.models.minimax_m3.common.ops import index_topk, sparse_attn

    for local_cls, base, community_cls in (
        (
            indexer.MiniMaxM3IndexerIxformerImpl,
            common_indexer.MiniMaxM3IndexerImpl,
            common_indexer.MiniMaxM3IndexerTritonImpl,
        ),
        (
            sparse_attention.MiniMaxM3SparseIxformerImpl,
            common_sparse.MiniMaxM3SparseImpl,
            common_sparse.MiniMaxM3SparseTritonImpl,
        ),
    ):
        assert issubclass(local_cls, base)
        assert not issubclass(local_cls, community_cls)
        assert local_cls.forward.__module__ == local_cls.__module__

    # Community kernel names still resolve to community ops.
    assert common_indexer.minimax_m3_index_score is index_topk.minimax_m3_index_score
    assert common_sparse.minimax_m3_sparse_attn is sparse_attn.minimax_m3_sparse_attn


def test_local_forwards_dispatch_to_the_local_adapters():
    for module, cls, names in (
        (
            indexer,
            indexer.MiniMaxM3IndexerIxformerImpl,
            (
                "minimax_m3_index_score",
                "minimax_m3_index_topk",
                "minimax_m3_index_decode",
            ),
        ),
        (
            sparse_attention,
            sparse_attention.MiniMaxM3SparseIxformerImpl,
            (
                "_minimax_m3_sparse_attn_ixformer",
                "minimax_m3_sparse_attn_decode",
            ),
        ),
    ):
        referenced = cls.forward.__code__.co_names
        for name in names:
            assert name in referenced
            assert getattr(module, name).__module__ == module.__name__


def test_adapter_signatures_track_community_contracts():
    from vllm.models.minimax_m3.common.ops import index_topk, sparse_attn

    for community_mod, local_mod, names in (
        (
            index_topk,
            indexer,
            (
                "minimax_m3_index_score",
                "minimax_m3_index_topk",
                "minimax_m3_index_decode",
            ),
        ),
        (
            sparse_attn,
            sparse_attention,
            ("minimax_m3_sparse_attn", "minimax_m3_sparse_attn_decode"),
        ),
    ):
        for name in names:
            assert inspect.signature(
                getattr(community_mod, name), eval_str=True
            ) == inspect.signature(getattr(local_mod, name), eval_str=True)


def test_quantized_kv_is_rejected_before_ixformer_dispatch():
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("torch has no FP8 dtype")
    with pytest.raises(NotImplementedError, match="requires KV in"):
        sparse_attention._check_unquantized_kv(
            torch.empty(1, dtype=fp8),
            None,
            None,
        )
    # The backend advertises BF16 only, so the kernel-level guard must agree
    # rather than quietly accepting FP16 as it used to.
    with pytest.raises(NotImplementedError, match="requires KV in"):
        sparse_attention._check_unquantized_kv(
            torch.empty(1, dtype=torch.float16),
            None,
            None,
        )
    sparse_attention._check_unquantized_kv(
        torch.empty(1, dtype=torch.bfloat16), None, None
    )


@pytest.mark.parametrize(
    ("adapter_name", "ixformer_name", "args"),
    [
        (
            "minimax_m3_index_decode",
            "minimax_m3_index_decode",
            (
                torch.empty(2, 3, 128),
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                128,
                4,
                1,
                1,
                3,
                1,
                1,
            ),
        ),
        (
            "minimax_m3_index_topk",
            "minimax_m3_index_topk",
            (
                torch.empty(3, 2, 16),
                torch.empty(1),
                torch.empty(1),
                2,
                4,
                1,
                1,
            ),
        ),
    ],
)
def test_indexer_stages_into_token_major_buffer(
    monkeypatch, adapter_name, ixformer_name, args
):
    """vLLM's persistent buffer is token-major [Qmax,H,K].

    ixformer writes a contiguous [Q,H,K] prefix in place; the padded suffix
    must stay untouched.
    """
    h, q, topk, qmax = 3, 2, 4, 6
    expected = torch.arange(q * h * topk, dtype=torch.int32).view(q, h, topk)
    token_major = torch.full((qmax, h, topk), -7, dtype=torch.int32)
    seen = {}

    def fake_ixformer(*call_args):
        seen["out"] = call_args[-1]
        call_args[-1].copy_(expected)
        return call_args[-1]

    monkeypatch.setattr(indexer.ixf_ops, ixformer_name, fake_ixformer)
    result = getattr(indexer, adapter_name)(*args, out=token_major)

    assert tuple(seen["out"].shape) == (q, h, topk)
    assert seen["out"].data_ptr() == token_major[:q].data_ptr()
    assert torch.equal(result, expected)
    assert torch.equal(token_major[:q], expected)
    assert torch.equal(
        token_major[q:], torch.full((qmax - q, h, topk), -7, dtype=torch.int32)
    )


@pytest.mark.parametrize(
    ("adapter_name", "ixformer_name", "extra_args"),
    [
        (
            "minimax_m3_sparse_attn",
            "minimax_m3_sparse_attn",
            (
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                1,
                1,
                1.0,
                torch.empty(1),
            ),
        ),
        (
            "minimax_m3_sparse_attn_decode",
            "minimax_m3_sparse_attn_decode",
            (
                torch.empty(1),
                torch.empty(1),
                1,
                1.0,
                torch.empty(1),
                1,
            ),
        ),
    ],
)
def test_sparse_attention_passes_token_major_topk(
    monkeypatch, adapter_name, ixformer_name, extra_args
):
    token_major = torch.arange(6 * 3 * 4, dtype=torch.int32).view(6, 3, 4)
    topk = token_major[:2]
    seen = {}

    def fake_ixformer(q, kv_cache, topk_idx, *args, **kwargs):
        seen["topk"] = topk_idx
        seen["backend"] = kwargs.get("backend")

    monkeypatch.setattr(sparse_attention.ixf_ops, ixformer_name, fake_ixformer)
    getattr(sparse_attention, adapter_name)(
        torch.empty(1), torch.empty(1, dtype=torch.bfloat16), topk, *extra_args
    )

    assert seen["topk"].is_contiguous()
    assert torch.equal(seen["topk"], topk)
    assert seen["backend"] == (
        "ixformer_perf" if adapter_name == "minimax_m3_sparse_attn" else None
    )


def test_sparse_impl_reserves_workspace_from_shared_manager(monkeypatch):
    impl = object.__new__(sparse_attention.MiniMaxM3SparseIxformerImpl)
    impl.num_kv_heads = 2
    impl.topk_blocks = 16
    seen = {}
    expected = torch.empty(123, dtype=torch.uint8)

    class FakeWorkspaceManager:
        def get_simultaneous(self, *specs):
            seen["specs"] = specs
            return (expected,)

    def fake_size(total_q, num_kv_heads, topk):
        seen["size_args"] = (total_q, num_kv_heads, topk)
        return 123

    monkeypatch.setattr(
        sparse_attention.ixf_ops,
        "minimax_m3_sparse_attn_tcu_workspace_size",
        fake_size,
    )
    monkeypatch.setattr(
        sparse_attention,
        "current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    got = impl.reserve_prefill_workspace(32768)
    assert got is expected
    assert seen["size_args"] == (32768, 2, 16)
    assert seen["specs"] == (((123,), torch.uint8),)


def test_indexer_impl_rejects_fp8_cache_and_non_max_score():
    kwargs = dict(
        num_kv_heads=1,
        scale=1.0,
        topk_blocks=4,
        sparse_block_size=128,
        num_index_heads=1,
        index_head_dim=128,
        prefix="x.attn",
    )
    with pytest.raises(NotImplementedError, match="indexer_kv_dtype"):
        indexer.MiniMaxM3IndexerIxformerImpl(**kwargs, indexer_kv_dtype="fp8")
    with pytest.raises(NotImplementedError, match="score_type"):
        indexer.MiniMaxM3IndexerIxformerImpl(**kwargs, score_type="sum")
