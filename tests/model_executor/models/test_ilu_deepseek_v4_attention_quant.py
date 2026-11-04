# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)
from vllm_iluvatar.models.deepseek_v4 import attention
from vllm_iluvatar.models.deepseek_v4.attention import (
    IluDeepseekV4MultiHeadLatentAttentionWrapper,
    _can_fuse_qr_norm_quant,
    _can_merge_c4a_wq_b,
)


def _w8a8_scheme() -> IluCompressedTensorsW8A8Int8:
    return object.__new__(IluCompressedTensorsW8A8Int8)


def _wrapper(*, c4a: bool = False, main_decoder: bool = True, q_k: int = 1536):
    scheme = _w8a8_scheme()
    indexer = SimpleNamespace(wq_b=SimpleNamespace(scheme=scheme)) if c4a else None
    return SimpleNamespace(
        is_main_decoder_layer=main_decoder,
        q_norm=SimpleNamespace(weight=torch.empty(q_k, dtype=torch.bfloat16)),
        kv_norm=SimpleNamespace(weight=torch.empty(512, dtype=torch.bfloat16)),
        wq_b=SimpleNamespace(scheme=scheme),
        indexer=indexer,
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        _c4a_merged_wq_b_weight=(
            torch.empty(24576, 1536, dtype=torch.int8) if c4a else None
        ),
        _c4a_merged_wq_b_scale=(
            torch.empty(24576, 1, dtype=torch.float32) if c4a else None
        ),
    )


def _decode_metadata():
    return {"swa": SimpleNamespace(num_prefills=0)}


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize("q_k", [1024, 1536])
def test_qr_norm_quant_decode_guard_accepts_supported_batches(
    num_tokens: int,
    q_k: int,
):
    wrapper = _wrapper(q_k=q_k)
    qr = torch.empty(num_tokens, q_k, dtype=torch.bfloat16)
    kv = torch.empty(num_tokens, 512, dtype=torch.bfloat16)

    assert _can_fuse_qr_norm_quant(wrapper, qr, kv, _decode_metadata())


@pytest.mark.parametrize(
    ("num_tokens", "main_decoder", "num_prefills"),
    [(0, True, 0), (17, True, 0), (1, False, 0), (1, True, 1)],
)
def test_qr_norm_quant_decode_guard_rejects_fallback_cases(
    num_tokens: int,
    main_decoder: bool,
    num_prefills: int,
):
    wrapper = _wrapper(main_decoder=main_decoder)
    qr = torch.empty(num_tokens, 1536, dtype=torch.bfloat16)
    kv = torch.empty(num_tokens, 512, dtype=torch.bfloat16)
    metadata = {"swa": SimpleNamespace(num_prefills=num_prefills)}

    assert not _can_fuse_qr_norm_quant(wrapper, qr, kv, metadata)


def test_qr_norm_quant_decode_guard_rejects_unsupported_q_k():
    """The kernel only implements the two DeepSeek-V4 q_lora_rank widths."""
    wrapper = _wrapper(q_k=2048)
    qr = torch.empty(1, 2048, dtype=torch.bfloat16)
    kv = torch.empty(1, 512, dtype=torch.bfloat16)

    assert not _can_fuse_qr_norm_quant(wrapper, qr, kv, _decode_metadata())


def test_qr_norm_quant_c4a_requires_both_w8a8_consumers():
    wrapper = _wrapper(c4a=True)
    wrapper.indexer.wq_b.scheme = object()
    qr = torch.empty(1, 1536, dtype=torch.bfloat16)
    kv = torch.empty(1, 512, dtype=torch.bfloat16)

    assert not _can_fuse_qr_norm_quant(wrapper, qr, kv, _decode_metadata())


def test_c4a_merged_projection_guard_is_m1_only(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(attention, "_DS4_C4A_MERGED_WQ_B", True)
    wrapper = _wrapper(c4a=True)
    scale = torch.ones(1, 1, dtype=torch.float32)

    m1_input = (torch.empty(1, 1536, dtype=torch.int8), scale, torch.bfloat16)
    assert _can_merge_c4a_wq_b(wrapper, m1_input, _decode_metadata())

    m2_input = (
        torch.empty(2, 1536, dtype=torch.int8),
        torch.ones(2, 1, dtype=torch.float32),
        torch.bfloat16,
    )
    assert not _can_merge_c4a_wq_b(wrapper, m2_input, _decode_metadata())


def test_c4a_merged_weight_disabled_does_not_allocate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(attention, "_DS4_C4A_MERGED_WQ_B", False)
    wrapper = SimpleNamespace()
    wrapper._ensure_c4a_wq_b_merge = MethodType(
        IluDeepseekV4MultiHeadLatentAttentionWrapper._ensure_c4a_wq_b_merge,
        wrapper,
    )

    IluDeepseekV4MultiHeadLatentAttentionWrapper.process_weights_after_loading(
        wrapper,
        torch.bfloat16,
    )

    assert not hasattr(wrapper, "_c4a_merged_wq_b_weight")


def test_c4a_merged_weight_lifecycle_concatenates_once(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(attention, "_DS4_C4A_MERGED_WQ_B", True)
    wrapper = SimpleNamespace(
        is_main_decoder_layer=True,
        indexer=SimpleNamespace(
            wq_b=SimpleNamespace(
                weight=torch.full((8192, 1536), 2, dtype=torch.int8),
                weight_scale=torch.full((8192, 1), 4.0),
            )
        ),
        wq_b=SimpleNamespace(
            weight=torch.full((16384, 1536), 1, dtype=torch.int8),
            weight_scale=torch.full((16384, 1), 3.0),
        ),
        _c4a_merged_wq_b_weight=None,
        _c4a_merged_wq_b_scale=None,
    )
    wrapper._ensure_c4a_wq_b_merge = MethodType(
        IluDeepseekV4MultiHeadLatentAttentionWrapper._ensure_c4a_wq_b_merge,
        wrapper,
    )

    IluDeepseekV4MultiHeadLatentAttentionWrapper.process_weights_after_loading(
        wrapper,
        torch.bfloat16,
    )
    first_weight = wrapper._c4a_merged_wq_b_weight

    assert first_weight.shape == (24576, 1536)
    assert torch.equal(first_weight[:16384], wrapper.wq_b.weight)
    assert torch.equal(first_weight[16384:], wrapper.indexer.wq_b.weight)
    assert torch.equal(
        wrapper._c4a_merged_wq_b_scale[:16384],
        wrapper.wq_b.weight_scale,
    )
    assert torch.equal(
        wrapper._c4a_merged_wq_b_scale[16384:],
        wrapper.indexer.wq_b.weight_scale,
    )

    IluDeepseekV4MultiHeadLatentAttentionWrapper.process_weights_after_loading(
        wrapper,
        torch.bfloat16,
    )
    assert wrapper._c4a_merged_wq_b_weight is first_weight


@pytest.mark.parametrize(
    "first,second", [((True, False), (False, True)), ((True, True), (False, False))]
)
def test_sparse_inverse_rope_is_planned_for_each_slice_and_batch(
    first, second, monkeypatch
):
    owner = attention.IluDeepseekV4MLAAttention.__new__(
        attention.IluDeepseekV4MLAAttention
    )
    torch.nn.Module.__init__(owner)
    owner.prefix, owner.compress_ratio = "main", 1
    owner.kv_cache_dtype, owner.scale = "bfloat16", 0.1
    owner.attn_sink = torch.zeros(16)
    owner.kv_cache = torch.empty(0)
    owner.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    )
    md = SimpleNamespace(
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=2,
        num_prefill_tokens=3,
        decode_swa_indices=torch.zeros(2, 1, 128, dtype=torch.int32),
        decode_swa_lens=torch.full((2,), 128, dtype=torch.int32),
        prefill_swa_indices=torch.zeros(3, 1, 128, dtype=torch.int32),
        prefill_swa_lens=torch.full((3,), 128, dtype=torch.int32),
    )
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"swa": md}),
    )
    monkeypatch.setattr(attention.envs, "VLLM_ILU_DS4_PREFILL_AS_DECODE", True)
    decisions, planned, executed, rotations = [], [], [], []

    def plan(**kw):
        assert set(
            (
                "q",
                "k_cache",
                "indices",
                "topk_length",
                "sm_scale",
                "attn_sink",
                "extra_k_cache",
                "extra_indices",
                "extra_topk_length",
                "output",
                "positions",
                "cos_sin_cache",
            )
        ) <= set(kw)
        planned.append(kw["q"].shape[0])
        return "ixformer_perf", decisions.pop(0)

    def execute(**kw):
        executed.append((kw["q"].shape[0], kw["backend"], kw["positions"] is not None))
        assert (kw["positions"] is None) == (kw["cos_sin_cache"] is None)
        kw["output"].zero_()

    monkeypatch.setattr(attention.ixfops, "flash_mla_sparse_with_kvcache_plan", plan)
    monkeypatch.setattr(attention.ixfops, "flash_mla_sparse_with_kvcache", execute)
    monkeypatch.setattr(
        attention.ixfops,
        "ds4_inv_rope",
        lambda value, *a, **kw: rotations.append(value.shape[0]),
    )
    q = torch.zeros(5, 16, 512, dtype=torch.bfloat16)
    positions = torch.arange(5)
    trig = torch.zeros(8, 64)
    for choices in (first, second):
        decisions[:] = choices
        planned.clear()
        executed.clear()
        rotations.clear()
        owner.forward(
            q=q,
            kv=None,
            positions=positions,
            output=torch.empty_like(q),
            cos_sin_cache=trig,
        )
        assert planned == [3, 2]
        assert executed == [
            (3, "ixformer_perf", choices[0]),
            (2, "ixformer_perf", choices[1]),
        ]
        assert rotations == [n for n, fused in zip((3, 2), choices) if not fused]
