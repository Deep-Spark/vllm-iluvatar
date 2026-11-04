# SPDX-License-Identifier: Apache-2.0
"""Provider loading, frozen scales and allocator-owned FP8 cache metadata."""

import json
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear

from vllm_iluvatar.models.step5.fp8_attention import (
    Step5SparseFP8Backend,
    Step5SparseFP8MetadataBuilder,
    load_scales,
    sparse_fp8_enabled,
)
from vllm_iluvatar.models.step5.sparse_attention import (
    indexer_parallelism,
    validate_sparse_geometry,
)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_provider_weight_loader(tp, monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    torch.set_num_threads(4)
    with set_current_vllm_config(VllmConfig()):
        for rank in range(tp):
            monkeypatch.setattr(
                parallel_state,
                "_TP",
                SimpleNamespace(rank_in_group=rank, world_size=tp),
            )
            layout = indexer_parallelism(tp, rank)
            for rows, replicated in ((4096, False), (16, False), (256, True)):
                kwargs = {"disable_tp": True} if replicated else layout
                layer = ColumnParallelLinear(
                    32, rows, bias=False, params_dtype=torch.bfloat16, **kwargs
                )
                whole = (torch.arange(rows)[:, None] % 128).expand(
                    rows, 32
                ).bfloat16() / 128
                layer.weight.weight_loader(layer.weight, whole)
                n = rows if replicated else rows // layout["tp_size"]
                start = 0 if replicated else layout["tp_rank"] * n
                assert torch.equal(layer.weight, whole[start : start + n])
                assert layer.output_size_per_partition == n


@pytest.mark.parametrize(
    "tp,rank", [(0, 0), (3, 0), (6, 0), (4, 4), (4, -1), (True, 0), (4, 0.5), (None, 0)]
)
def test_bad_provider_geometry(tp, rank):
    with pytest.raises(ValueError):
        indexer_parallelism(tp, rank)


@pytest.mark.parametrize(
    "heads,groups,dim", [(64, 4, 192), (32, 2, 192), (16, 1, 192), (8, 1, 192)]
)
def test_supported_native_sparse_geometry(heads, groups, dim):
    validate_sparse_geometry(heads, groups, dim)


@pytest.mark.parametrize(
    "heads,groups,dim",
    [
        (4, 1, 192),
        (2, 1, 192),
        (0, 1, 192),
        (16, 0, 192),
        (16, 3, 192),
        (17, 2, 192),
        (16, 1, 128),
        (16.0, 1, 192),
    ],
)
def test_unsupported_native_sparse_geometry(heads, groups, dim):
    with pytest.raises(ValueError, match="per-rank"):
        validate_sparse_geometry(heads, groups, dim)


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"k": 1},
        {"k": 0, "v": 1},
        {"k": -1, "v": 1},
        {"k": True, "v": 1},
        {"k": float("nan"), "v": 1},
        {"k": 1, "v": float("inf")},
    ],
)
def test_invalid_frozen_scales(entry, tmp_path, monkeypatch):
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"layer.attn": entry}))
    monkeypatch.setenv("VLLM_STEP5_FP8_KV_SCALE_PATH", str(path))
    with pytest.raises(ValueError):
        load_scales("layer.attn")


def test_missing_scale_layer(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", raising=False)
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"layer.attn": {"k": 0.01, "v": 0.03}}))
    monkeypatch.setenv("VLLM_STEP5_FP8_KV_SCALE_PATH", str(path))
    assert load_scales("layer.attn") == (0.01, 0.03)
    with pytest.raises(ValueError):
        load_scales("other.attn")
    monkeypatch.delenv("VLLM_STEP5_FP8_KV_SCALE_PATH")
    assert load_scales("layer.attn") == (1.0, 1.0)


def test_explicit_missing_scale_file_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", str(tmp_path / "missing.json")
    )
    with pytest.raises(FileNotFoundError):
        load_scales("layer.attn")


def test_explicit_fp8_switch(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_SPARSE_FP8", "1")
    assert sparse_fp8_enabled()
    monkeypatch.setenv("VLLM_STEP5_SPARSE_FP8", "0")
    assert not sparse_fp8_enabled()
    monkeypatch.setenv("VLLM_STEP5_SPARSE_FP8", "auto")
    with pytest.raises(ValueError):
        sparse_fp8_enabled()


def test_real_cache_config_clone_preserves_global_dtype():
    from vllm.config import CacheConfig

    from vllm_iluvatar.models.step5.fp8_attention import sparse_fp8_cache_config

    global_config = CacheConfig(cache_dtype="auto", block_size=16)
    local = sparse_fp8_cache_config(global_config)
    assert global_config.cache_dtype == "auto"
    assert local.cache_dtype == "fp8" and local.block_size == 16
    with pytest.raises(ValueError):
        sparse_fp8_cache_config(CacheConfig(cache_dtype="fp8"))


def test_eager_table_bound_and_graph_full_context():
    assert Step5SparseFP8Backend.get_supported_kernel_block_sizes() == [16]
    spec = object()
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1600))
    builder = Step5SparseFP8MetadataBuilder(
        spec,
        ["layer.attn"],
        config,
        torch.device("cpu"),
    )
    assert builder.kv_cache_spec is spec and builder.vllm_config is config
    table = torch.empty((2, 100), dtype=torch.int32)
    meta = SimpleNamespace(
        query_start_loc=object(),
        slot_mapping=object(),
        block_table_tensor=table,
        max_seq_len=31,
        max_query_len=1,
        num_actual_tokens=2,
    )
    eager = builder.build(0, meta)
    assert eager.query_start_loc is meta.query_start_loc
    assert eager.slot_mapping is meta.slot_mapping
    assert eager.max_seq_len == 31 and eager.num_actual_tokens == 2
    assert eager.max_query_len == 1
    assert eager.block_table.shape == (2, 2)
    assert eager.block_table.stride() == table.stride()
    assert eager.block_table.data_ptr() == table.data_ptr()
    captured = builder.build_for_cudagraph_capture(meta)
    assert captured.block_table is table
    assert captured.max_seq_len == 1600


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_real_hybrid_allocator_keeps_fp8_page16(groups):
    from vllm.config import CacheConfig
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowSpec

    from vllm_iluvatar.models.step5.fp8_attention import configure_sparse_fp8_attention

    layer = SimpleNamespace(num_kv_heads=groups)
    configure_sparse_fp8_attention(layer, 16)
    config = SimpleNamespace(cache_config=CacheConfig(block_size=16))
    spec = layer.get_kv_cache_spec(config)
    specs = unify_kv_cache_spec_page_size(
        {
            "sparse": spec,
            "swa": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=groups,
                head_size=192,
                dtype=torch.bfloat16,
                sliding_window=512,
            ),
            "ring": CircularBufferSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=512,
                head_size_v=0,
                dtype=torch.bfloat16,
            ),
        }
    )
    assert specs["sparse"].block_size == 16
    assert specs["sparse"].dtype == torch.uint8
    assert len({s.page_size_bytes for s in specs.values()}) == 1
    config.cache_config.block_size = 32
    with pytest.raises(ValueError, match="block-size 16"):
        layer.get_kv_cache_spec(config)


@pytest.mark.parametrize("with_spec", [False, True])
@pytest.mark.parametrize("quant_type", ["", "KVINT8"])
def test_swa_kernel_block_query_matches_v031_allocator_contract(
    with_spec, quant_type, monkeypatch
):
    from vllm.model_executor.layers.attention.attention import (
        _largest_kernel_block_within,
    )
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
        IluFlashAttentionBackend,
    )

    monkeypatch.setenv("VLLM_ILU_FLASH_ATTN_QUANT_TYPE", quant_type)
    spec = (
        SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=192,
            dtype=torch.bfloat16,
            sliding_window=512,
        )
        if with_spec
        else None
    )
    # Both unquantized and INT8 paged FMHA use bounded kernel pages.
    # MultipleOf(16) would also advertise unsupported pages above 256.
    assert IluFlashAttentionBackend.get_supported_kernel_block_sizes(spec) == [
        16,
        32,
        64,
        128,
        256,
    ]
    # 1536 B/token * 16 tokens = 24576 B/page. The allocator must keep
    # the 16-token SWA kernel page, with or without the cache spec.
    assert (
        _largest_kernel_block_within(IluFlashAttentionBackend, 1536, 24576, 16, spec)
        == 16
    )


def test_head_wise_gate_reaches_existing_native_facade(monkeypatch):
    from vllm_iluvatar.models.step5 import model

    class Projection(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x):
            return self.value, None

    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return q

    layer = model.Step5Attention.__new__(model.Step5Attention)
    torch.nn.Module.__init__(layer)
    x = torch.randn(3, 8)
    qkv = torch.randn(3, 16)
    gates = torch.randn(3, 2)
    layer.qkv_proj = Projection(qkv)
    layer.q_size, layer.kv_size = 8, 4
    layer.q_norm = layer.k_norm = torch.nn.Identity()
    layer.head_dim, layer.num_heads = 4, 2
    layer.use_rope = layer.sparse_enabled = False
    layer.use_head_wise_attn_gate = True
    layer.g_proj = Projection(gates)
    layer.attn = Attention()
    layer.o_proj = lambda value: (value, None)
    calls = []

    def gate(input, other):
        calls.append((input.shape, other.shape))
        return input * other.sigmoid()

    monkeypatch.setattr(model, "step3_5_mul_sigmoid", gate)
    actual = layer(torch.arange(3), x)
    assert calls == [(torch.Size([3, 2, 4]), torch.Size([3, 2, 1]))]
    torch.testing.assert_close(
        actual,
        (qkv[:, :8].reshape(3, 2, 4) * gates.sigmoid().unsqueeze(-1)).reshape(3, 8),
    )


@pytest.mark.parametrize(
    "available,selected",
    [(True, "ixattn"), (False, "ixattn"), (True, "ixformer_native")],
)
def test_ixattn_runtime_requires_registered_backend(monkeypatch, available, selected):
    """Resolve only after the lazy facade import; never accept fallback."""
    from vllm_iluvatar.models.step5 import runtime_contract as contract

    registered = []

    class Functions:
        def __getattr__(self, name):
            assert name == "sparse_gqa_fp8"
            registered.append(name)
            return lambda: None

    def resolve(op, backend):
        assert registered == [op]
        assert backend == "ixattn"
        return selected

    monkeypatch.setenv("VLLM_STEP5_SPARSE_FP8", "1")
    monkeypatch.setenv("VLLM_STEP5_FP8_ATTENTION_BACKEND", "ixattn")
    monkeypatch.delenv("VLLM_STEP5_FP8_KV_SCALE_PATH", raising=False)
    monkeypatch.delenv("VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", raising=False)
    monkeypatch.delenv("VLLM_STEP5_FP8_CALIBRATION_DIR", raising=False)
    monkeypatch.setattr(contract, "_is_cuda", lambda c: True)
    monkeypatch.setattr(contract, "_cuda_arch", lambda: "ivcore30")
    infer = SimpleNamespace(
        sparse_gqa_fp8=lambda: None,
        sparse_gqa_fp8_ixattn_available=lambda: available,
    )
    monkeypatch.setattr(
        contract,
        "_native_runtime",
        lambda: (Functions(), infer, None, SimpleNamespace(resolve=resolve)),
    )
    monkeypatch.setattr(contract, "_native_requirements", lambda c, draft: ([], [], {}))
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(zero_centered=True)),
    )
    if available and selected == "ixattn":
        contract.validate_step5_runtime(config, draft=True)
    else:
        with pytest.raises(RuntimeError, match="missing Step5 ixattn"):
            contract.validate_step5_runtime(config, draft=True)


def test_sparse_opaque_op_fake_does_not_read_runtime_metadata(monkeypatch):
    from vllm_iluvatar.models.step5 import sparse_attention

    def forbidden():
        raise AssertionError("fake implementation must not read forward context")

    monkeypatch.setattr(sparse_attention, "get_forward_context", forbidden)
    tensors = [torch.empty(0, device="meta") for _ in range(12)]
    op = torch.ops.vllm.ilu_step5_sparse_attention.default
    assert op(*tensors, "layer.sparse_attention") is None
    mutations = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info and arg.alias_info.is_write
    ]
    assert mutations == ["output"]


def test_sparse_opaque_op_routes_to_context_layer(monkeypatch):
    from vllm_iluvatar.models.step5 import sparse_attention

    calls = []
    layer = SimpleNamespace(attention_impl=lambda *args: calls.append(args))
    monkeypatch.setattr(
        sparse_attention,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.sparse_attention": layer}),
    )
    tensors = tuple(torch.empty(0) for _ in range(12))
    assert (
        sparse_attention.ilu_step5_sparse_attention(*tensors, "layer.sparse_attention")
        is None
    )
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], tensors))


@pytest.mark.parametrize(
    "prefix", ["model.layers.0.self_attn", "model.layers.60.self_attn"]
)
@pytest.mark.parametrize("scales", [None, (0.01, 0.02)])
def test_sparse_context_key_and_scale_buffers(monkeypatch, tmp_path, prefix, scales):
    from vllm_iluvatar.models.step5 import sparse_attention

    monkeypatch.setenv("VLLM_STEP5_FP8_ATTENTION_BACKEND", "ixformer_native")
    monkeypatch.setattr(
        sparse_attention,
        "get_rope",
        lambda **kwargs: SimpleNamespace(cos_sin_cache=torch.empty(1, 32)),
    )
    monkeypatch.setattr(sparse_attention, "Step5CSA", lambda *args: torch.nn.Module())
    monkeypatch.delenv("VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", raising=False)
    monkeypatch.delenv("VLLM_STEP5_FP8_KV_SCALE_PATH", raising=False)
    if scales is not None:
        path = tmp_path / "scales.json"
        path.write_text(json.dumps({f"{prefix}.attn": dict(zip(("k", "v"), scales))}))
        monkeypatch.setenv("VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", str(path))
    context = {}
    config = SimpleNamespace(
        sparse_config={}, rms_norm_eps=1e-5, max_position_embeddings=16
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    layer = sparse_attention.Step5SparseAttention(vllm_config, config, prefix, {})
    assert layer.attention_layer_name == f"{prefix}.attn"
    assert context[f"{prefix}.sparse_attention"] is layer
    expected = scales if scales is not None else (1.0, 1.0)
    for name, value in zip(("fp8_k_scale", "fp8_v_scale"), expected):
        buffer = dict(layer.named_buffers())[name]
        assert buffer.dtype == torch.float32 and buffer.ndim == 0
        torch.testing.assert_close(buffer, torch.tensor(value, dtype=torch.float32))
        assert name not in layer.state_dict()
