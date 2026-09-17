# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GDN OOT chunk entry."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import vllm.envs as envs
import vllm_iluvatar._C  # noqa: F401
from conftest import COMPILED_CUDA_ARCH, requires_cuda

import vllm_iluvatar.ilu_register as ilu_register

PREFIX = "model.layers.0.linear_attn"
NUM_K_HEADS = 2
NUM_V_HEADS = 2
HEAD_K_DIM = 16
HEAD_V_DIM = 16
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_KERNEL = 4

supported_architectures = ["ivcore11"]
test_forward_core = pytest.mark.skipif(
    COMPILED_CUDA_ARCH not in supported_architectures,
    reason=f"ixinfer gdn kernel need {supported_architectures} build, got {COMPILED_CUDA_ARCH}",
)


def _install_fake_community_gdn(monkeypatch):
    chunk_registered = []
    attention_registered = []

    class FakeChunkGatedDeltaRule:
        def __init__(self) -> None:
            self.gdn_prefill_backend = "flashinfer"
            self._forward_method = self.forward_flashinfer

        @classmethod
        def register_oot(cls):
            def decorator(oot_cls):
                chunk_registered.append(oot_cls)
                return oot_cls

            return decorator

        def forward_flashinfer(self, *args, **kwargs):
            raise AssertionError("flashinfer path should not be used")

    class FakeQwenGatedDeltaNetAttention:
        @classmethod
        def register_oot(cls, oot_cls):
            attention_registered.append(oot_cls)
            return oot_cls

    module_name = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    fake_module = ModuleType(module_name)
    fake_module.ChunkGatedDeltaRule = FakeChunkGatedDeltaRule
    fake_module.QwenGatedDeltaNetAttention = FakeQwenGatedDeltaNetAttention
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return chunk_registered, attention_registered


def _install_fake_import_dependencies(monkeypatch):
    ixformer = ModuleType("ixformer")
    ixformer.__path__ = []
    ixformer_c = ModuleType("ixformer._C")
    ixformer_c.infer = object()
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer._C", ixformer_c)

    metadata_module = ModuleType("vllm_iluvatar.v1.attention.backends.ilu_gdn_attn")

    class FakeIluGDNAttentionMetadata:
        pass

    metadata_module.IluGDNAttentionMetadata = FakeIluGDNAttentionMetadata
    monkeypatch.setitem(sys.modules, metadata_module.__name__, metadata_module)


def _load_plugin_module(monkeypatch, *, gdn_backend: bool = False):
    chunk_registered, attention_registered = _install_fake_community_gdn(monkeypatch)
    _install_fake_import_dependencies(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_ILU_GDN_ATTN_BACKEND", gdn_backend, raising=False)
    module_name = (
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name), chunk_registered, attention_registered


def test_qwen_gdn_oot_module_is_auto_discovered():
    assert "vllm_iluvatar.model_executor.layers.mamba.gdn" in (
        ilu_register._OOT_OP_PACKAGES
    )


def test_ilu_chunk_gated_delta_rule_registers_and_forces_plugin_native(monkeypatch):
    module, chunk_registered, attention_registered = _load_plugin_module(monkeypatch)

    assert chunk_registered == [module.IluChunkGatedDeltaRule]
    assert attention_registered == []

    calls = []

    def fake_chunk_gated_delta_rule(**kwargs):
        calls.append(kwargs)
        return "plugin-output"

    fake_chunk_module = ModuleType(
        "vllm_iluvatar.third_party.flash_linear_attention.ops.chunk"
    )
    fake_chunk_module.chunk_gated_delta_rule = fake_chunk_gated_delta_rule
    monkeypatch.setitem(sys.modules, fake_chunk_module.__name__, fake_chunk_module)

    op = module.IluChunkGatedDeltaRule()

    assert op.gdn_prefill_backend == "triton"
    assert op._forward_method.__self__ is op
    assert op._forward_method.__func__ is op.forward_native.__func__
    assert (
        op.forward_native(
            q="q",
            k="k",
            v="v",
            g="g",
            beta="beta",
            initial_state="state",
            output_final_state=True,
            cu_seqlens="cu",
            chunk_indices="indices",
            chunk_offsets="offsets",
            use_qk_l2norm_in_kernel=False,
            core_attn_out="out",
        )
        == "plugin-output"
    )
    assert calls == [
        {
            "q": "q",
            "k": "k",
            "v": "v",
            "g": "g",
            "beta": "beta",
            "initial_state": "state",
            "output_final_state": True,
            "cu_seqlens": "cu",
            "chunk_indices": "indices",
            "chunk_offsets": "offsets",
            "use_qk_l2norm_in_kernel": False,
            "core_attn_out": "out",
        }
    ]


def test_qwen_gdn_attention_registers_only_with_plugin_backend(monkeypatch):
    module, chunk_registered, attention_registered = _load_plugin_module(
        monkeypatch, gdn_backend=True
    )

    assert chunk_registered == [module.IluChunkGatedDeltaRule]
    assert attention_registered == [module.IluQwenGatedDeltaNetAttention]


def test_rearrange_mixed_qkv_splits_qkv_projection(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    op = object.__new__(module.IluQwenGatedDeltaNetAttention)
    op.key_dim = 4
    op.value_dim = 2
    op.tp_size = 1
    op.head_k_dim = 2
    op.head_v_dim = 1

    mixed_qkv = torch.arange(30).reshape(3, 10)

    query, key, value = op.rearrange_mixed_qkv(mixed_qkv)

    assert query.shape == (3, 2, 2)
    assert key.shape == (3, 2, 2)
    assert value.shape == (3, 2, 1)
    assert torch.equal(query.reshape(3, 4), mixed_qkv[:, :4])
    assert torch.equal(key.reshape(3, 4), mixed_qkv[:, 4:8])
    assert torch.equal(value.reshape(3, 2), mixed_qkv[:, 8:])


def test_packed_qkv_views_matches_rearrange(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    op = object.__new__(module.IluQwenGatedDeltaNetAttention)
    op.key_dim = 4
    op.value_dim = 2
    op.tp_size = 1
    op.head_k_dim = 2
    op.head_v_dim = 1
    packed = torch.arange(30).reshape(3, 10).contiguous()
    q0, k0, v0 = op.rearrange_mixed_qkv(packed)
    q1, k1, v1 = op.packed_qkv_views(packed)
    assert torch.equal(q0, q1)
    assert torch.equal(k0, k1)
    assert torch.equal(v0, v1)
    assert q1.stride() == (10, 2, 1)
    assert not q1.is_contiguous()


def test_ixformer_core_uses_cuda_gating(monkeypatch):
    import inspect

    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    src = inspect.getsource(module.IluQwenGatedDeltaNetAttention._forward_core_ixformer)
    assert "fused_gdn_gating" not in src
    assert "qwen_gdn_gating" in src
    assert "_g_buf" in src and "_beta_buf" in src
    assert "self._conv_w" in src
    assert "weight.view" not in src
    assert "torch.ones" not in src
    assert "to(dtype=torch.uint8)" not in src
    assert "to(dtype=torch.int32).contiguous()" not in src
    assert "conv_cache.contiguous()" not in src
    assert "conv_cache.copy_" not in src


def test_mixed_spec_prefill_routes_to_ixformer(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    op = object.__new__(module.IluQwenGatedDeltaNetAttention)
    op.prefix = PREFIX
    op.enable_packed_recurrent_decode = False
    op._qwen38_kernel = lambda: "ixformer"
    op._ixformer_spec_enabled = lambda: True
    calls = []
    op._forward_core_ixformer = lambda **kwargs: calls.append(kwargs)
    metadata = module.IluGDNAttentionMetadata()
    metadata.spec_sequence_masks = torch.tensor([True, False])
    metadata.num_prefills = 1
    metadata.num_decodes = 0
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={PREFIX: metadata}
    )
    mixed_qkv = torch.empty(2, 8)
    gate = torch.empty(2, 1)
    out = torch.empty(2, 1, 1)

    op._forward_core(mixed_qkv, gate, gate, out)

    assert len(calls) == 1
    assert calls[0]["attn_metadata"] is metadata


def test_ixformer_forward_cuda_isolates_other_kernels(monkeypatch):
    import inspect

    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    src = inspect.getsource(module.IluQwenGatedDeltaNetAttention.forward_cuda)
    assert 'self._qwen38_kernel() != "ixformer"' in src
    assert "super().forward_cuda" in src
    assert "b = b.contiguous()" not in src
    assert "a = a.contiguous()" not in src
    assert "_core_attn_out" in src


def test_ixformer_output_projection_uses_cuda_rmsnorm(monkeypatch):
    import inspect

    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    src = inspect.getsource(module.IluQwenGatedDeltaNetAttention._output_projection)
    assert "_run_ixformer_rmsnorm_gated" in src
    assert "self.norm.weight" in src
    assert "self._gate_act_code" in src
    assert "reshape(-1" not in src
    assert 'self._qwen38_kernel() != "ixformer"' in src
    assert "super()._output_projection" in src


def test_ixformer_has_init_reuses_buffers(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    op = object.__new__(module.IluQwenGatedDeltaNetAttention)
    op._has_init_u8 = torch.empty(8, dtype=torch.uint8)
    op._has_init_i32 = torch.empty(8, dtype=torch.int32)
    src = torch.tensor([1, 0, 1], dtype=torch.int32)
    u8 = op._ixformer_has_init(src, 3, as_uint8=True)
    assert u8 is not None and u8.dtype == torch.uint8
    assert u8.data_ptr() == op._has_init_u8.data_ptr()
    assert torch.equal(u8, src.to(torch.uint8))
    i32 = op._ixformer_has_init(src, 3, as_uint8=False)
    assert i32 is not None and i32.data_ptr() == src.data_ptr()
    filled = op._ixformer_has_init(None, 4, as_uint8=False)
    assert filled is not None
    assert torch.equal(filled, torch.ones(4, dtype=torch.int32))
    assert filled.data_ptr() == op._has_init_i32.data_ptr()
    assert op._ixformer_has_init(None, 4, as_uint8=True) is None


def test_ixformer_workspace_is_packed_and_shared(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    module._shared_gdn_workspaces.clear()

    def make_op():
        op = object.__new__(module.IluQwenGatedDeltaNetAttention)
        op._gdn_workspace = None
        op._conv_out = None
        op._g_buf = None
        op._beta_buf = None
        op._core_attn_out = None
        op._conv_w = None
        op._has_init_u8 = None
        op._has_init_i32 = None
        op._gdn_workspace_group = 123
        op.num_v_heads = 12
        op.tp_size = 4
        op._gdn_b_max = 64
        op._gdn_t_max = 8192
        op.conv_dim = 24
        op.head_v_dim = 8
        op.conv1d = SimpleNamespace(weight=torch.empty(24, 1, 4))
        return op

    first = make_op()
    second = make_op()
    first._ensure_ixformer_buffers(torch.device("cpu"))
    second._ensure_ixformer_buffers(torch.device("cpu"))
    expected = ((8192 + 15) // 16 + 64) * 3 * 13856
    assert first._gdn_workspace is not None
    assert first._gdn_workspace.numel() == expected
    assert second._gdn_workspace is first._gdn_workspace


def _import_real_plugin_module():
    module_name = (
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _import_community_gdn_module():
    return importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )


def _build_gdn_layer(
    attention_cls,
    conv_state,
    ssm_state,
    a_log,
    dt_bias,
    conv_weight,
    *,
    chunk_gated_delta_rule=None,
):
    layer = SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = NUM_K_HEADS
    layer.num_v_heads = NUM_V_HEADS
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.head_k_dim = HEAD_K_DIM
    layer.head_v_dim = HEAD_V_DIM
    layer.activation = "silu"
    layer.A_log = a_log
    layer.dt_bias = dt_bias
    layer.conv1d = SimpleNamespace(weight=conv_weight, bias=None)
    layer.kv_cache = [conv_state, ssm_state]
    if chunk_gated_delta_rule is not None:
        layer.chunk_gated_delta_rule = chunk_gated_delta_rule
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            MethodType(getattr(attention_cls, name), layer),
        )
    return layer


def _run_forward_core(module, layer, metadata, mixed_qkv, b, a):
    core_attn_out = torch.zeros(
        mixed_qkv.shape[0],
        NUM_V_HEADS,
        HEAD_V_DIM,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    ctx = SimpleNamespace(attn_metadata={PREFIX: metadata})
    module.get_forward_context = lambda: ctx
    layer._forward_core(
        mixed_qkv=mixed_qkv.clone(),
        b=b.clone(),
        a=a.clone(),
        core_attn_out=core_attn_out,
    )
    return core_attn_out


def _identity_causal_conv1d_fn(x, *args, **kwargs):
    return x


def _identity_causal_conv1d_update(x, *args, **kwargs):
    return x


def _build_split_and_plugin_chunk_metadata(module, query_lens, state_indices, device):
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    query_start_loc_cpu = torch.tensor(
        [0] + list(torch.tensor(query_lens, device="cpu").cumsum(0).tolist()),
        dtype=torch.int32,
        device="cpu",
    )
    query_start_loc = query_start_loc_cpu.to(device)
    num_decodes = sum(1 for query_len in query_lens if query_len == 1)
    num_decode_tokens = num_decodes
    num_prefills = len(query_lens) - num_decodes
    num_prefill_tokens = int(query_start_loc_cpu[-1].item()) - num_decode_tokens
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    has_initial_state = torch.tensor(
        [True] * num_decodes + [False] + [True] * (len(query_lens) - num_decodes - 1),
        device=device,
    )
    prefill_query_start_loc_cpu = query_start_loc_cpu[num_decodes:] - num_decode_tokens
    prefill_query_start_loc = prefill_query_start_loc_cpu.to(device)
    chunk_indices = prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE)
    meta_split = module.IluGDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(query_start_loc_cpu[-1].item()),
        has_initial_state=has_initial_state,
        chunk_indices=chunk_indices.to(device),
        chunk_offsets=chunk_offsets.to(device),
        prefill_query_start_loc=prefill_query_start_loc,
        prefill_state_indices=state_indices[num_decodes:],
        prefill_has_initial_state=has_initial_state[num_decodes:],
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
        non_spec_query_start_loc_cpu=query_start_loc_cpu,
    )
    meta_plugin_chunk = dataclasses.replace(
        meta_split,
        has_initial_state=has_initial_state.to(torch.int32),
        chunk_offsets=torch.tensor(
            [0]
            + list(
                torch.tensor(
                    [(length + 15) // 16 for length in query_lens],
                    dtype=torch.int32,
                )
                .cumsum(0)
                .tolist()
            ),
            dtype=torch.int32,
            device=device,
        ),
        n_chunk_max=max((length + 15) // 16 for length in query_lens),
        n_chunk_total=sum((length + 15) // 16 for length in query_lens),
    )
    return meta_split, meta_plugin_chunk


def _build_community_chunk_rule(community_module):
    from vllm.config import VllmConfig, set_current_vllm_config

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_key_head_dim=HEAD_K_DIM)
    )
    vllm_config.additional_config = {"gdn_prefill_backend": "triton"}
    with set_current_vllm_config(vllm_config):
        return community_module.ChunkGatedDeltaRule()


@test_forward_core
@requires_cuda
def test_forward_core_split_reference_matches_plugin_non_split(
    cuda_device, monkeypatch
):
    """Compare community split_non_spec against plugin all-in-one chunk path."""
    plugin_module = _import_real_plugin_module()
    community_module = _import_community_gdn_module()
    for module in (plugin_module, community_module):
        monkeypatch.setattr(module, "causal_conv1d_fn", _identity_causal_conv1d_fn)
        monkeypatch.setattr(
            module, "causal_conv1d_update", _identity_causal_conv1d_update
        )
    device = torch.device(cuda_device)
    torch.manual_seed(0)
    query_lens = [1, 1, 5, 3]
    num_tokens = sum(query_lens)
    dtype = torch.bfloat16

    # vLLM GDN recurrent kernels reserve state index 0 as NULL_BLOCK_ID.
    # Use positive cache slots so community split decode and ixformer chunk load
    # the same initial states.
    state_indices = torch.arange(
        1, len(query_lens) + 1, device=device, dtype=torch.int32
    )
    pool_size = int(state_indices.max().item()) + 1
    meta_split, meta_plugin_chunk = _build_split_and_plugin_chunk_metadata(
        plugin_module, query_lens, state_indices, device
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    conv_state_shape = (
        (pool_size, CONV_DIM, CONV_KERNEL - 1)
        if is_conv_state_dim_first()
        else (pool_size, CONV_KERNEL - 1, CONV_DIM)
    )
    conv_state0 = torch.randn(*conv_state_shape, dtype=dtype, device=device).clamp_(
        -0.1, 0.1
    )
    ssm_state0 = torch.randn(
        pool_size,
        NUM_V_HEADS,
        HEAD_V_DIM,
        HEAD_K_DIM,
        dtype=torch.float32,
        device=device,
    ).clamp_(-0.1, 0.1)
    conv_weight = torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=dtype, device=device
    ).clamp_(-0.1, 0.1)
    a_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device) * 0.1
    mixed_qkv = torch.randn(num_tokens, CONV_DIM, dtype=dtype, device=device).clamp_(
        -0.1, 0.1
    )
    b = torch.randn(num_tokens, NUM_V_HEADS, dtype=dtype, device=device).clamp_(
        -0.1, 0.1
    )
    a = torch.randn(num_tokens, NUM_V_HEADS, dtype=dtype, device=device).clamp_(
        -0.1, 0.1
    )

    layer_community_split = _build_gdn_layer(
        community_module.QwenGatedDeltaNetAttention,
        conv_state0.clone(),
        ssm_state0.clone(),
        a_log,
        dt_bias,
        conv_weight,
        chunk_gated_delta_rule=_build_community_chunk_rule(community_module),
    )
    out_community_split = _run_forward_core(
        community_module, layer_community_split, meta_split, mixed_qkv, b, a
    )

    layer_plugin_non_split = _build_gdn_layer(
        plugin_module.IluQwenGatedDeltaNetAttention,
        conv_state0.clone(),
        ssm_state0.clone(),
        a_log,
        dt_bias,
        conv_weight,
    )
    out_plugin_non_split = _run_forward_core(
        plugin_module, layer_plugin_non_split, meta_plugin_chunk, mixed_qkv, b, a
    )

    num_decode_tokens = meta_split.num_decode_tokens
    # ixformer all-in-one chunk and community split use different kernels; the
    # remaining bf16 decode delta is numeric, not a metadata split mismatch.
    torch.testing.assert_close(
        out_plugin_non_split[:num_decode_tokens],
        out_community_split[:num_decode_tokens],
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        out_plugin_non_split[num_decode_tokens:],
        out_community_split[num_decode_tokens:],
        atol=3e-2,
        rtol=3e-2,
    )


def test_qwen38_kernel_follows_arch(monkeypatch):
    from vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn import (
        IluQwenGatedDeltaNetAttention,
    )

    dummy = SimpleNamespace()
    bound = MethodType(IluQwenGatedDeltaNetAttention._qwen38_kernel, dummy)
    monkeypatch.setattr(
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn.is_ivcore30",
        lambda: True,
    )
    assert bound() == "ixformer"
    monkeypatch.setattr(
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn.is_ivcore30",
        lambda: False,
    )
    assert bound() == "plugin_fla"


def _make_attention(
    monkeypatch,
    *,
    ivcore30=True,
    output_gate_type=None,
    speculative_config=None,
    **geometry,
):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    parent = module.IluQwenGatedDeltaNetAttention.__mro__[1]

    def fake_parent_init(self, config, *args, **kwargs):
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads

    monkeypatch.setattr(parent, "__init__", fake_parent_init)
    monkeypatch.setattr(module, "is_ivcore30", lambda: ivcore30)
    config_values = {
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_kernel_size": 4,
        "num_k_heads": 16,
        "num_v_heads": 48,
    }
    config_values.update(geometry)
    if output_gate_type is not None:
        config_values["output_gate_type"] = output_gate_type
    config = SimpleNamespace(**config_values)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=128,
        ),
        speculative_config=speculative_config,
    )
    return module.IluQwenGatedDeltaNetAttention(config, vllm_config), module


@pytest.mark.parametrize(
    ("configured", "normalized", "code"),
    [(None, "silu", 1), ("swish", "silu", 1), ("sigmoid", "sigmoid", 0)],
)
def test_output_gate_type_is_normalized(monkeypatch, configured, normalized, code):
    op, _ = _make_attention(
        monkeypatch,
        ivcore30=False,
        output_gate_type=configured,
    )
    assert op.output_gate_type == normalized
    assert op._gate_act_code == code


def test_output_gate_type_rejects_unsupported_value(monkeypatch):
    with pytest.raises(ValueError, match="unsupported output_gate_type"):
        _make_attention(
            monkeypatch,
            ivcore30=False,
            output_gate_type="gelu",
        )


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"head_k_dim": 64}, "head_k_dim"),
        ({"conv_kernel_size": 3}, "conv_kernel_size"),
        ({"num_k_heads": 2, "num_v_heads": 3}, "num_v_heads"),
    ],
)
def test_ixformer_geometry_fails_at_construction(monkeypatch, geometry, message):
    with pytest.raises(ValueError, match=message) as exc_info:
        _make_attention(monkeypatch, **geometry)
    assert "VLLM_ILU_GDN_ATTN_BACKEND=0" in str(exc_info.value)


def test_ixformer_speculative_decode_is_accepted_at_construction(monkeypatch):
    op, _ = _make_attention(monkeypatch, speculative_config=object())
    assert op._qwen38_kernel() == "ixformer"
    assert op._ixformer_spec_enabled()


def test_ixformer_spec_switch_can_disable_only_spec_path(monkeypatch):
    op, _ = _make_attention(monkeypatch, speculative_config=object())
    monkeypatch.setenv("VLLM_ILU_GDN_IXFORMER_SPEC", "0")
    assert not op._ixformer_spec_enabled()


def test_non_ivcore30_keeps_speculative_decode_fallback(monkeypatch):
    op, _ = _make_attention(
        monkeypatch,
        ivcore30=False,
        speculative_config=object(),
    )
    assert op._qwen38_kernel() == "plugin_fla"


def test_stale_ixformer_activation_abi_has_clear_error(monkeypatch):
    module, _, _ = _load_plugin_module(monkeypatch, gdn_backend=True)
    module._ixformer_gdn_activation_abi_ready = False

    def old_abi(*args):
        raise TypeError("takes 5 positional arguments")

    monkeypatch.setattr(
        module.ixf_c_ops,
        "infer",
        SimpleNamespace(qwen_gdn_rmsnorm_gated=old_abi),
    )
    with pytest.raises(RuntimeError, match="rebuild and reinstall ixformer"):
        module._run_ixformer_rmsnorm_gated(None, None, None, None, 1e-6, 1)
