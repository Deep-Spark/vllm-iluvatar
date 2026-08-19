# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar MLA OOT wrapper."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

import vllm_iluvatar.ilu_register as ilu_register
from vllm_iluvatar.model_executor.layers.mla import IluMultiHeadLatentAttentionWrapper

_MAIN_B35FF5F7739E_FLASH_ATTN_DIFF_HEAD_DIMS_SHA = (
    "cb6ba62216aa9d3148bcc84f699e417fc71519f1b6824121dab417f5cf750a9b"
)


def test_mla_layer_package_is_auto_discovered():
    assert "vllm_iluvatar.model_executor.layers" in ilu_register._OOT_OP_PACKAGES


def test_ilu_mla_wrapper_registers_oot():
    assert issubclass(
        IluMultiHeadLatentAttentionWrapper, MultiHeadLatentAttentionWrapper
    )
    assert (
        op_registry_oot["MultiHeadLatentAttentionWrapper"]
        is IluMultiHeadLatentAttentionWrapper
    )


def test_ilu_mla_wrapper_sets_inner_rotary_emb(monkeypatch):
    calls = []
    install_calls = []

    def fake_parent_init(self, *args, **kwargs):
        import torch

        calls.append((args, kwargs))
        torch.nn.Module.__init__(self)
        self.rotary_emb = object()
        self.mla_attn = SimpleNamespace(rotary_emb=None)

    monkeypatch.setattr(
        MultiHeadLatentAttentionWrapper,
        "__init__",
        fake_parent_init,
    )

    def fake_install_ilu_mla_attention_overrides(mla_attn, rotary_emb):
        install_calls.append((mla_attn, rotary_emb))
        mla_attn.rotary_emb = rotary_emb

    monkeypatch.setattr(
        "vllm_iluvatar.model_executor.layers.mla.install_ilu_mla_attention_overrides",
        fake_install_ilu_mla_attention_overrides,
    )

    mla_modules = object()
    wrapper = IluMultiHeadLatentAttentionWrapper(
        hidden_size=16,
        num_heads=2,
        scale=1.0,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        mla_modules=mla_modules,
    )

    assert wrapper.mla_attn.rotary_emb is wrapper.rotary_emb
    assert install_calls == [(wrapper.mla_attn, wrapper.rotary_emb)]
    assert calls == [
        (
            (
                16,
                2,
                1.0,
                4,
                4,
                4,
                None,
                8,
                mla_modules,
                None,
                None,
                "",
                False,
                False,
                False,
            ),
            {},
        )
    ]


def test_install_ilu_mla_attention_overrides_binds_instance_methods(monkeypatch):
    import vllm_iluvatar.model_executor.layers.attention.mla_attention as ilu_mla

    layer = SimpleNamespace(impl=SimpleNamespace())
    rotary_emb = object()

    ilu_mla.install_ilu_mla_attention_overrides(layer, rotary_emb)

    assert layer.rotary_emb is rotary_emb
    assert layer.use_int8_mla is False
    assert layer.forward.__self__ is layer
    assert layer.process_weights_after_loading.__self__ is layer


def test_ilu_mla_metadata_builder_expands_spec_decode_rows():
    import torch

    from vllm_iluvatar.v1.attention.backends.mla.ilu_mla import (
        IluMLACommonMetadataBuilder,
    )

    builder = object.__new__(IluMLACommonMetadataBuilder)
    builder.decode_use_graph = False
    builder.use_full_cuda_graph = False
    builder.dcp_world_size = 1

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)

    expanded_block_table, expanded_seq_lens = builder._expand_spec_decode_metadata(
        block_table,
        seq_lens,
        query_start_loc,
        num_decode_tokens=5,
    )

    assert expanded_block_table.tolist() == [
        [10, 11],
        [10, 11],
        [20, 21],
        [20, 21],
        [20, 21],
    ]
    assert expanded_seq_lens.tolist() == [4, 5, 5, 6, 7]


def test_ilu_full_forward_requires_positions_and_rotary_emb():
    import pytest
    import torch

    import vllm_iluvatar.model_executor.layers.attention.mla_attention as ilu_mla

    layer = SimpleNamespace(
        impl=SimpleNamespace(forward_full=lambda *args, **kwargs: None),
        rotary_emb=None,
    )

    with pytest.raises(AssertionError, match="requires positions and rotary_emb"):
        ilu_mla._ilu_mla_attention_forward(
            layer,
            q=torch.empty(0),
            kv_c_normed=torch.empty(0),
            k_pe=torch.empty(0),
            position=None,
        )


def test_ilu_process_weights_unsupported_impl_uses_community(monkeypatch):
    import torch

    import vllm_iluvatar.model_executor.layers.attention.mla_attention as ilu_mla

    calls = []
    layer = SimpleNamespace(impl=SimpleNamespace())

    def fake_community_process_weights(layer_arg, act_dtype):
        calls.append((layer_arg, act_dtype))

    monkeypatch.setattr(
        ilu_mla,
        "_COMMUNITY_PROCESS_WEIGHTS_AFTER_LOADING",
        fake_community_process_weights,
    )

    ilu_mla._ilu_process_weights_after_loading(layer, torch.bfloat16)

    assert calls == [(layer, torch.bfloat16)]


def test_ilu_process_weights_none_setup_uses_community(monkeypatch):
    import torch

    import vllm_iluvatar.model_executor.layers.attention.mla_attention as ilu_mla

    calls = []

    class IluMLAImpl:
        def setup_projections(self, layer, act_dtype):
            calls.append((layer, act_dtype))
            return None

    layer = SimpleNamespace(impl=IluMLAImpl(), rotary_emb=None)

    def fake_community_process_weights(layer_arg, act_dtype):
        calls.append(("community", layer_arg, act_dtype))

    monkeypatch.setattr(
        ilu_mla,
        "_COMMUNITY_PROCESS_WEIGHTS_AFTER_LOADING",
        fake_community_process_weights,
    )

    ilu_mla._ilu_process_weights_after_loading(layer, torch.bfloat16)

    assert calls == [
        (layer, torch.bfloat16),
        ("community", layer, torch.bfloat16),
    ]


def test_ilu_mla_metadata_builder_enables_uniform_spec_decode():
    import vllm.model_executor.layers.attention.mla_attention as community_mla

    from vllm_iluvatar.v1.attention.backends.mla.ilu_mla import (
        IluMLACommonMetadataBuilder,
    )

    assert (
        IluMLACommonMetadataBuilder.query_len_support
        is community_mla.QueryLenSupport.UNIFORM
    )


def test_ilu_mla_metadata_builder_supports_uniform_cuda_graph():
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_iluvatar.v1.attention.backends.mla.ilu_mla import (
        IluMLACommonMetadataBuilder,
    )

    assert (
        IluMLACommonMetadataBuilder._cudagraph_support
        is AttentionCGSupport.UNIFORM_BATCH
    )


def test_mla_prefill_flash_attn_fingerprint_baseline_hash():
    from vllm_iluvatar.ilu_patch import general_patch as p
    from vllm_iluvatar.v1.attention.backends.mla.prefill import flash_attn

    assert (
        p.source_sha(flash_attn._FLASH_ATTN_VARLEN_DIFF_HEADDIMS_BASELINE)
        == _MAIN_B35FF5F7739E_FLASH_ATTN_DIFF_HEAD_DIMS_SHA
    )


def test_mla_prefill_flash_attn_drops_output_scale_and_unpads_v(monkeypatch):
    import torch

    from vllm_iluvatar.v1.attention.backends.mla.prefill import flash_attn

    calls = {}

    def fake_kernel(**kwargs):
        calls.update(kwargs)
        assert kwargs["v"].shape[-1] == kwargs["q"].shape[-1]
        return torch.zeros((1, 1, 4), dtype=kwargs["q"].dtype)

    monkeypatch.setattr(
        flash_attn,
        "FA4_MLA_PREFILL_KERNEL",
        fake_kernel,
    )

    prefill_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
    )
    prefill_backend = object.__new__(flash_attn.FlashAttnPrefillBackend)
    prefill_backend.flash_attn_varlen_func = object()
    prefill_backend.requires_v_padding = True
    prefill_backend._is_vllm_fa = True
    prefill_backend.scale = 0.5
    prefill_backend._prefill_metadata = prefill_metadata

    output = prefill_backend.run_prefill_new_tokens(
        q=torch.zeros((1, 1, 4)),
        k=torch.zeros((1, 1, 4)),
        v=torch.zeros((1, 1, 2)),
        return_softmax_lse=False,
    )

    assert output.shape[-1] == 2
    assert "out" not in calls
    assert "output_scale" not in calls
    assert calls["return_softmax_lse"] is False


def test_ilu_platform_registers_mla_prefill_flash_attn_backend(monkeypatch):
    import vllm.envs as envs
    import vllm.v1.attention.backends.mla.prefill.registry as prefill_registry
    import vllm.v1.attention.backends.registry as attention_registry
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
    )

    from vllm_iluvatar.platforms import ilu_platform

    calls = []
    monkeypatch.setattr(
        prefill_registry,
        "register_mla_prefill_backend",
        lambda backend, class_path: calls.append((backend, class_path)),
    )
    monkeypatch.setattr(attention_registry, "register_backend", lambda *_, **__: None)
    monkeypatch.setattr(envs, "VLLM_ILU_GDN_ATTN_BACKEND", False, raising=False)
    monkeypatch.setattr(ilu_platform, "_ATTN_BACKENDS_REGISTERED", False)

    ilu_platform.register_attention_backends()

    assert calls == [
        (
            MLAPrefillBackendEnum.FLASH_ATTN,
            ilu_platform._ILU_MLA_PREFILL_FLASH_ATTN_BACKEND,
        )
    ]
