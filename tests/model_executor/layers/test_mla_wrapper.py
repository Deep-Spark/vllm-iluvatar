# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar MLA OOT wrapper."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

import vllm_iluvatar.ilu_register as ilu_register
from vllm_iluvatar.model_executor.layers.mla import IluMultiHeadLatentAttentionWrapper


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
            ),
            {},
        )
    ]
