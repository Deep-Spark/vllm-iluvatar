# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GDN OOT chunk entry."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import vllm_iluvatar.ilu_register as ilu_register


def _install_fake_community_gdn(monkeypatch):
    registered = []

    class FakeChunkGatedDeltaRule:
        def __init__(self) -> None:
            self.gdn_prefill_backend = "flashinfer"
            self._forward_method = self.forward_flashinfer

        @classmethod
        def register_oot(cls):
            def decorator(oot_cls):
                registered.append(oot_cls)
                return oot_cls

            return decorator

        def forward_flashinfer(self, *args, **kwargs):
            raise AssertionError("flashinfer path should not be used")

    module_name = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    fake_module = ModuleType(module_name)
    fake_module.ChunkGatedDeltaRule = FakeChunkGatedDeltaRule
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return registered


def _load_plugin_module(monkeypatch):
    registered = _install_fake_community_gdn(monkeypatch)
    module_name = (
        "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name), registered


def test_qwen_gdn_oot_module_is_auto_discovered():
    assert "vllm_iluvatar.model_executor.layers.mamba.gdn" in (
        ilu_register._OOT_OP_PACKAGES
    )


def test_ilu_chunk_gated_delta_rule_registers_and_forces_plugin_native(monkeypatch):
    module, registered = _load_plugin_module(monkeypatch)

    assert registered == [module.IluChunkGatedDeltaRule]

    calls = []

    def fake_chunk_gated_delta_rule(**kwargs):
        calls.append(kwargs)
        return "plugin-output"

    fake_chunk_module = ModuleType("vllm_iluvatar.model_executor.layers.fla.ops.chunk")
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
