# SPDX-License-Identifier: Apache-2.0
"""Check private DS4 adapters against the unchanged native contract."""

import pytest
import torch

import vllm_iluvatar._custom_ops as ilu_ops


# New ixformer-only hints must not leak into ixaite's existing sparse attention ABI.
def test_ds4_ixaite_adapters_preserve_unfused_contract(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ilu_ops

    calls = []

    # Capture exactly the arguments the bound kernel receives.
    def kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    monkeypatch.setattr(
        ilu_ops, "load_ixaite_python_op", lambda *_: (kernel, "available")
    )
    decode, _ = ilu_ops._load_ixaite_routed_op("ds4_flash_mla_with_kvcache")
    assert decode("q", positions=None, cos_sin_cache=None, output="out") == "output"
    assert calls[-1] == (("q",), {"output": "out"})
    with pytest.raises(ValueError, match="separate inverse RoPE"):
        decode("q", positions=torch.zeros(1))
    logits, _ = ilu_ops._load_ixaite_routed_op(
        "dsa_indexer_mqa_logits_with_blocks_bf16"
    )
    assert logits("q", backend="ixformer_native", k="k") == "output"
    assert calls[-1] == (("q",), {"k": "k"})


# Fusion eligibility must describe the selected provider, not an available native library.
@pytest.mark.parametrize("baseline", [False, True])
def test_ds4_inverse_rope_capability_follows_selected_provider(monkeypatch, baseline):
    from vllm_iluvatar.backends.ixaite import ops as own

    monkeypatch.setattr(
        ilu_ops.ixf_ops,
        "ds4_flash_mla_with_kvcache_supports_inv_rope",
        lambda **_: True,
    )
    implementation = ilu_ops.ixf_ops if baseline else own
    assert implementation.ds4_flash_mla_with_kvcache_supports_inv_rope() is baseline
