# SPDX-License-Identifier: Apache-2.0
"""Tests for packed ViT QKV layout and RoPE provider routing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from vllm_iluvatar.backends.ixaite.attention import vit as wrappers


# The BF16 path delegates padding once; quantized and FP16 calls retain native-width providers.
@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("dtype,opt_level,fused", [("bfloat16", 0, True), ("float16", 0, False), ("bfloat16", 1, False)])
def test_preparation_padding_contract(monkeypatch, rotary, dtype, opt_level, fused):
    import torch

    monkeypatch.setattr(wrappers.envs, "VLLM_ILU_VIT_ATTN_OPT_LEVEL", opt_level)
    packed = torch.randn((5, 1, 3 * 2 * 72), dtype=getattr(torch, dtype))
    requested = []

    # Record the optional ABI while retaining real output shapes and native values.
    def layout(x, heads, dim, padded_head_dim=None):
        requested.append(padded_head_dim)
        parts = x.reshape(5, 1, 3, heads, dim).permute(1, 0, 2, 3, 4).unbind(2)
        width = dim if padded_head_dim is None else padded_head_dim
        outputs = []
        for part in parts:
            out = torch.zeros((1, 5, heads, width), dtype=x.dtype)
            out[..., :dim] = part
            outputs.append(out)
        return tuple(outputs)

    monkeypatch.setattr(wrappers, "_VIT_QKV_LAYOUT", layout)
    monkeypatch.setattr(wrappers, "_VIT_QKV_ROPE", lambda x, c, s, h, d, **kw: layout(x, h, d, **kw))
    table = torch.ones((5, 36)) if rotary else None
    actual = wrappers.vit_fused_qkv_preproc(packed, table, table, 1, 5, 2, 72, 128)
    expected = packed.reshape(5, 1, 3, 2, 72).permute(1, 0, 2, 3, 4).unbind(2)
    assert requested == [128 if fused else None]
    for got, reference in zip(actual, expected):
        assert got.shape == (1, 5, 2, 128) and torch.count_nonzero(got[..., 72:]) == 0
        torch.testing.assert_close(got[..., :72], reference, rtol=0, atol=0)

_LAYOUT = (
    wrappers.configure_vit_qkv_layout_provider,
    "_VIT_QKV_LAYOUT",
    None,
    "vit_qkv_layout",
)
_ROPE = (
    wrappers.configure_vit_qkv_rope_provider,
    "_VIT_QKV_ROPE",
    None,
    "vit_qkv_rope",
)


@pytest.mark.parametrize(("configure", "attr", "_eager", "op_name"), (_LAYOUT, _ROPE))
def test_forced_ixaite_binds_kernel(
    monkeypatch, configure, attr, _eager, op_name
) -> None:
    ixaite_op = Mock(return_value="ixaite")
    loader = Mock(return_value=(ixaite_op, "supported on ivcore11"))
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(wrappers, "load_ixaite_torch_op", loader)

    configure()

    assert getattr(wrappers, attr) is ixaite_op
    loader.assert_called_once_with(op_name)


@pytest.mark.parametrize(("configure", "_attr", "_eager", "_op_name"), (_LAYOUT, _ROPE))
def test_forced_ixaite_raises_when_unavailable(
    monkeypatch, configure, _attr, _eager, _op_name
) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        wrappers,
        "load_ixaite_torch_op",
        Mock(return_value=(None, "not validated on this arch")),
    )

    with pytest.raises(RuntimeError, match="not validated on this arch"):
        configure()
