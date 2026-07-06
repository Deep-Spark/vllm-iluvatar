# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar rotary and Conv3D OOT replacements.

The 10 base RotaryEmbedding ``forward_cuda`` OOT overrides were removed once the
plugin gained a native community-signature ``torch.ops._C.rotary_embedding``
(csrc/pos_encoding_kernels.cu): the community ``RotaryEmbedding.forward_cuda``
now works unchanged, so those subclasses must NOT be in ``op_registry_oot`` and
the plugin's ``_custom_ops`` no longer defines ``rotary_embedding``. The numeric
kernel correctness is covered by tests/kernels/core/test_pos_encoding.py.

The ``ApplyRotaryEmb`` OOT override is retained (single-class, the sanctioned
CustomOp extension point) but its ``forward_cuda`` now routes to the plugin's
``torch.ops._C.apply_rotary_emb`` kernel instead of falling back to
``forward_native``; numeric correctness is covered by
tests/kernels/core/test_apply_rotary_emb.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.conv import Conv3dLayer  # noqa: F401
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb  # noqa: F401

import vllm_iluvatar._custom_ops as ilu_ops
from vllm_iluvatar.model_executor.layers.conv import IluConv3dLayer
from vllm_iluvatar.model_executor.layers.rotary_embedding.common import (
    IluApplyRotaryEmb,
)

# The base RotaryEmbedding classes the plugin used to override with
# per-subclass forward_cuda deltas (now handled by the native _C kernel).
_REMOVED_ROPE_OOT_KEYS = [
    "RotaryEmbedding",
    "Llama3RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "NTKScalingRotaryEmbedding",
    "DynamicNTKAlphaRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "FourierRotaryEmbedding",
    "YaRNScalingRotaryEmbedding",
    "TeleChat3RoPEScaledRotaryEmbedding",
    "Gemma4RotaryEmbedding",
]


def test_base_rotary_overrides_removed():
    """The 10 base rope forward_cuda overrides are gone (native _C kernel)."""
    for key in _REMOVED_ROPE_OOT_KEYS:
        assert key not in op_registry_oot, (
            f"{key} should no longer be OOT-overridden; the community "
            f"forward_cuda now calls torch.ops._C.rotary_embedding directly"
        )


def test_plugin_custom_ops_has_no_rotary_embedding():
    """Plugin _custom_ops no longer overrides rotary_embedding."""
    assert not hasattr(ilu_ops, "rotary_embedding")
    assert "rotary_embedding" not in ilu_ops._ILU_OVERRIDES
    assert "rotary_embedding" not in ilu_ops._ILU_ADDED


def test_rotary_embedding_not_in_skip_sigcheck():
    from vllm_iluvatar.ilu_patch.patch_custom_ops import _SKIP_SIGCHECK

    assert "rotary_embedding" not in _SKIP_SIGCHECK


def test_apply_rotary_emb_oot_uses_native_c_kernel():
    """forward_cuda now routes through the plugin's torch.ops._C.apply_rotary_emb
    (via _pre_process/_post_process), replacing the previous forward_native
    fallback. CPU-only: the _C kernel call itself is mocked; numeric correctness
    is covered by tests/kernels/core/test_apply_rotary_emb.py."""
    assert op_registry_oot["ApplyRotaryEmb"] is IluApplyRotaryEmb

    op = object.__new__(IluApplyRotaryEmb)
    op.is_neox_style = True
    x = torch.empty((1, 2, 1, 4))
    cos = torch.empty((2, 2))
    sin = torch.empty((2, 2))

    kernel_out = torch.empty((1, 2, 1, 4))
    processed = (x, cos, sin, x.shape, x.dtype)
    op._pre_process = MagicMock(return_value=processed)
    op._post_process = MagicMock(return_value=kernel_out)

    apply_op = MagicMock(return_value=kernel_out)
    with patch.object(torch.ops._C, "apply_rotary_emb", apply_op, create=True):
        result = IluApplyRotaryEmb.forward_cuda(op, x, cos, sin)

    assert result is kernel_out
    op._pre_process.assert_called_once_with(x, cos, sin)
    # is_neox_style=True -> interleaved=False
    apply_op.assert_called_once_with(x, cos, sin, False)
    op._post_process.assert_called_once_with(kernel_out, x.shape, x.dtype)


def test_conv3d_oot_uses_matmul_cuda_path():
    assert op_registry_oot["Conv3dLayer"] is IluConv3dLayer

    conv = object.__new__(IluConv3dLayer)
    expected = torch.empty((1, 1, 1, 1, 1))
    conv._forward_mulmat = MagicMock(return_value=expected)
    x = torch.empty((1, 1, 1, 1, 1))

    assert IluConv3dLayer.forward_cuda(conv, x) is expected
    conv._forward_mulmat.assert_called_once_with(x)


def test_rotary_package_is_auto_discovered():
    from vllm_iluvatar.ilu_register import _OOT_OP_PACKAGES

    assert "vllm_iluvatar.model_executor.layers.rotary_embedding" in _OOT_OP_PACKAGES
