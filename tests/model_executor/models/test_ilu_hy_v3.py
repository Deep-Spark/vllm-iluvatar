# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar OOT HunYuan-V3 model.

Covers:
  * OOT registration: ``HYV3ForCausalLM`` resolves to the plugin subclass, which
    is a subclass of the community model (so ``load_weights`` / W4A8 loading are
    inherited unchanged).
  * Numerical: the fused q/k RMSNorm CUDA kernel (``RMSNormQK.forward_cuda``,
    ixformer ``rms_norm_qk``) matches the pure-torch reference math used by the
    fused attention forward. This is the only Iluvatar delta vs community.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap

import torch
from conftest import requires_cuda


def has_attr(cls: type, attr_name: str) -> bool:
    # static ast check
    init = cls.__init__
    source = textwrap.dedent(inspect.getsource(init))
    tree = ast.parse(source)

    return any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attr_name
            for target in node.targets
        )
        for node in ast.walk(tree)
    )


def test_hy_v3_oot_registration_points_at_plugin_subclass():
    from vllm import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_iluvatar.ilu_register import apply_oot_registrations

    # for a model that keep official name, first check it must in _VLLM_MODELS
    assert "HYV3ForCausalLM" in _VLLM_MODELS

    # check our code minimum import dependency
    try:
        hy_v3 = importlib.import_module("vllm.model_executor.models.hy_v3")
    except ImportError as exc:
        raise AssertionError(
            "Failed to import HYV3 community model dependencies"
        ) from exc
    assert hasattr(hy_v3, "HYV3ForCausalLM")
    assert hasattr(hy_v3, "HYV3Attention")
    assert hasattr(hy_v3, "HYV3MoEFused")
    HYV3Attention = hy_v3.HYV3Attention
    HYV3MoEFused = hy_v3.HYV3MoEFused

    # check qk norm dependencies(ast check), check if use_qk_norm is True, but ast shows no q_norm and k_norm
    assert has_attr(HYV3Attention, "q_norm")
    assert has_attr(HYV3Attention, "k_norm")
    # check gate dependencies(ast check), HYV3MoEFused must has a gate attribute
    assert has_attr(HYV3MoEFused, "gate")

    # after dependency check, process oot registrations and check module_name and class_name
    apply_oot_registrations()
    entry = ModelRegistry.models["HYV3ForCausalLM"]
    assert entry.module_name == "vllm_iluvatar.model_executor.models.hy_v3"
    assert entry.class_name == "HYV3ForCausalLM"


def test_hy_v3_mtp_oot_registration_points_at_plugin_subclass():
    from vllm import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_iluvatar.ilu_register import apply_oot_registrations

    # for a model that keep official name, first check it must in _VLLM_MODELS
    assert "HYV3MTPModel" in _VLLM_MODELS

    # check our code minimum import dependency
    try:
        hy_v3_mtp = importlib.import_module("vllm.model_executor.models.hy_v3_mtp")
    except ImportError as exc:
        raise AssertionError(
            "Failed to import HYV3 MTP community model dependencies"
        ) from exc
    assert hasattr(hy_v3_mtp, "HYV3MTP")

    from vllm.model_executor.models.hy_v3 import HYV3DecoderLayer

    HYV3MultiTokenPredictorLayer = hy_v3_mtp.HYV3MultiTokenPredictorLayer

    # check replace dependencies(ast check)
    assert has_attr(HYV3MultiTokenPredictorLayer, "mtp_block")
    assert has_attr(HYV3DecoderLayer, "mlp")
    # gate already checked on has_attr(HYV3MoEFused, "gate")

    # after dependency check, process oot registrations and check module_name and class_name
    apply_oot_registrations()
    entry = ModelRegistry.models["HYV3MTPModel"]
    assert entry.module_name == "vllm_iluvatar.model_executor.models.hy_v3_mtp"
    assert entry.class_name == "HYV3MTP"


def test_hy_v3_plugin_class_subclasses_community():
    # Subclassing (not a whole-file copy) is what lets community bugfixes and the
    # W4A8 load path be inherited unchanged.
    from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM as _CommunityHYV3

    from vllm_iluvatar.model_executor.models.hy_v3 import HYV3ForCausalLM

    assert issubclass(HYV3ForCausalLM, _CommunityHYV3)

    from vllm.model_executor.models.hy_v3_mtp import HYV3MTP as _CommunityHYV3MTP

    from vllm_iluvatar.model_executor.models.hy_v3_mtp import HYV3MTP

    assert issubclass(HYV3MTP, _CommunityHYV3MTP)


@requires_cuda
def test_hy_v3_fused_qk_norm_matches_reference(cuda_device, default_vllm_config):
    from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK

    torch.manual_seed(0)
    num_tokens, num_q_heads, num_kv_heads, head_dim, eps = 4, 8, 2, 128, 1e-6
    dtype = torch.bfloat16

    q = torch.randn(num_tokens, num_q_heads, head_dim, device=cuda_device, dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=cuda_device, dtype=dtype)
    weight_q = torch.randn(head_dim, device=cuda_device, dtype=dtype)
    weight_k = torch.randn(head_dim, device=cuda_device, dtype=dtype)

    op = RMSNormQK(head_dim, head_dim, eps=eps)

    ref_q, ref_k = op.forward_native(q, k, weight_q, weight_k)
    out_q, out_k = op.forward_cuda(q, k, weight_q, weight_k)

    torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)
