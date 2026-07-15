# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar OOT HunYuan-V3 (no-MTP) model.

Covers:
  * OOT registration: ``HYV3ForCausalLM`` resolves to the plugin subclass, which
    is a subclass of the community model (so ``load_weights`` / W4A8 loading are
    inherited unchanged).
  * Numerical: the fused q/k RMSNorm CUDA kernel (``RMSNormQK.forward_cuda``,
    ixformer ``rms_norm_qk``) matches the pure-torch reference math used by the
    fused attention forward. This is the only Iluvatar delta vs community.
"""

from __future__ import annotations

import torch
from conftest import requires_cuda


def test_hy_v3_oot_registration_points_at_plugin_subclass():
    from vllm import ModelRegistry

    from vllm_iluvatar.ilu_register import apply_oot_registrations

    apply_oot_registrations()

    entry = ModelRegistry.models["HYV3ForCausalLM"]
    assert entry.module_name == "vllm_iluvatar.model_executor.models.hy_v3"
    assert entry.class_name == "HYV3ForCausalLM"


def test_hy_v3_plugin_class_subclasses_community():
    # Subclassing (not a whole-file copy) is what lets community bugfixes and the
    # W4A8 load path be inherited unchanged.
    from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM as _CommunityHYV3

    from vllm_iluvatar.model_executor.models.hy_v3 import HYV3ForCausalLM

    assert issubclass(HYV3ForCausalLM, _CommunityHYV3)


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
