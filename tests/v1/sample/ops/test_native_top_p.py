# SPDX-License-Identifier: Apache-2.0
"""Native top-p dispatch and fallback contracts."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_iluvatar.envs import apply
from vllm_iluvatar.v1.sample.ops import topk_topp_sampler as impl

apply()


def fake_cuda(*shape, dtype=torch.float32):
    # Dispatch contracts need CUDA metadata, not a GPU allocation/context.
    return FakeTensor(
        FakeTensorMode(),
        torch.empty(*shape, dtype=dtype, device="meta"),
        torch.device("cuda"),
    )


@pytest.mark.parametrize(
    "reason",
    [
        "supported",
        "top_k",
        "no_p",
        "small_vocab",
        "empty",
        "noncontiguous",
        "half",
        "p_shape",
        "p_device",
        "old_ixformer",
        "arch",
    ],
)
def test_dispatch(monkeypatch, reason):
    import ixformer.inference.backend_registry as registry
    import ixformer.inference.functions as F

    calls = []
    monkeypatch.setattr(
        F, "apply_top_p_radix", lambda *a: calls.append("native") or "native"
    )
    monkeypatch.setattr(
        impl,
        "_original_apply_top_k_top_p",
        lambda logits, k, p: (
            logits
            if p is None and k is None
            else calls.append("fallback") or "fallback"
        ),
    )
    x = fake_cuda(3, 4097)
    p = fake_cuda(3)
    k = None
    if reason == "top_k":
        k = fake_cuda(3, dtype=torch.int32)
    if reason == "no_p":
        p = None
    if reason == "small_vocab":
        x = x[:, :4096].contiguous()
    if reason == "empty":
        x = x[:0]
        p = p[:0]
    if reason == "noncontiguous":
        x = fake_cuda(3, 8194)[:, ::2]
    if reason == "half":
        x = x.half()
    if reason == "p_shape":
        p = p[:, None]
    if reason == "p_device":
        p = torch.ones(3)
    if reason == "old_ixformer":
        monkeypatch.delattr(F, "apply_top_p_radix")
    if reason == "arch":
        monkeypatch.setattr(registry, "current_cuda_arch", lambda: "ivcore20")
    result = impl.apply_top_k_top_p(x, k, p)
    if reason == "no_p":
        assert result is x and not calls
    else:
        assert calls == ["native" if reason == "supported" else "fallback"]


def test_patch_rebinds_v2_caller(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler as community
    from vllm.v1.worker.gpu.sample import states

    from vllm_iluvatar.ilu_patch.general_patches import patch_topk_topp_sampler

    monkeypatch.setattr(
        community, "apply_top_k_top_p", impl._original_apply_top_k_top_p
    )
    monkeypatch.setattr(states, "apply_top_k_top_p", impl._original_apply_top_k_top_p)
    patch_topk_topp_sampler._apply()
    assert states.apply_top_k_top_p is impl.apply_top_k_top_p
    assert community.TopKTopPSampler.forward_native is impl.forward_native
