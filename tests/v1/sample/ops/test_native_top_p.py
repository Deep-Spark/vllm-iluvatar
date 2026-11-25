# SPDX-License-Identifier: Apache-2.0
"""Top-p integration contracts and full random rejection output/RNG parity."""

from types import SimpleNamespace

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.envs import apply
from vllm_iluvatar.v1.sample.ops import topk_topp_sampler as impl

apply()


@pytest.mark.parametrize(
    "reason",
    [
        "enabled",
        "disabled",
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
@requires_corex30
def test_dispatch(monkeypatch, reason):
    import ixformer.inference.functions as F

    from vllm_iluvatar.ilu_util import arch

    monkeypatch.setenv("VLLM_ILU_TOP_P_RADIX", "0" if reason == "disabled" else "1")
    calls = []
    monkeypatch.setattr(
        F, "apply_top_p_radix", lambda *a: calls.append("native") or "native"
    )
    monkeypatch.setattr(
        impl,
        "_original_apply_top_k_top_p",
        lambda *a: calls.append("torch") or "torch",
    )
    x = torch.empty(3, 4097, device="cuda")
    p = torch.ones(3, device="cuda")
    k = None
    if reason == "top_k":
        k = torch.ones(3, device="cuda", dtype=torch.int32)
    if reason == "no_p":
        p = None
    if reason == "small_vocab":
        x = x[:, :4096].contiguous()
    if reason == "empty":
        x = x[:0]
        p = p[:0]
    if reason == "noncontiguous":
        x = torch.empty(3, 8194, device="cuda")[:, ::2]
    if reason == "half":
        x = x.half()
    if reason == "p_shape":
        p = p[:, None]
    if reason == "p_device":
        p = p.cpu()
    if reason == "old_ixformer":
        monkeypatch.delattr(F, "apply_top_p_radix")
    if reason == "arch":
        monkeypatch.setattr(arch, "is_ivcore30", lambda: False)
    result = impl.apply_top_k_top_p(x, k, p)
    assert result == ("native" if reason == "enabled" else "torch")
    assert calls == [result]


@requires_corex30
@pytest.mark.parametrize("p_value", [0.0, 0.5, 0.95, 1.0])
def test_native_mask_matches_exact_nucleus_without_rng(monkeypatch, p_value):
    import ixformer.inference.functions as F

    assert callable(F.apply_top_p_radix)
    torch.manual_seed(32)
    x = torch.randn(3, 8192, device="cuda")
    p = torch.full((3,), p_value, device="cuda")
    state = torch.cuda.get_rng_state()
    expected = impl._original_apply_top_k_top_p(x.clone(), None, p)
    monkeypatch.setenv("VLLM_ILU_TOP_P_RADIX", "1")
    actual = impl.apply_top_k_top_p(x.clone(), None, p)
    # Use FP64 CPU mass as an independent oracle. The 0.31 Triton mask can
    # retain one extra boundary token on this fixture (p=.95, row zero).
    values, ids = x.cpu().double().sort(dim=-1, descending=True)
    cumulative = values.softmax(-1).cumsum(-1)
    keep = cumulative - values.softmax(-1) < p_value
    keep[:, 0] = True
    oracle = torch.zeros_like(keep).scatter_(1, ids, keep)
    assert torch.equal(torch.isfinite(actual).cpu(), oracle)
    changed = torch.isfinite(actual) != torch.isfinite(expected)
    assert bool((changed.sum(-1) <= 1).all())
    assert torch.equal(state, torch.cuda.get_rng_state())


def test_actual_v2_sampling_states_reach_mask(monkeypatch):
    from vllm.v1.worker.gpu.sample import sampler, states

    from vllm_iluvatar.ilu_patch.general_patches import patch_topk_topp_sampler

    patch_topk_topp_sampler.register()
    assert sampler.apply_top_k_top_p is impl.apply_top_k_top_p
    assert states.apply_top_k_top_p is impl.apply_top_k_top_p
    calls = []
    monkeypatch.setenv("VLLM_ILU_TOP_P_RADIX", "0")
    monkeypatch.setattr(
        impl, "_original_apply_top_k_top_p", lambda *a: calls.append(a) or a[0]
    )
    fake = SimpleNamespace(get_top_k_top_p=lambda *a: (None, torch.ones(2)))
    x = torch.ones(2, 8)
    assert states.SamplingStates.apply_top_k_top_p(fake, x, None, None) is x
    assert len(calls) == 1
