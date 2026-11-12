# SPDX-License-Identifier: Apache-2.0
"""Model sampling integration: generator advancement, outputs and fallbacks."""

from types import SimpleNamespace

import pytest
import torch
from vllm import envs as opt
from vllm.v1.sample.ops import topk_topp_sampler as upstream

from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.v1.sample.ops import topk_topp_sampler as impl

# GLM CQ 采样 patch 的 Scope 是 ivcore30（见 patch_topk_topp_sampler.py）；
# ivcore11 机器上 forward_native 恒回退上游，且上游新版不再接受广播 p，
# 这组用例只能在 CQ 机器上验证。
pytestmark = pytest.mark.skipif(
    not arch.is_ivcore30(),
    reason="requires ivcore30 (CQ); GLM sampling override scope",
)


@pytest.mark.parametrize("mode", ["sorted", "radix"])
@pytest.mark.parametrize("seeded", ["none", "mixed", "all"])
def test_sampling_generators(monkeypatch, mode, seeded):
    monkeypatch.setattr(opt, "VLLM_ILU_GLM_SAMPLING_MODE", mode)
    import ixformer.inference.functions as ixf_ops

    op_name = f"sampling_{mode}_top_p"
    native = getattr(ixf_ops, op_name)
    noise_inputs = []

    def checked(*args, **kwargs):
        noise_inputs.append(args[2] if mode == "sorted" else args[1])
        return native(*args, **kwargs)

    monkeypatch.setattr(ixf_ops, op_name, checked)
    logits = torch.randn(3, 8193, device="cuda") * 2
    p = torch.tensor([0.8, 0.95, 1.0], device="cuda")
    sampler = SimpleNamespace(logprobs_mode="raw_logprobs", use_fp64_gumbel=False)
    rows = [] if seeded == "none" else [1] if seeded == "mixed" else [0, 1, 2]
    generators = {i: torch.Generator(device="cuda").manual_seed(72 + i) for i in rows}
    initial = {i: g.get_state() for i, g in generators.items()}
    torch.cuda.manual_seed(815)
    global_initial = torch.cuda.get_rng_state()
    masked = upstream.apply_top_k_top_p_pytorch(logits.clone(), None, p)
    expected = upstream.random_sample(masked.softmax(-1), generators)
    global_final = torch.cuda.get_rng_state()
    final = {i: g.get_state() for i, g in generators.items()}
    torch.cuda.set_rng_state(global_initial)
    for i, g in generators.items():
        g.set_state(initial[i])
    actual, processed = impl.forward_native(sampler, logits, generators, None, p)
    assert processed is None
    assert len(noise_inputs) == 1
    assert (noise_inputs[0] is None) == (seeded == "none")
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), global_final)
    for i, g in generators.items():
        assert torch.equal(g.get_state(), final[i])


@pytest.mark.parametrize(
    "reason", ["off", "cpu", "fp64", "processed", "topk", "no_p", "stride"]
)
def test_fallback_preserves_upstream_contract(monkeypatch, reason):
    monkeypatch.setattr(
        opt, "VLLM_ILU_GLM_SAMPLING_MODE", "off" if reason == "off" else "sorted"
    )
    x = torch.zeros(2, 32, device="cpu" if reason == "cpu" else "cuda")
    if reason == "stride":
        x = x[:, ::2]
    sampler = SimpleNamespace(
        logprobs_mode="processed_logprobs" if reason == "processed" else "raw_logprobs",
        use_fp64_gumbel=reason == "fp64",
    )
    k = torch.tensor([3, 3], device=x.device) if reason == "topk" else None
    p = None if reason == "no_p" else torch.ones(2, device=x.device)
    calls = []
    monkeypatch.setattr(
        impl,
        "_original_forward_native",
        lambda *a: calls.append(a) or ("original", "processed"),
    )
    assert impl.forward_native(sampler, x, {}, k, p) == ("original", "processed")
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["sorted", "radix"])
def test_broadcast_top_p_uses_upstream(monkeypatch, mode):
    monkeypatch.setattr(opt, "VLLM_ILU_GLM_SAMPLING_MODE", mode)
    logits = torch.randn(3, 257, device="cuda")
    p = torch.tensor([0.9], device="cuda")
    sampler = SimpleNamespace(logprobs_mode="raw_logprobs", use_fp64_gumbel=False)
    state = torch.cuda.get_rng_state()
    expected = impl._original_forward_native(
        sampler, logits.clone(), {}, None, p.expand(logits.shape[0]).contiguous()
    )
    expected_state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    actual = impl.forward_native(sampler, logits.clone(), {}, None, p)
    assert torch.equal(actual[0], expected[0])
    assert actual[1] is expected[1] is None
    assert torch.equal(torch.cuda.get_rng_state(), expected_state)
