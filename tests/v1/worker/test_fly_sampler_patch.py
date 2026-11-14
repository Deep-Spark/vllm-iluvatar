# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the plugin-owned MRV2 verifier and its native fallback."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as community

from vllm_iluvatar.v1.worker.gpu.spec_decode import rejection_sampler as plugin


def _verifier(method):
    config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=5,
        draft_sample_method="probabilistic",
        rejection_sample_method=method,
        fly_window_size=2,
        fly_min_target_probability=0.01 if method == "fly" else 0.0,
    )
    sampler = SimpleNamespace(
        use_fp64_gumbel=False,
        apply_sampling_params=lambda *args: "processed logits",
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu="temperature"),
            seeds=SimpleNamespace(gpu="seeds"),
        ),
    )
    return plugin.RejectionSampler(sampler, config, torch.device("cpu"))


def test_sampler_is_rebound_before_community_importers_capture_it():
    assert community.RejectionSampler is plugin.RejectionSampler
    # Upstream chunking and logprob implementations remain inherited.
    for name in ("_verify_in_chunks", "_get_logprobs_tensors"):
        assert getattr(plugin.RejectionSampler, name) is getattr(
            plugin.CommunityRejectionSampler, name
        )


def test_standard_verification_delegates_to_the_community_implementation(monkeypatch):
    verifier = _verifier("standard")
    args = tuple(object() for _ in range(10))
    result = object()
    calls = []

    def native_verify(self, *received):
        calls.append((self, received))
        return result

    monkeypatch.setattr(plugin.CommunityRejectionSampler, "_verify", native_verify)
    assert verifier._verify(*args) is result
    assert calls == [(verifier, args)]


@pytest.mark.parametrize("floor", [0.0, 0.01])
def test_fly_uses_processed_logits_and_passes_the_probability_gate(monkeypatch, floor):
    verifier = _verifier("fly")
    verifier.fly_min_target_probability = floor
    args = tuple(object() for _ in range(10))
    calls = []

    def fly_verify(*received, **kwargs):
        calls.append((received, kwargs))
        return "sampled", "count"

    monkeypatch.setattr(plugin, "rejection_sample", fly_verify)
    assert verifier._verify(*args) == ("processed logits", "sampled", "count")
    received, kwargs = calls[0]
    assert received[0] == "processed logits"
    assert received[1:3] == args[1:3]
    assert received[8:11] == ("temperature", "seeds", 5)
    assert kwargs == {
        "use_fp64": False,
        "use_block_verification": False,
        "fly_window_size": 2,
        "fly_entropy_threshold": 0.3,
        "fly_entropy_top_k": 3,
        "fly_min_target_probability": floor,
    }
