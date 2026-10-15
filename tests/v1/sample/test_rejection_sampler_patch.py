# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar rejection-sampler patch."""

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType

from vllm_iluvatar.ilu_patch.general_patches import patch_rejection_sampler

_MAIN_DB9527A468_REJECTION_SAMPLE_SHA = (
    "adadcc804bccf89cc3568b637d14b90480f903d0a5de417719200ad6a284c747"
)
_MAIN_DB9527A468_GENERATE_UNIFORM_PROBS_SHA = (
    "016fdca2217c3f2b2ce40b75d94cac8c49fbea92202180cb57a3f481654fd83f"
)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_vllm(monkeypatch) -> ModuleType:
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.Generator = object
    torch.device = object
    torch.float32 = "float32"

    vllm = _package("vllm")
    v1 = _package("vllm.v1")
    sample = _package("vllm.v1.sample")
    community = ModuleType("vllm.v1.sample.rejection_sampler")
    consumer = ModuleType("vllm.v1.sample.consumer")

    def rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
        use_fp64_gumbel=False,
    ):
        return "community-sample"

    def generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        generators,
        device,
    ):
        return "community-uniform"

    def rejection_greedy_sample_kernel():
        return "community-greedy"

    def rejection_random_sample_kernel():
        return "community-random"

    community.rejection_sample = rejection_sample
    community.generate_uniform_probs = generate_uniform_probs
    community.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
    community.rejection_random_sample_kernel = rejection_random_sample_kernel
    consumer.rejection_sample = rejection_sample
    consumer.generate_uniform_probs = generate_uniform_probs
    consumer.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
    consumer.rejection_random_sample_kernel = rejection_random_sample_kernel

    for name, module in {
        "torch": torch,
        "vllm": vllm,
        "vllm.v1": v1,
        "vllm.v1.sample": sample,
        "vllm.v1.sample.rejection_sampler": community,
        "vllm.v1.sample.consumer": consumer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    vllm.v1 = v1
    v1.sample = sample
    sample.rejection_sampler = community
    sample.consumer = consumer
    return community


def _load_impl(monkeypatch) -> ModuleType:
    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.v1.sample.rejection_sampler", raising=False
    )
    return importlib.import_module("vllm_iluvatar.v1.sample.rejection_sampler")


def test_rejection_sampler_fingerprint_baselines_match_declared_main_source():
    assert patch_rejection_sampler._OVERRIDES == ["generate_uniform_probs"]
    assert patch_rejection_sampler._BASELINE_SOURCE_SHAS == {
        "rejection_sample": _MAIN_DB9527A468_REJECTION_SAMPLE_SHA,
        "generate_uniform_probs": _MAIN_DB9527A468_GENERATE_UNIFORM_PROBS_SHA,
    }


def test_rejection_sampler_impl_keeps_only_float32_uniform_delta(monkeypatch):
    _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)
    uniform_source = inspect.getsource(impl.generate_uniform_probs)

    assert not hasattr(impl, "rejection_sample")
    assert not hasattr(impl, "rejection_greedy_sample_kernel")
    assert not hasattr(impl, "rejection_random_sample_kernel")
    assert not hasattr(impl, "sample_recovered_tokens_kernel")
    assert "# >>> ILU MODIFICATION >>>" in uniform_source
    assert "dtype=torch.float32" in uniform_source


def test_patch_apply_rebinds_uniform_probs_only(monkeypatch):
    community = _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    import vllm.v1.sample.consumer as consumer

    old_sample = community.rejection_sample
    old_uniform = community.generate_uniform_probs
    old_greedy = community.rejection_greedy_sample_kernel
    old_random = community.rejection_random_sample_kernel
    assert consumer.rejection_sample is old_sample
    assert consumer.generate_uniform_probs is old_uniform

    monkeypatch.setattr(
        patch_rejection_sampler,
        "_check_source_sha",
        lambda *args, **kwargs: None,
    )

    patch_rejection_sampler._apply()

    assert community.generate_uniform_probs is impl.generate_uniform_probs
    assert consumer.generate_uniform_probs is impl.generate_uniform_probs
    assert community.rejection_sample is old_sample
    assert consumer.rejection_sample is old_sample
    assert community.rejection_greedy_sample_kernel is old_greedy
    assert community.rejection_random_sample_kernel is old_random
    assert consumer.rejection_greedy_sample_kernel is old_greedy
    assert consumer.rejection_random_sample_kernel is old_random
