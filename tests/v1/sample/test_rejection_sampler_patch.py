# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar rejection-sampler patch."""

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace

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
    torch.int32 = "int32"
    torch.float32 = "float32"
    torch.long = "long"

    vllm = _package("vllm")
    triton_utils = ModuleType("vllm.triton_utils")
    v1 = _package("vllm.v1")
    sample = _package("vllm.v1.sample")
    metadata = ModuleType("vllm.v1.sample.metadata")
    community = ModuleType("vllm.v1.sample.rejection_sampler")
    consumer = ModuleType("vllm.v1.sample.consumer")
    metadata.SamplingMetadata = object

    class FakeTriton:
        @staticmethod
        def jit(fn=None, **_kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

    fake_tl = SimpleNamespace(
        constexpr=object,
        int32="int32",
    )

    def rejection_greedy_sample_kernel(
        output_token_ids_ptr,
        cu_num_draft_tokens_ptr,
        draft_token_ids_ptr,
        target_argmax_ptr,
        bonus_token_ids_ptr,
        is_greedy_ptr,
        max_spec_len,
        uniform_probs_ptr,
        synthetic_conditional_rates_ptr,
        SYNTHETIC_MODE,
    ):
        return "community-greedy"

    def rejection_random_sample_kernel(
        output_token_ids_ptr,
        cu_num_draft_tokens_ptr,
        draft_token_ids_ptr,
        draft_probs_ptr,
        target_probs_ptr,
        bonus_token_ids_ptr,
        recovered_token_ids_ptr,
        uniform_probs_ptr,
        is_greedy_ptr,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates_ptr,
        NO_DRAFT_PROBS,
        SYNTHETIC_MODE,
    ):
        return "community-random"

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

    community.GREEDY_TEMPERATURE = 0
    community.PLACEHOLDER_TOKEN_ID = -1
    community.sample_recovered_tokens = lambda *args, **kwargs: "recovered"
    community.rejection_sample = rejection_sample
    community.generate_uniform_probs = generate_uniform_probs
    community.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
    community.rejection_random_sample_kernel = rejection_random_sample_kernel
    consumer.rejection_sample = rejection_sample
    consumer.generate_uniform_probs = generate_uniform_probs
    consumer.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
    consumer.rejection_random_sample_kernel = rejection_random_sample_kernel
    triton_utils.tl = fake_tl
    triton_utils.triton = FakeTriton()

    for name, module in {
        "torch": torch,
        "vllm": vllm,
        "vllm.triton_utils": triton_utils,
        "vllm.v1": v1,
        "vllm.v1.sample": sample,
        "vllm.v1.sample.metadata": metadata,
        "vllm.v1.sample.rejection_sampler": community,
        "vllm.v1.sample.consumer": consumer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    vllm.triton_utils = triton_utils
    vllm.v1 = v1
    v1.sample = sample
    sample.metadata = metadata
    sample.rejection_sampler = community
    sample.consumer = consumer
    return community


def _load_impl(monkeypatch) -> ModuleType:
    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.v1.sample.rejection_sampler", raising=False
    )
    return importlib.import_module("vllm_iluvatar.v1.sample.rejection_sampler")


def test_rejection_sampler_fingerprint_baselines_match_declared_main_source():
    assert patch_rejection_sampler._BASELINE_SOURCE_SHAS == {
        "rejection_sample": _MAIN_DB9527A468_REJECTION_SAMPLE_SHA,
        "generate_uniform_probs": _MAIN_DB9527A468_GENERATE_UNIFORM_PROBS_SHA,
    }


def test_rejection_sampler_impl_keeps_only_float32_uniform_delta(monkeypatch):
    _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    rejection_source = inspect.getsource(impl.rejection_sample)
    uniform_source = inspect.getsource(impl.generate_uniform_probs)

    assert not hasattr(impl, "rejection_greedy_sample_kernel")
    assert not hasattr(impl, "rejection_random_sample_kernel")
    assert not hasattr(impl, "sample_recovered_tokens_kernel")
    assert "# >>> ILU MODIFICATION >>>" in rejection_source
    assert "# >>> ILU MODIFICATION >>>" in uniform_source
    assert "_community.rejection_greedy_sample_kernel[" in rejection_source
    assert "_community.rejection_random_sample_kernel[" in rejection_source
    assert "_community.sample_recovered_tokens(" in rejection_source
    assert "NO_DRAFT_PROBS=True" in rejection_source
    assert "dtype=torch.float32" in uniform_source


def test_rejection_sample_delegates_to_community_when_draft_probs_exist(monkeypatch):
    _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)
    calls = []

    def community_rejection_sample(*args, **kwargs):
        calls.append((args, kwargs))
        return "community-result"

    monkeypatch.setattr(impl, "_community_rejection_sample", community_rejection_sample)

    result = impl.rejection_sample(
        "draft_token_ids",
        [1],
        2,
        "cu_num_draft_tokens",
        "draft_probs",
        "target_logits",
        "bonus_token_ids",
        "sampling_metadata",
        synthetic_mode=True,
        synthetic_conditional_rates="rates",
        use_fp64_gumbel=True,
    )

    assert result == "community-result"
    assert calls == [
        (
            (
                "draft_token_ids",
                [1],
                2,
                "cu_num_draft_tokens",
                "draft_probs",
                "target_logits",
                "bonus_token_ids",
                "sampling_metadata",
            ),
            {
                "synthetic_mode": True,
                "synthetic_conditional_rates": "rates",
                "use_fp64_gumbel": True,
            },
        )
    ]


def test_rejection_sample_draft_probs_none_uses_triton_kernels(monkeypatch):
    _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    class FakeTensor:
        def __init__(self, *, ndim=1, shape=(1,), device="cpu"):
            self.ndim = ndim
            self.shape = shape
            self.device = device

        def is_contiguous(self):
            return True

    class FakeTargetLogits(FakeTensor):
        def argmax(self, dim):
            return "target-argmax"

        def softmax(self, dim, dtype):
            return FakeTensor(ndim=2, shape=self.shape, device=self.device)

    class GridKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return launch

    output_token_ids = object()
    recovered = object()
    random_kernel = GridKernel()
    recovered_calls = []

    monkeypatch.setattr(
        impl.torch,
        "full",
        lambda *args, **kwargs: output_token_ids,
        raising=False,
    )
    monkeypatch.setattr(
        impl,
        "generate_uniform_probs",
        lambda *args, **kwargs: "uniform-probs",
    )

    def sample_recovered_tokens(*args, **kwargs):
        recovered_calls.append(args)
        return recovered

    monkeypatch.setattr(
        impl._community, "sample_recovered_tokens", sample_recovered_tokens
    )
    monkeypatch.setattr(
        impl._community, "rejection_random_sample_kernel", random_kernel
    )

    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature="temperature",
        generators={},
    )
    draft_token_ids = FakeTensor(ndim=1, shape=(1,))
    cu_num_draft_tokens = FakeTensor(ndim=1, shape=(1,))
    target_logits = FakeTargetLogits(ndim=2, shape=(1, 4), device="corex")
    bonus_token_ids = FakeTensor(ndim=2, shape=(1, 1))

    result = impl.rejection_sample(
        draft_token_ids,
        [1],
        2,
        cu_num_draft_tokens,
        None,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
    )

    assert result is output_token_ids
    assert len(recovered_calls) == 1
    assert recovered_calls[0][0] == 2
    assert recovered_calls[0][2] is cu_num_draft_tokens
    assert len(random_kernel.calls) == 1
    grid, args, kwargs = random_kernel.calls[0]
    assert grid == (1,)
    assert args[0] is output_token_ids
    assert args[6] is recovered
    assert args[7] == "uniform-probs"
    assert kwargs["NO_DRAFT_PROBS"] is True
    assert kwargs["SYNTHETIC_MODE"] is False


def test_patch_apply_rebinds_rejection_sampler_overrides(monkeypatch):
    community = _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    import vllm.v1.sample.consumer as consumer

    old_sample = community.rejection_sample
    old_uniform = community.generate_uniform_probs
    old_greedy = community.rejection_greedy_sample_kernel
    old_random = community.rejection_random_sample_kernel
    assert consumer.rejection_sample is old_sample
    assert consumer.generate_uniform_probs is old_uniform
    assert consumer.rejection_greedy_sample_kernel is old_greedy
    assert consumer.rejection_random_sample_kernel is old_random

    monkeypatch.setattr(
        patch_rejection_sampler,
        "_check_source_sha",
        lambda *args, **kwargs: None,
    )

    patch_rejection_sampler._apply()

    assert impl._community_rejection_sample is old_sample
    assert community.rejection_sample is impl.rejection_sample
    assert community.generate_uniform_probs is impl.generate_uniform_probs
    assert community.rejection_greedy_sample_kernel is old_greedy
    assert community.rejection_random_sample_kernel is old_random
    assert consumer.rejection_sample is impl.rejection_sample
    assert consumer.generate_uniform_probs is impl.generate_uniform_probs
    assert consumer.rejection_greedy_sample_kernel is old_greedy
    assert consumer.rejection_random_sample_kernel is old_random
