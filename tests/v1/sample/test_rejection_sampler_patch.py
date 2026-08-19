# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar rejection-sampler patch."""

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_rejection_sampler

_MAIN_6E448D0_REJECTION_GREEDY_SAMPLE_KERNEL_SHA = (
    "9cd5a8f8fe5a065937797b59a1fede6aadb711391d3288a09fc7ffa65cfc5551"
)
_MAIN_6E448D0_REJECTION_RANDOM_SAMPLE_KERNEL_SHA = (
    "63a939862a8810443cdd667248e16b8aba6c33e0f921812b5b6a34cc597d2bad"
)
_MAIN_6E448D0_REJECTION_SAMPLE_SHA = (
    "35d7236c81a06800520691e8eb79014b33653e05ce5e982901fdb68ea003a1ea"
)
_MAIN_6E448D0_GENERATE_UNIFORM_PROBS_SHA = (
    "f3e8fbcf4f6e8332bbe61141fad916087b61256f4dada65663d274b48e27524b"
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
        "rejection_sample": _MAIN_6E448D0_REJECTION_SAMPLE_SHA,
        "generate_uniform_probs": _MAIN_6E448D0_GENERATE_UNIFORM_PROBS_SHA,
        "rejection_greedy_sample_kernel": (
            _MAIN_6E448D0_REJECTION_GREEDY_SAMPLE_KERNEL_SHA
        ),
        "rejection_random_sample_kernel": (
            _MAIN_6E448D0_REJECTION_RANDOM_SAMPLE_KERNEL_SHA
        ),
    }


def test_rejection_sampler_impl_marks_ilu_predicate_delta(monkeypatch):
    _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    greedy_source = inspect.getsource(impl.rejection_greedy_sample_kernel)
    random_source = inspect.getsource(impl.rejection_random_sample_kernel)
    rejection_source = inspect.getsource(impl.rejection_sample)
    uniform_source = inspect.getsource(impl.generate_uniform_probs)

    assert "# >>> ILU MODIFICATION >>>" in greedy_source
    assert "# >>> ILU MODIFICATION >>>" in random_source
    assert "# >>> ILU MODIFICATION >>>" in rejection_source
    assert "# >>> ILU MODIFICATION >>>" in uniform_source
    assert "tl.load(is_greedy_ptr + req_idx).to(tl.int32) != 0" in greedy_source
    assert "tl.load(is_greedy_ptr + req_idx).to(tl.int32) != 0" in random_source
    assert "rejection_sample_deterministic_draft_torch" in rejection_source
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


def test_rejection_sample_draft_probs_none_uses_deterministic_fallback(monkeypatch):
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

    output_token_ids = object()
    fallback_calls = []

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
    monkeypatch.setattr(
        impl,
        "rejection_sample_deterministic_draft_torch",
        lambda *args, **kwargs: fallback_calls.append(args),
    )

    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature="temperature",
        generators={},
    )

    result = impl.rejection_sample(
        FakeTensor(ndim=1, shape=(1,)),
        [1],
        2,
        FakeTensor(ndim=1, shape=(1,)),
        None,
        FakeTargetLogits(ndim=2, shape=(1, 4), device="corex"),
        FakeTensor(ndim=2, shape=(1, 1)),
        sampling_metadata,
    )

    assert result is output_token_ids
    assert len(fallback_calls) == 1
    assert fallback_calls[0][0] is output_token_ids


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
    assert community.rejection_greedy_sample_kernel is (
        impl.rejection_greedy_sample_kernel
    )
    assert community.rejection_random_sample_kernel is (
        impl.rejection_random_sample_kernel
    )
    assert consumer.rejection_sample is impl.rejection_sample
    assert consumer.generate_uniform_probs is impl.generate_uniform_probs
    assert consumer.rejection_greedy_sample_kernel is impl.rejection_greedy_sample_kernel
    assert consumer.rejection_random_sample_kernel is impl.rejection_random_sample_kernel
