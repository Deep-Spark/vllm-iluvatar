# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar spec-decode utils patch."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_spec_decode_utils

_MAIN_6E448D0_EAGLE_NEXT_TOKEN_PADDED_KERNEL_SHA = (
    "930294908083bb508b264eb1823404268160ef8b2f8b0a812fe2126b32ae8d8c"
)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_vllm(monkeypatch) -> ModuleType:
    vllm = _package("vllm")
    triton_utils = ModuleType("vllm.triton_utils")
    v1 = _package("vllm.v1")
    spec_decode = _package("vllm.v1.spec_decode")
    community = ModuleType("vllm.v1.spec_decode.utils")
    consumer = ModuleType("vllm.v1.spec_decode.llm_base_proposer")

    class FakeTriton:
        @staticmethod
        def jit(fn):
            return fn

    fake_tl = SimpleNamespace(
        constexpr=object,
        int32="int32",
        uint32="uint32",
    )

    def eagle_prepare_next_token_padded_kernel(
        sampled_token_ids_ptr,
        discard_request_mask_ptr,
        backup_next_token_ids_ptr,
        next_token_ids_ptr,
        valid_sampled_tokens_count_ptr,
        vocab_size,
        num_sampled_tokens_per_req,
        num_reqs,
        stride_sampled_token_ids,
        BLOCK_SIZE_TOKENS,
    ):
        return "community"

    community.eagle_prepare_next_token_padded_kernel = (
        eagle_prepare_next_token_padded_kernel
    )
    consumer.eagle_prepare_next_token_padded_kernel = (
        eagle_prepare_next_token_padded_kernel
    )
    triton_utils.tl = fake_tl
    triton_utils.triton = FakeTriton()

    for name, module in {
        "vllm": vllm,
        "vllm.triton_utils": triton_utils,
        "vllm.v1": v1,
        "vllm.v1.spec_decode": spec_decode,
        "vllm.v1.spec_decode.utils": community,
        "vllm.v1.spec_decode.llm_base_proposer": consumer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    vllm.triton_utils = triton_utils
    vllm.v1 = v1
    v1.spec_decode = spec_decode
    spec_decode.utils = community
    spec_decode.llm_base_proposer = consumer
    return community


def _load_impl(monkeypatch) -> ModuleType:
    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.v1.spec_decode.utils", raising=False
    )
    return importlib.import_module("vllm_iluvatar.v1.spec_decode.utils")


def test_eagle_kernel_fingerprint_baseline_matches_declared_main_source():
    assert (
        p.source_sha(
            patch_spec_decode_utils._EAGLE_PREPARE_NEXT_TOKEN_PADDED_KERNEL_BASELINE
        )
        == _MAIN_6E448D0_EAGLE_NEXT_TOKEN_PADDED_KERNEL_SHA
    )


def test_eagle_kernel_impl_marks_ilu_dtype_delta(monkeypatch):
    import inspect

    _install_fake_vllm(monkeypatch)
    impl_module = _load_impl(monkeypatch)

    impl_source = inspect.getsource(impl_module.eagle_prepare_next_token_padded_kernel)
    assert "# >>> ILU MODIFICATION >>>" in impl_source
    assert "tl.full((), 0, dtype=tl.int32)" in impl_source
    assert "tl.sum(is_valid_mask.to(tl.int32)).to(tl.int32)" in impl_source


def test_patch_apply_rebinds_kernel_and_from_import_consumer(monkeypatch):
    community = _install_fake_vllm(monkeypatch)
    impl = _load_impl(monkeypatch)

    import vllm.v1.spec_decode.llm_base_proposer as consumer

    old = community.eagle_prepare_next_token_padded_kernel
    assert consumer.eagle_prepare_next_token_padded_kernel is old

    monkeypatch.setattr(
        patch_spec_decode_utils.p,
        "check_fingerprint",
        lambda *args, **kwargs: None,
    )

    patch_spec_decode_utils._apply()

    assert community.eagle_prepare_next_token_padded_kernel is (
        impl.eagle_prepare_next_token_padded_kernel
    )
    assert consumer.eagle_prepare_next_token_padded_kernel is (
        impl.eagle_prepare_next_token_padded_kernel
    )
