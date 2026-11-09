# SPDX-License-Identifier: Apache-2.0
"""Self-checks for the assertions used by strict functional tests."""

from contextlib import suppress
from types import ModuleType

import pytest
import torch

from .ixformer_guard import forbid_ixformer_calls


# Check saved Python references fail and are restored after exceptional exit.
def test_python_guard_restores_facade():
    vendor = ModuleType("ixformer.inference.functions")
    vendor.launch = lambda: "original"
    original = vendor.launch
    with (
        pytest.raises(AssertionError, match="ixformer call"),
        forbid_ixformer_calls([vendor]),
    ):
        copied = vendor.launch
        copied()
    assert vendor.launch is original
    assert vendor.launch() == "original"


# Check dispatcher calls fail before their implementation can run.
def test_dispatch_guard_rejects_native_namespace():
    library = torch.library.Library("_ixformer_C", "FRAGMENT")
    try:
        library.define("backend_isolation_probe(Tensor x) -> Tensor")
        library.impl("backend_isolation_probe", lambda x: x + 1, "CPU")
        x = torch.tensor([1])
        with (
            pytest.raises(AssertionError, match="ixformer op"),
            forbid_ixformer_calls([]),
        ):
            assert torch.equal(x + 1, torch.tensor([2]))
            torch.ops._ixformer_C.backend_isolation_probe(x)
        assert torch.equal(
            torch.ops._ixformer_C.backend_isolation_probe(x), torch.tensor([2])
        )
    finally:
        library._destroy()


# Even a swallowed warmup exception must make strict validation fail.
def test_caught_vendor_call_still_fails():
    vendor = ModuleType("ixformer._C.infer")
    vendor.repack = lambda: None
    with (
        pytest.raises(AssertionError, match="repack"),
        forbid_ixformer_calls([vendor]),
        suppress(AssertionError),
    ):
        vendor.repack()
