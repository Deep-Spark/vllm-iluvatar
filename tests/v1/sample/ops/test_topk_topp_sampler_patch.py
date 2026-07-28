# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the top-k/top-p sampler general patch.

The implementation imports ``torch`` and the paired community sampler module,
but this patch's ownership is pure Python wiring and branch selection. The tests
therefore inject minimal fake modules instead of requiring a real CoreX runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


_IMPL_NAME = "vllm_iluvatar.v1.sample.ops.topk_topp_sampler"
_IMPL_PATH = (
    Path(__file__).parents[4]
    / "vllm_iluvatar/v1/sample/ops/topk_topp_sampler.py"
)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_vllm(monkeypatch, *, is_cpu: bool) -> ModuleType:
    torch = ModuleType("torch")
    torch.Tensor = object

    vllm = _package("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_cpu=lambda: is_cpu)
    v1 = _package("vllm.v1")
    sample = _package("vllm.v1.sample")
    ops = _package("vllm.v1.sample.ops")
    community = ModuleType("vllm.v1.sample.ops.topk_topp_sampler")

    def apply_top_k_top_p(logits, k, p):
        return "community"

    community.HAS_TRITON = True
    community.apply_top_k_top_p = apply_top_k_top_p
    community.apply_top_k_top_p_triton = lambda *args, **kwargs: "triton"
    community.apply_top_k_top_p_pytorch = lambda *args, **kwargs: "pytorch"

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample", sample)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.ops", ops)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.sample.ops.topk_topp_sampler", community
    )

    vllm.platforms = platforms
    vllm.v1 = v1
    v1.sample = sample
    sample.ops = ops
    ops.topk_topp_sampler = community

    return community


def _load_impl(monkeypatch, *, is_cpu: bool = False) -> ModuleType:
    _install_fake_vllm(monkeypatch, is_cpu=is_cpu)
    monkeypatch.delitem(sys.modules, _IMPL_NAME, raising=False)

    spec = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
    assert spec is not None
    assert spec.loader is not None
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    monkeypatch.setitem(sys.modules, _IMPL_NAME, impl)
    import vllm_iluvatar.v1.sample.ops as ops_pkg

    monkeypatch.setattr(ops_pkg, "topk_topp_sampler", impl, raising=False)
    return impl


def test_ilu_override_gpu_uses_pytorch_fallback(monkeypatch):
    impl = _load_impl(monkeypatch, is_cpu=False)

    calls = []

    def triton(*args, **kwargs):
        calls.append(("triton", args, kwargs))
        return "triton"

    def pytorch(*args, **kwargs):
        calls.append(("pytorch", args, kwargs))
        return "pytorch"

    monkeypatch.setattr(
        impl, "current_platform", SimpleNamespace(is_cpu=lambda: False)
    )
    monkeypatch.setattr(impl._community, "HAS_TRITON", True)
    monkeypatch.setattr(impl._community, "apply_top_k_top_p_triton", triton)
    monkeypatch.setattr(impl._community, "apply_top_k_top_p_pytorch", pytorch)

    logits = SimpleNamespace(shape=(8, 1024))

    assert impl.apply_top_k_top_p(logits, "k", "p") == "pytorch"
    assert calls == [("pytorch", (logits, "k", "p"), {})]


def test_ilu_override_preserves_cpu_triton_path(monkeypatch):
    impl = _load_impl(monkeypatch, is_cpu=True)

    calls = []

    def triton(*args, **kwargs):
        calls.append(("triton", args, kwargs))
        return "triton"

    def pytorch(*args, **kwargs):
        calls.append(("pytorch", args, kwargs))
        return "pytorch"

    monkeypatch.setattr(
        impl, "current_platform", SimpleNamespace(is_cpu=lambda: True)
    )
    monkeypatch.setattr(impl._community, "HAS_TRITON", True)
    monkeypatch.setattr(impl._community, "apply_top_k_top_p_triton", triton)
    monkeypatch.setattr(impl._community, "apply_top_k_top_p_pytorch", pytorch)

    logits = SimpleNamespace(shape=(8, 1024))

    assert impl.apply_top_k_top_p(logits, "k", "p") == "triton"
    assert calls == [("triton", (logits, "k", "p"), {})]


def test_patch_apply_rebinds_community_function(monkeypatch):
    impl = _load_impl(monkeypatch, is_cpu=False)
    community = sys.modules["vllm.v1.sample.ops.topk_topp_sampler"]
    from vllm_iluvatar.ilu_patch.general_patches import patch_topk_topp_sampler

    patch_topk_topp_sampler._apply()

    assert community.apply_top_k_top_p is impl.apply_top_k_top_p
