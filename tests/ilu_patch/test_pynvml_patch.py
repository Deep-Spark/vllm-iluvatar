# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the bundled pynvml CDLL redirect patch."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_patch_pynvml(monkeypatch):
    import_hook = ModuleType("vllm_iluvatar.ilu_patch.import_hook")
    import_hook.registrations = []

    def register(target_fullname, callback):
        import_hook.registrations.append((target_fullname, callback))

    import_hook.register = register
    monkeypatch.setitem(sys.modules, import_hook.__name__, import_hook)
    sys.modules.pop("vllm_iluvatar.ilu_patch.patch_pynvml", None)
    return importlib.import_module("vllm_iluvatar.ilu_patch.patch_pynvml")


def _build_fake_pynvml_module():
    calls: list[str] = []

    class FakeCDLL:
        def __init__(self, name, *args, **kwargs):
            calls.append(name)

    return SimpleNamespace(CDLL=FakeCDLL), calls


def test_pynvml_patch_redirects_nvml_soname_to_ixml(monkeypatch):
    patch_pynvml = _load_patch_pynvml(monkeypatch)
    module, calls = _build_fake_pynvml_module()

    patch_pynvml._patch_pynvml(module)
    module.CDLL("libnvidia-ml.so.1")

    assert calls == ["libixml.so"]


def test_pynvml_patch_leaves_other_libraries_unchanged(monkeypatch):
    patch_pynvml = _load_patch_pynvml(monkeypatch)
    module, calls = _build_fake_pynvml_module()

    patch_pynvml._patch_pynvml(module)
    module.CDLL("libcuda.so.1")

    assert calls == ["libcuda.so.1"]


def test_pynvml_patch_is_idempotent(monkeypatch):
    patch_pynvml = _load_patch_pynvml(monkeypatch)
    module, _ = _build_fake_pynvml_module()

    patch_pynvml._patch_pynvml(module)
    first_cdll = module.CDLL
    patch_pynvml._patch_pynvml(module)

    assert module.CDLL is first_cdll
