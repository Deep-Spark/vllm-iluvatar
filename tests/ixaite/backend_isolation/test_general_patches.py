# SPDX-License-Identifier: Apache-2.0
"""Check patch replacement order, import isolation, and retry after startup failure."""

import importlib
import pkgutil
from types import SimpleNamespace

import pytest

from vllm_iluvatar.backends.ixaite import register as registration
from vllm_iluvatar.ilu_patch import general_patch as p


# Replacements must occupy their original positions without importing baseline patches.
# A failed install must remain retryable, and successful patches must not run twice.
@pytest.mark.parametrize(
    "failure_stage", [None, "package", "registration", "replacement"]
)
def test_selected_patch_order_isolation_and_retry(monkeypatch, failure_stage):
    package_name = "vllm_iluvatar.ilu_patch.general_patches"
    names = [
        "patch_common_before",
        "patch_layer_utils",
        "patch_grouped_topk_router",
        "patch_common_middle",
        "patch_kv_cache_utils",
        "patch_sparse_mla_warmup",
        "patch_stateless_coordinator",
        "patch_fused_moe_layer",
        "patch_common_after",
    ]
    replaced = {
        "patch_layer_utils",
        "patch_grouped_topk_router",
        "patch_sparse_mla_warmup",
        "patch_stateless_coordinator",
        "patch_fused_moe_layer",
        "patch_kv_cache_utils",
    }
    events, imports = [], []
    state = {"failure": failure_stage}
    monkeypatch.setattr(registration, "_general_patches_applied", False)
    monkeypatch.setattr(p, "_APPLIED", set())

    # Fail once before recording an event, so retry proves the remaining order.
    def fail_once(stage):
        if state["failure"] == stage:
            state["failure"] = None
            raise ImportError(f"failed {stage}")

    # Successful selected callbacks remain protected by the real register_once.
    def selected(name):
        # Model a selected implementation that can fail before completing setup.
        def apply():
            if name == "linear":
                fail_once("replacement")
            events.append(name)

        return apply

    for name in ("linear", "moe", "warmup", "stateless", "kv_cache"):
        monkeypatch.setattr(registration, "_apply_" + name, selected(name))

    original_import = importlib.import_module

    # Baseline replacement modules must never be imported, even on retry.
    def import_module(name, package=None):
        if name == package_name:
            imports.append(name)
            fail_once("package")
            return SimpleNamespace(__name__=name, __path__=[])
        if not name.startswith(package_name + "."):
            return original_import(name, package)
        short_name = name.rsplit(".", 1)[-1]
        assert short_name not in replaced, name
        imports.append(name)

        # Shared patches retain their own idempotence and propagate failures.
        def register():
            if short_name == "patch_common_middle":
                fail_once("registration")
            p.register_once(name, lambda: events.append(short_name))

        return SimpleNamespace(register=register)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(
        pkgutil,
        "iter_modules",
        lambda *args: [
            SimpleNamespace(name=package_name + ".ignored_package", ispkg=True),
            *(
                SimpleNamespace(name=package_name + "." + name, ispkg=False)
                for name in names
            ),
        ],
    )
    if failure_stage is not None:
        with pytest.raises(ImportError, match="failed " + failure_stage):
            registration.apply_general_patches()
        assert not registration._general_patches_applied

    registration.apply_general_patches()
    assert events == [
        "patch_common_before",
        "linear",
        "patch_common_middle",
        "kv_cache",
        "warmup",
        "stateless",
        "moe",
        "patch_common_after",
    ]
    assert registration._general_patches_applied
    expected_keys = {
        package_name + "." + name
        for name in names
        if name != "patch_grouped_topk_router"
    }
    assert expected_keys == p._APPLIED
    saved_imports = list(imports)
    registration.apply_general_patches()
    assert imports == saved_imports
    assert len(events) == 8


@pytest.mark.parametrize("error", [ImportError, NameError])
def test_ray_patch_skips_an_unavailable_optional_executor(monkeypatch, error):
    def unavailable(name):
        assert name == "vllm.v1.executor.ray_executor_v2"
        raise error("ray is unavailable")

    monkeypatch.setattr(importlib, "import_module", unavailable)
    registration._apply_ray_executor_v2()
