# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Check shared launch controls without extending the original platform tests."""

import importlib
import os
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest

vllm_envs = importlib.import_module("vllm.envs")


# Private environment defaults must not change the baseline process settings.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_apply_scopes_nccl_defaults_to_ixaite(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    envs = importlib.import_module("vllm_iluvatar.envs")
    monkeypatch.setattr(
        vllm_envs,
        "environment_variables",
        dict(vllm_envs.environment_variables),
    )
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)

    envs.apply()
    assert os.environ.get("NCCL_P2P_DISABLE") is None
    expected_ib = "1" if mode == "ixaite" else None
    assert os.environ.get("NCCL_IB_DISABLE") == expected_ib

    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    envs.apply()
    assert os.environ["NCCL_P2P_DISABLE"] == "0"


# Peer access is disabled only when the final configuration spans multiple nodes.
@pytest.mark.parametrize("nnodes,expected", [(1, None), (2, "1")])
def test_peer_access_follows_node_count(monkeypatch, nnodes, expected):
    from vllm_iluvatar.backends.ixaite.envs import configure_peer_access

    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    configure_peer_access(nnodes)
    assert os.environ.get("NCCL_P2P_DISABLE") == expected


# A user's explicit communication setting takes precedence over the default.
def test_peer_access_keeps_an_explicit_value(monkeypatch):
    from vllm_iluvatar.backends.ixaite.envs import configure_peer_access

    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    configure_peer_access(2)
    assert os.environ["NCCL_P2P_DISABLE"] == "0"


# Private backend paths must replace the shared paths before platform hooks finish.
@pytest.mark.parametrize(
    "mode,renamed_fp8", [("baseline", False), ("ixaite", False), ("ixaite", True)]
)
def test_platform_delegation_keeps_registration_order(monkeypatch, mode, renamed_fp8):
    import vllm_iluvatar as plugin

    calls = []
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    monkeypatch.setenv(plugin.STARTUP_PATCH_ENV, "1")
    ixformer = ModuleType("ixformer")
    inference = ModuleType("ixformer.inference")
    functions = ModuleType("ixformer.inference.functions")
    fp8 = object()
    functions.flash_attn_unpad_fp8 = fp8 if renamed_fp8 else object()
    if not renamed_fp8:
        functions.flash_attn_varlen_fp8_func = fp8
    ixformer.inference = inference
    inference.functions = functions
    for module in (ixformer, inference, functions):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        plugin, "_load_import_compat_patch", lambda: calls.append("compat")
    )
    for module_name, name, event in (
        ("vllm_iluvatar.envs", "apply", "envs"),
        (
            "vllm_iluvatar.platforms.ilu_platform",
            "register_attention_backends",
            "baseline",
        ),
        (
            "vllm_iluvatar.backends.ixaite.attention.register",
            "register_attention_backends",
            "ixaite",
        ),
        (
            "vllm_iluvatar.ilu_patch.patch_platform_import_hooks",
            "register_platform_import_hooks",
            "hooks",
        ),
    ):
        module = ModuleType(module_name)
        setattr(module, name, lambda event=event: calls.append(event))
        monkeypatch.setitem(sys.modules, module_name, module)
    path = plugin.register_platform()
    if mode == "ixaite":
        assert functions.flash_attn_varlen_fp8_func is fp8
    # The private registry must run after common registration and before hooks.
    expected = ["compat", "envs", "baseline"]
    if mode == "ixaite":
        expected.append("ixaite")
    assert calls == [*expected, "hooks"]
    assert path == (
        "vllm_iluvatar.backends.ixaite.platform.IxaitePlatform"
        if mode == "ixaite"
        else "vllm_iluvatar.platforms.ilu_platform.IluPlatform"
    )


# Check stateless providers are fixed before OOT classes can be instantiated.
# Check baseline avoids ixaite startup while both modes retain patch/OOT/KV order.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_register_general_orders_selected_startup(monkeypatch, mode):
    import vllm_iluvatar as plugin
    from vllm_iluvatar.backends.ixaite import register as registration

    calls = []
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    for name, event in (
        ("apply_general_patches", "ixaite_patches"),
        ("configure_before_registration", "ixaite_before"),
        ("configure_after_registration", "ixaite_after"),
        ("complete_native_registration", "native"),
    ):
        monkeypatch.setattr(registration, name, lambda event=event: calls.append(event))

    # Import-time decorators stay filtered until the final private registration.
    @contextmanager
    def suppress_baseline_oot():
        calls.append("filter_enter")
        try:
            yield
        finally:
            calls.append("filter_exit")

    for module_name, bindings in (
        ("vllm_iluvatar.ilu_patch", {"apply_general_patches": "baseline_patches"}),
        ("vllm_iluvatar.ilu_register", {"apply_oot_registrations": "oot"}),
        (
            "vllm_iluvatar.distributed.kv_transfer",
            {"apply_kv_transfer_registrations": "kv"},
        ),
        (
            "vllm_iluvatar.backends.ixaite.kv_transfer",
            {"apply_kv_transfer_registrations": "kv"},
        ),
        ("vllm_iluvatar.backends.ixaite.oot_registration", {}),
    ):
        module = ModuleType(module_name)
        for name, event in bindings.items():
            setattr(module, name, lambda event=event: calls.append(event))
        if module_name == "vllm_iluvatar.backends.ixaite.oot_registration":
            module.suppress_baseline_oot = suppress_baseline_oot
        monkeypatch.setitem(sys.modules, module_name, module)
    plugin.register_general()
    expected = ["baseline_patches", "oot", "kv"]
    if mode == "ixaite":
        expected = [
            "filter_enter",
            "native",
            "ixaite_patches",
            "ixaite_before",
            "oot",
            "ixaite_after",
            "kv",
            "filter_exit",
        ]
    assert calls == expected
