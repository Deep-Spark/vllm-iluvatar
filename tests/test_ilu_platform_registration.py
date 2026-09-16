# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar plugin registration wiring."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

vllm_envs = importlib.import_module("vllm.envs")


def test_new_ilu_env_readers_are_registered(monkeypatch):
    envs = importlib.import_module("vllm_iluvatar.envs")
    monkeypatch.setattr(
        vllm_envs,
        "environment_variables",
        dict(vllm_envs.environment_variables),
    )
    monkeypatch.delenv("VLLM_ILU_GDN_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_GDN_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("USE_DEFAULT_FLA_NORM", raising=False)
    monkeypatch.delenv("VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV", raising=False)
    monkeypatch.delenv("VLLM_ILU_FUSE_NORM_QUANT", raising=False)
    monkeypatch.delenv("VLLM_ILU_FUSE_MOE_W13", raising=False)

    envs.apply()

    registered = vllm_envs.environment_variables
    assert (
        registered["VLLM_ILU_GDN_ATTN_BACKEND"] is registered["VLLM_GDN_ATTN_BACKEND"]
    )
    assert "VLLM_ILU_USE_DEFAULT_FLA_NORM" not in registered
    assert "VLLM_USE_DEFAULT_FLA_NORM" not in registered

    monkeypatch.setenv("VLLM_GDN_ATTN_BACKEND", "true")
    assert registered["VLLM_ILU_GDN_ATTN_BACKEND"]() is True
    monkeypatch.setenv("VLLM_ILU_GDN_ATTN_BACKEND", "0")
    assert registered["VLLM_ILU_GDN_ATTN_BACKEND"]() is False

    assert "VLLM_GLM52_RMS_ROPE_FP8_KV" not in registered
    assert "VLLM_FUSE_NORM_QUANT" not in registered
    assert "VLLM_FUSE_MOE_W13" not in registered

    rms_rope_reader = registered["VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV"]
    assert rms_rope_reader() is True
    monkeypatch.setenv("VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV", "0")
    assert rms_rope_reader() is False

    fuse_norm = registered["VLLM_ILU_FUSE_NORM_QUANT"]
    fuse_w13 = registered["VLLM_ILU_FUSE_MOE_W13"]
    assert fuse_norm() is True
    assert fuse_w13() is True
    monkeypatch.setenv("VLLM_ILU_FUSE_NORM_QUANT", "0")
    monkeypatch.setenv("VLLM_ILU_FUSE_MOE_W13", "0")
    assert fuse_norm() is False
    assert fuse_w13() is False


@pytest.mark.parametrize(
    ("is_ivcore30", "expected"),
    [(True, True), (False, False)],
    ids=["ivcore30", "other-arch"],
)
def test_gdn_backend_default_follows_arch(monkeypatch, is_ivcore30, expected):
    envs = importlib.import_module("vllm_iluvatar.envs")
    arch = importlib.import_module("vllm_iluvatar.ilu_util.arch")
    monkeypatch.setattr(
        vllm_envs,
        "environment_variables",
        dict(vllm_envs.environment_variables),
    )
    monkeypatch.delenv("VLLM_ILU_GDN_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_GDN_ATTN_BACKEND", raising=False)
    monkeypatch.setattr(arch, "is_ivcore30", lambda: is_ivcore30)

    envs.apply()

    reader = vllm_envs.environment_variables["VLLM_ILU_GDN_ATTN_BACKEND"]
    assert reader() is expected


def test_register_attention_backends_gates_gdn_backend(monkeypatch):
    platform = importlib.import_module("vllm_iluvatar.platforms.ilu_platform")
    monkeypatch.setattr(platform, "_ATTN_BACKENDS_REGISTERED", False)

    flash_attn = SimpleNamespace(name="FLASH_ATTN")
    flashmla = SimpleNamespace(name="FLASHMLA")
    flashmla_sparse = SimpleNamespace(name="FLASHMLA_SPARSE")
    flashmla_sparse_dsv4 = SimpleNamespace(name="FLASHMLA_SPARSE_DSV4")
    gdn_attn = SimpleNamespace(name="GDN_ATTN")
    calls = []

    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = SimpleNamespace(
        FLASH_ATTN=flash_attn,
        FLASHMLA=flashmla,
        FLASHMLA_SPARSE=flashmla_sparse,
        FLASHMLA_SPARSE_DSV4=flashmla_sparse_dsv4,
    )
    registry.MambaAttentionBackendEnum = SimpleNamespace(GDN_ATTN=gdn_attn)
    registry.register_backend = lambda member, path, is_mamba=False: calls.append(
        (member, path, is_mamba)
    )
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    monkeypatch.setattr(vllm_envs, "VLLM_ILU_GDN_ATTN_BACKEND", False, raising=False)
    platform.register_attention_backends()

    assert [call[0] for call in calls] == [
        flash_attn,
        flashmla,
        flashmla_sparse,
        flashmla_sparse_dsv4,
    ]
    assert all(call[2] is False for call in calls)

    calls.clear()
    monkeypatch.setattr(platform, "_ATTN_BACKENDS_REGISTERED", False)
    monkeypatch.setattr(vllm_envs, "VLLM_ILU_GDN_ATTN_BACKEND", True, raising=False)
    platform.register_attention_backends()

    assert calls[-1] == (
        gdn_attn,
        "vllm_iluvatar.v1.attention.backends.ilu_gdn_attn.IluGDNAttentionBackend",
        True,
    )


def test_register_platform_applies_envs_before_attention_backends(monkeypatch):
    plugin = importlib.import_module("vllm_iluvatar")
    calls = []

    monkeypatch.setitem(sys.modules, "ixformer", ModuleType("ixformer"))
    monkeypatch.setattr(
        plugin, "_load_import_compat_patch", lambda: calls.append("compat")
    )
    monkeypatch.delenv(plugin.STARTUP_PATCH_ENV, raising=False)

    fake_envs = ModuleType("vllm_iluvatar.envs")
    fake_envs.apply = lambda: calls.append("envs")
    monkeypatch.setitem(sys.modules, fake_envs.__name__, fake_envs)

    fake_platform = ModuleType("vllm_iluvatar.platforms.ilu_platform")
    fake_platform.register_attention_backends = lambda: calls.append("backends")
    monkeypatch.setitem(sys.modules, fake_platform.__name__, fake_platform)

    fake_hooks = ModuleType("vllm_iluvatar.ilu_patch.patch_platform_import_hooks")
    fake_hooks.register_platform_import_hooks = lambda: calls.append("hooks")
    monkeypatch.setitem(sys.modules, fake_hooks.__name__, fake_hooks)

    assert plugin.register_platform() == (
        "vllm_iluvatar.platforms.ilu_platform.IluPlatform"
    )
    assert os.environ[plugin.STARTUP_PATCH_ENV] == "1"
    assert calls == ["compat", "envs", "backends", "hooks"]
