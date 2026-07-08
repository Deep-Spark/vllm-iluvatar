# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar mixed-precision linear selector."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_ixformer(monkeypatch) -> None:
    ixformer = types.ModuleType("ixformer")
    core = types.ModuleType("ixformer.core")
    config = types.SimpleNamespace(IXFORMER_USE_TORCH_OPS=False)
    inference = types.ModuleType("ixformer.inference")
    functions = types.ModuleType("ixformer.inference.functions")
    core.config = config
    ixformer.core = core
    ixformer.inference = inference
    inference.functions = functions
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.core", core)
    monkeypatch.setitem(sys.modules, "ixformer.core.config", config)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _load_selector(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    from vllm_iluvatar.envs import apply as apply_ilu_envs

    apply_ilu_envs()
    return importlib.import_module("vllm_iluvatar.model_executor.kernels.linear")


def test_ilu_linear_backend_env_is_registered_and_normalized(monkeypatch):
    selector = _load_selector(monkeypatch)

    monkeypatch.setenv("VLLM_ILU_LINEAR_BACKEND", "gptq-marlin")

    import vllm.envs as envs

    assert envs.VLLM_ILU_LINEAR_BACKEND == "gptq_marlin"
    assert selector._get_linear_backend() == "gptq_marlin"


def test_choose_mp_linear_kernel_uses_ilu_gptq_marlin_for_gptq_marlin_backend(
    monkeypatch,
):
    selector = _load_selector(monkeypatch)

    monkeypatch.setenv("VLLM_ILU_LINEAR_BACKEND", "gptq_marlin")
    monkeypatch.setattr(
        selector.IluGPTQMarlinLinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    kernel = selector.choose_mp_linear_kernel(object(), compute_capability=90)

    assert kernel is selector.IluGPTQMarlinLinearKernel


def test_choose_mp_linear_kernel_uses_community_for_auto_backend(monkeypatch):
    selector = _load_selector(monkeypatch)
    community_kernel = object()

    monkeypatch.delenv("VLLM_ILU_LINEAR_BACKEND", raising=False)
    monkeypatch.setattr(selector, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(
        selector,
        "_community_choose_mp_linear_kernel",
        lambda config, compute_capability=None: community_kernel,
    )

    kernel = selector.choose_mp_linear_kernel(object(), compute_capability=90)

    assert kernel is community_kernel
