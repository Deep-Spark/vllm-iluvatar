# SPDX-License-Identifier: Apache-2.0
"""Execution preferences must not confuse architecture defaults with overrides."""

from types import SimpleNamespace

import pytest

from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.ilu_util import arch


@pytest.mark.parametrize(
    "device,setting,expected",
    [
        ("ivcore11", "auto", "w8a8_int8"),
        ("ivcore30", "auto", "fp8"),
        ("ivcore30", "w8a8_int8", "w8a8_int8"),
        ("ivcore11", "w8a8_int8", "w8a8_int8"),
    ],
)
def test_linear_selection(monkeypatch, device, setting, expected):
    monkeypatch.setattr(arch, "cuda_arch", lambda: device)
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": setting}}}
        ),
    )
    assert ilu_compute.fp8_linear_backend() == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"linear": {"fp8_backend": "int9"}},
        {"linear": {"backend": "fp8"}},
        {"schema_version": True},
    ],
)
def test_invalid_compute_config(monkeypatch, raw):
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(additional_config={"ilu_compute": raw}),
    )
    with pytest.raises(ValueError):
        ilu_compute.fp8_linear_backend()


def test_explicit_fp8_rejects_ivcore11(monkeypatch):
    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": {"fp8_backend": "fp8"}}}
        ),
    )
    with pytest.raises(ValueError, match="not supported"):
        ilu_compute.fp8_linear_backend()


@pytest.mark.parametrize(
    "device,mode,expected",
    [
        ("ivcore11", "auto", "ixinfer_w4a8_nn_v2"),
        ("ivcore11", "ixinfer_w4a8_nn_v2", "ixinfer_w4a8_nn_v2"),
        ("ivcore11", "ixinfer_w4a8_tn_v2", "ixinfer_w4a8_tn_v2"),
        ("ivcore30", "auto", "ixmm_fp4fp8"),
        ("ivcore30", "ixmm_fp4fp8", "ixmm_fp4fp8"),
        ("ivcore30", "ixmm_w4a8_nn", "ixmm_w4a8_nn"),
    ],
)
def test_mxfp4_public_modes(monkeypatch, device, mode, expected):
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        get_moe_backend,
    )
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
        select_mxfp4_compute,
    )

    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"moe": {"mode": mode}}}
        ),
    )
    selection = select_mxfp4_compute(device, ilu_compute.compute_options("moe")["mode"])
    assert (
        get_moe_backend(selection).weight_specs[selection.layout].format_id == expected
    )


@pytest.mark.parametrize(
    "device,mode",
    [
        ("ivcore11", "ixmm_w4a8_nn"),
        ("ivcore11", "ixmm_fp4fp8"),
        ("ivcore30", "ixinfer_w4a8_tn_v2"),
        ("ivcore30", "ixinfer_w4a8_nn_v2"),
        ("ivcore11", "NN"),
        ("ivcore30", None),
        ("ivcore30", {}),
    ],
)
def test_mxfp4_public_modes_reject_cross_architecture(device, mode):
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
        select_mxfp4_compute,
    )

    with pytest.raises(ValueError, match="Unsupported MXFP4 mode"):
        select_mxfp4_compute(device, mode)


def test_mxfp4_explicit_mode_overrides_legacy_env(monkeypatch):
    from vllm_iluvatar import envs
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4 import (
        mxfp4_moe as mod,
    )

    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(envs, "VLLM_ILU_MOE_RUNTIME_LAYOUT", "NN")
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"moe": {"mode": "ixinfer_w4a8_tn_v2"}}}
        ),
    )
    monkeypatch.setattr(mod, "IluMxfp4MoEMethod", lambda config, selection: selection)
    selected = mod.select_mxfp4_method(SimpleNamespace(moe_config=None), object())
    assert selected.layout == "TN"


@pytest.mark.parametrize(
    "device,mode,backend,layout",
    [
        ("ivcore11", "auto", "ixinfer_w4a8", "NN"),
        ("ivcore11", "ixmm_w4a8_nn", "ixmm_w4a8", "NN"),
        ("ivcore11", "ixinfer_w4a8_tn_v2", "ixinfer_w4a8", "TN"),
        ("ivcore30", "auto", "ixmm_fp4fp8", "fp4fp8"),
        ("ivcore30", "ixmm_w4a8_nn", "ixmm_w4a8", "NN"),
    ],
)
def test_fp8_compute_modes(device, mode, backend, layout):
    """The public mode table every converted expert source resolves through."""
    selection = ilu_compute.select_four_bit_expert_compute(
        device, mode, source="block-FP8"
    )
    assert (selection.name, selection.layout) == (backend, layout)


def test_bi150_nn_v1_only_accepts_block_fp8():
    select = ilu_compute.select_four_bit_expert_compute
    selection = select("ivcore11", "ixinfer_w4a8_nn_v1", source="block-FP8")
    assert (selection.name, selection.layout) == ("ixinfer_w4a8_nn_v1", "NN")
    with pytest.raises(ValueError, match="Unsupported"):
        select("ivcore11", "ixinfer_w4a8_nn_v1", source="MXFP4")


def test_fp8_runtime_compute_conflict(monkeypatch):
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
        select_fp8_method,
    )

    original, runtime = object(), object()
    monkeypatch.setattr(
        ilu_compute, "compute_options", lambda section: {"mode": "ixmm_fp4fp8"}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        select_fp8_method(original, runtime)
    monkeypatch.setattr(
        ilu_compute, "compute_options", lambda section: {"mode": "auto"}
    )
    assert select_fp8_method(original, runtime) is runtime
    assert select_fp8_method(original, None) is original
