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
    selected = mod.make_mxfp4_method(SimpleNamespace(moe_config=None))
    assert selected.layout == "TN"


@pytest.mark.parametrize("moe", [{}, {"mode": "ixinfer_w4a8_tn_v2"}])
def test_mxfp4_rejects_the_retired_backend_env(monkeypatch, moe):
    """A launch script still selecting a kernel must hear that it cannot."""
    from vllm_iluvatar import envs
    from vllm_iluvatar.model_executor.layers.quantization.backends.fp4 import (
        mxfp4_moe as mod,
    )

    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore30")
    monkeypatch.setattr(envs, "VLLM_ILU_MXFP4_MOE_BACKEND", "torch")
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(additional_config={"ilu_compute": {"moe": moe}}),
    )
    with pytest.raises(ValueError, match="no longer"):
        mod.make_mxfp4_method(SimpleNamespace(moe_config=None))


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


@pytest.mark.parametrize("source", ["block-FP8", "MXFP4", "MXFP8"])
def test_bi150_ixmm_w4a8_sources(source):
    selection = ilu_compute.select_four_bit_expert_compute(
        "ivcore11", "ixmm_w4a8_nn", source=source
    )
    assert (selection.name, selection.layout) == ("ixmm_w4a8", "NN")


@pytest.mark.parametrize("source", ["block-FP8", "MXFP4", "MXFP8"])
def test_bi150_nn_v1_sources(source):
    select = ilu_compute.select_four_bit_expert_compute
    selection = select("ivcore11", "ixinfer_w4a8_nn_v1", source=source)
    assert (selection.name, selection.layout) == ("ixinfer_w4a8_nn_v1", "NN")
    with pytest.raises(ValueError, match="Unsupported"):
        select("ivcore30", "ixinfer_w4a8_nn_v1", source=source)


@pytest.mark.parametrize("mode", ["ixinfer_w4a8_nn_v1", "ixmm_w4a8_nn"])
@pytest.mark.parametrize("source", ["native BF16/FP16", "NVFP4"])
def test_bi150_converted_modes_reject_other_sources(mode, source):
    with pytest.raises(ValueError, match="requires one of"):
        ilu_compute.select_four_bit_expert_compute("ivcore11", mode, source=source)


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


@pytest.mark.parametrize(
    "linear,moe,expected_linear,expected_layout",
    [
        ({}, {}, "w8a8_int8", "NN"),
        (
            {"native_backend": "w8a8_int8"},
            {"mode": "ixinfer_w4a8_nn_v2"},
            "w8a8_int8",
            "NN",
        ),
        (
            {"native_backend": "original"},
            {"mode": "ixinfer_w4a8_tn_v2"},
            "original",
            "TN",
        ),
        (
            {"native_backend": "original"},
            {"mode": "original"},
            "original",
            None,
        ),
    ],
)
def test_native_compute_modes(
    monkeypatch, linear, moe, expected_linear, expected_layout
):
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"linear": linear, "moe": moe}}
        ),
    )
    assert ilu_compute.native_linear_backend() == expected_linear
    selection = ilu_compute.native_moe_compute()
    if expected_layout is None:
        assert selection is None
    else:
        assert (selection.name, selection.layout) == ("ixinfer_w4a8", expected_layout)


@pytest.mark.parametrize(
    "mode", ["bf16.fp16", "bf16", "fp16", "ixinfer_w4a8_nn", "w8a8_int8", ""]
)
def test_native_modes_take_only_their_canonical_spelling(monkeypatch, mode):
    """``original`` is the only way to ask for the model's own precision."""
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={
                "ilu_compute": {
                    "linear": {"native_backend": mode},
                    "moe": {"mode": mode},
                }
            }
        ),
    )
    if mode != "w8a8_int8":
        with pytest.raises(ValueError, match="native_backend"):
            ilu_compute.native_linear_backend()
    with pytest.raises(ValueError, match="native BF16/FP16 mode"):
        ilu_compute.native_moe_compute()


@pytest.mark.parametrize("setting", ["fp8", "", None, ["int8"], {"mode": "int8"}])
def test_invalid_native_ple_backend(monkeypatch, setting):
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={"ilu_compute": {"ple": {"backend": setting}}}
        ),
    )
    with pytest.raises(ValueError, match=r"ilu_compute[.]ple[.]backend"):
        ilu_compute.native_ple_backend()
