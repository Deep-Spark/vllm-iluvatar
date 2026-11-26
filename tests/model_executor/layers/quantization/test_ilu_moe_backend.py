# SPDX-License-Identifier: Apache-2.0
"""Architecture/layout selection, including historical format aliases."""

import pytest
from conftest import requires_cuda

from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    select_moe_backend,
)


@pytest.mark.parametrize(
    "arch,name,layout,expected",
    [
        ("ivcore11", "auto", "auto", ("ixinfer_w4a8", "NN")),
        ("ivcore30", "auto", "auto", ("ixmm_fp4fp8", "fp4fp8")),
        ("ivcore11", "ixinfer_w4a8", "TN", ("ixinfer_w4a8", "TN")),
        ("ivcore30", "ixinfer_w4a8", "auto", ("ixinfer_w4a8", "TN")),
        ("ivcore11", "ixmm_w4a8", "auto", ("ixmm_w4a8", "NN")),
        ("ivcore30", "ixmm_w4a8", "auto", ("ixmm_w4a8", "NN")),
        ("ivcore30", "auto", "NN", ("ixmm_w4a8", "NN")),
    ],
)
def test_selection(arch, name, layout, expected):
    result = select_moe_backend(arch, name=name, layout=layout)
    assert (result.name, result.layout) == expected


@pytest.mark.parametrize(
    "arch,name,layout",
    [
        ("ivcore11", "ixmm_w4a8", "TN"),
        ("ivcore11", "ixmm_fp4fp8", "auto"),
        ("ivcore30", "ixinfer_w4a8", "NN"),
        ("ivcore30", "missing", "auto"),
        ("ivcore30", "ixmm_w4a8", "TN"),
    ],
)
def test_unsupported(arch, name, layout):
    with pytest.raises(ValueError):
        select_moe_backend(arch, name=name, layout=layout)


def _support_error(arch, name, layout, **config_overrides):
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        MoEComputeSelection,
        get_moe_backend,
    )

    config = MoEComputeConfig(
        **{
            "hidden": 6144,
            "intermediate": 384,
            "local_num_experts": 16,
            "global_num_experts": 128,
            "limit": 7.0,
            **config_overrides,
        }
    )
    selection = MoEComputeSelection(name, layout, arch)
    return get_moe_backend(selection).support_error(config, selection)


@pytest.mark.parametrize(
    "layout,intermediate,admitted",
    [
        # MiniMax-M3 at TP8 (hidden 6144, intermediate 3072/8). Both layouts'
        # kernels run it; the weight spec's single 256 used to turn it away.
        ("NN", 384, True),
        ("TN", 384, True),
        # w13's N is 2 * intermediate, so NN's N % 128 bites at 288, not 256.
        ("NN", 288, False),
        # NN needs K % 64, TN needs K % 128; w2's K is the intermediate.
        ("NN", 320, True),
        ("TN", 320, False),
        # Both need K >= 256, and w2's K is the intermediate.
        ("NN", 192, False),
    ],
)
def test_ixinfer_w4a8_admits_what_its_kernels_run(layout, intermediate, admitted):
    error = _support_error(
        "ivcore11", "ixinfer_w4a8", layout, intermediate=intermediate
    )
    assert (error is None) == admitted, error


def test_other_backends_keep_the_single_alignment_rule():
    # Only IxInfer states a per-projection rule. IxMM W4A8 still declares the
    # weight spec's 256, so the same dimensions stay out.
    assert "256-aligned" in _support_error("ivcore30", "ixmm_w4a8", "NN")
    # IxMM FP4FP8 declares 32 and takes them.
    assert _support_error("ivcore30", "ixmm_fp4fp8", "fp4fp8") is None


@pytest.mark.parametrize("activation", ["swigluoai", "swigluoai_uninterleave"])
@pytest.mark.parametrize(
    "overrides,rejected_for",
    [
        ({"alpha": 1.702, "beta": 1.0}, None),
        # Unset means the layer kept the activation's own default, which is
        # what the op has compiled in.
        ({}, None),
        ({"alpha": 1.0}, "alpha"),
        ({"beta": 0.0}, "beta"),
        ({"limit": 8.0}, "limit"),
    ],
)
def test_swigluoai_is_admitted_only_with_the_baked_coefficients(
    activation, overrides, rejected_for
):
    # ``activation_swigluoai_dynamic_scaled_int8`` exposes no coefficients, so
    # a checkpoint wanting different ones must be refused, not reinterpreted.
    error = _support_error(
        "ivcore11", "ixinfer_w4a8", "NN", activation=activation, **overrides
    )
    if rejected_for is None:
        assert error is None, error
    else:
        assert error is not None and rejected_for in error


def test_ixmm_takes_swigluoai_only_in_the_packed_form_and_only_clamped():
    """Its op reads packed halves and takes alpha/beta, unlike IxInfer's."""
    assert (
        _support_error(
            "ivcore30", "ixmm_fp4fp8", "fp4fp8", activation="swigluoai_uninterleave"
        )
        is None
    )
    # The interleaved variant is a different memory layout, not a coefficient
    # difference, so it must stay out rather than be reinterpreted.
    assert "does not support activation" in _support_error(
        "ivcore30", "ixmm_fp4fp8", "fp4fp8", activation="swigluoai"
    )
    # Every other activation here treats a missing limit as "no clamp"; this
    # one has no unclamped form.
    assert "clamp limit" in _support_error(
        "ivcore30",
        "ixmm_fp4fp8",
        "fp4fp8",
        activation="swigluoai_uninterleave",
        limit=None,
    )


def _ixinfer_backend(activation, rows=32, intermediate=384):
    from vllm_iluvatar.ilu_util.arch import cuda_arch
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        MoEComputeSelection,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixinfer_w4a8 import (
        IxInferW4A8Backend,
    )

    config = MoEComputeConfig(
        hidden=6144,
        intermediate=intermediate,
        local_num_experts=2,
        global_num_experts=2,
        activation=activation,
        limit=7.0,
    )
    # TN is the one layout both architectures run, so this does not pin the
    # test to the host's arch.
    return IxInferW4A8Backend(
        config, MoEComputeSelection("ixinfer_w4a8", "TN", cuda_arch())
    )


@requires_cuda
def test_swigluoai_activation_matches_the_oai_formula():
    import torch

    rows, intermediate = 32, 384
    backend = _ixinfer_backend("swigluoai_uninterleave", rows, intermediate)
    workspace = backend.workspace_spec(rows, torch.bfloat16).allocate("cuda")
    gemm1 = 3 * torch.randn(rows, 2 * intermediate, device="cuda", dtype=torch.bfloat16)
    order = torch.arange(rows, dtype=torch.int32, device="cuda")

    codes, scales = backend._activation_quant(gemm1, order, False, workspace)

    # The op reads interleaved gate/up pairs; getting the halves-to-interleaved
    # conversion wrong would mix the two branches and still look plausible.
    gate = gemm1[:, :intermediate].float().clamp(max=7.0)
    up = gemm1[:, intermediate:].float().clamp(-7.0, 7.0)
    expected = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    # One INT8 code is 1/127 of the row's amax; allow two.
    tolerance = 2 * expected.abs().amax(dim=-1, keepdim=True) / 127
    assert torch.all((codes.float() * scales.view(-1, 1) - expected).abs() <= tolerance)


def test_ixinfer_swiglustep_uses_step_activation(monkeypatch):
    from vllm_iluvatar.model_executor.layers.fused_moe.experts import (
        ilu_ixinfer_w4a8 as mod,
    )

    assert (
        _support_error(
            "ivcore11", "ixinfer_w4a8", "NN", activation="swiglustep", limit=7.0
        )
        is None
    )
    assert "limit 7" in _support_error(
        "ivcore11", "ixinfer_w4a8", "NN", activation="swiglustep", limit=3.0
    )
    monkeypatch.setattr(mod.ops, "activation_dynamic_scaled_int8", lambda **kw: kw)
    backend = _ixinfer_backend("swiglustep")
    intermediate, order = object(), object()
    assert backend._activation_quant(intermediate, order, True, None) == {
        "input": intermediate,
        "act_type": "swiglustep",
        "dst_to_src": order,
        "output_format": 1,
        "check_valid": True,
    }


def test_only_swigluoai_pays_for_the_interleave_buffer():
    import torch

    assert (
        "gemm1_interleaved"
        in _ixinfer_backend("swigluoai").workspace_spec(8, torch.bfloat16).buffers
    )
    assert (
        "gemm1_interleaved"
        not in _ixinfer_backend("silu").workspace_spec(8, torch.bfloat16).buffers
    )


def test_registration_and_explicit_priority(monkeypatch):
    from vllm_iluvatar.model_executor.layers.fused_moe.experts import (
        ilu_compute_backend as mod,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_w4a8 import (
        IxMMW4A8Backend,
    )

    mod._load_builtin_backends()
    monkeypatch.setattr(mod, "_BACKENDS", dict(mod._BACKENDS))

    class ExtraBackend(IxMMW4A8Backend):
        name = "test_extra"
        priority = 5

    mod.register_moe_backend(ExtraBackend)
    assert mod.select_moe_backend("ivcore30").name == "test_extra"
    with pytest.raises(ValueError, match="Duplicate"):
        mod.register_moe_backend(ExtraBackend)
    with pytest.raises(TypeError):
        mod.register_moe_backend(mod.MoEComputeBackend)
    # Import/dict order cannot change selection.
    monkeypatch.setattr(mod, "_BACKENDS", dict(reversed(list(mod._BACKENDS.items()))))
    assert mod.select_moe_backend("ivcore30").name == "test_extra"


def test_only_small_ixmm_batches_may_skip_the_expert_sort():
    """The group layout's width is how the caller asks for an unsorted route."""
    import torch

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )

    def columns(rows, skip_sort):
        backend = IxMMFP4FP8Backend(
            MoEComputeConfig(512, 256, 2, 4, limit=10.0),
            select_moe_backend("ivcore30"),
        )
        spec = backend.workspace_spec(rows, torch.bfloat16, skip_sort=skip_sort)
        return spec.buffers["layout"][0][1]

    assert columns(8, True) == 2
    assert columns(8, False) == 3
    # Past the bound the sort repays itself, so the opt-in is ignored.
    assert columns(257, True) == 3
    # IxInfer never sorts, so asking changes nothing there.
    assert (
        _ixinfer_backend("silu").workspace_spec(8, torch.bfloat16).buffers["layout"][0]
        == _ixinfer_backend("silu")
        .workspace_spec(8, torch.bfloat16, skip_sort=True)
        .buffers["layout"][0]
    )


def test_decode_tile_mode_is_opt_in_and_shape_matched(monkeypatch):
    from vllm_iluvatar import envs
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )

    mode = IxMMFP4FP8Backend._decode_mode
    monkeypatch.setattr(envs, "VLLM_ILU_V41_DECODE_MOE_MODE1", True)
    assert mode(768, 5120, 6) == 1
    # Every part of the shape the tiling was measured on has to hold.
    assert mode(769, 5120, 6) is None
    assert mode(768, 4096, 6) is None
    assert mode(768, 5120, 8) is None
    monkeypatch.setattr(envs, "VLLM_ILU_V41_DECODE_MOE_MODE1", False)
    assert mode(768, 5120, 6) is None


def test_runtime_weights_validate_format_and_alias_parameters():
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
        MoEWeights,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )

    backend = IxMMFP4FP8Backend(
        MoEComputeConfig(512, 256, 2, 4, limit=10.0),
        select_moe_backend("ivcore30"),
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 256, 512, dtype=torch.uint8),
        w13_weight_scale=torch.empty(2, 2, 512),
        w2_weight=torch.empty(2, 128, 512, dtype=torch.uint8),
        w2_weight_scale=torch.empty(2, 1, 512),
    )
    weights = MoEWeights.from_layer(layer, backend.weight_spec)
    backend.validate_weights(weights)
    assert weights.w13_weight is layer.w13_weight
    assert not hasattr(backend, "layer")
    with pytest.raises(ValueError, match="Expected ixmm_fp4fp8"):
        backend.validate_weights(
            replace(weights, spec=replace(weights.spec, format_id="other"))
        )
    with pytest.raises(ValueError, match="expected weight"):
        backend.validate_weights(
            replace(weights, w2_weight_scale=torch.empty(2, 512, 1))
        )
    with pytest.raises(TypeError, match="dtype"):
        backend.validate_weights(
            replace(weights, w2_weight=weights.w2_weight.to(torch.int8))
        )


def test_standard_finalize_does_not_weight_a_second_time():
    import torch
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
        IluBatchedComputeExperts,
        IluStandardComputeExperts,
    )

    standard = object.__new__(IluStandardComputeExperts)
    batched = object.__new__(IluBatchedComputeExperts)
    reduced = torch.tensor([[7.0, 11.0]])
    output = torch.empty_like(reduced)
    standard.finalize_weight_and_reduce_impl().apply(
        output, reduced, torch.tensor([[0.2, 0.8]]), torch.tensor([[0, 1]]), False
    )
    torch.testing.assert_close(output, reduced, rtol=0, atol=0)
    assert isinstance(
        batched.finalize_weight_and_reduce_impl(), TopKWeightAndReduceDelegate
    )


def test_workspace_is_scoped_to_call_and_device():
    import torch

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEWorkspaceSpec,
    )

    spec = MoEWorkspaceSpec({"gemm1": ((8, 16), torch.bfloat16)})
    first, second = spec.allocate("cpu"), spec.allocate("cpu")
    first_buffer = first.get("gemm1")
    assert first.get("gemm1") is first_buffer
    assert second.get("gemm1").data_ptr() != first_buffer.data_ptr()
    first.release("gemm1")
    assert not first.buffers
    with pytest.raises(ValueError, match="capacity"):
        first.get("gemm1", rows=9)


def test_ixmm_standard_keeps_parameter_weights_distinct_from_router_weights(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch

    from vllm_iluvatar.model_executor.layers.fused_moe.experts import (
        ilu_ixmm_w4a8 as mod,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEComputeConfig,
    )

    compute = mod.IxMMW4A8Backend(
        MoEComputeConfig(512, 256, 2, 2, limit=10.0),
        select_moe_backend("ivcore30", name="ixmm_w4a8"),
    )
    parameters = SimpleNamespace()
    routing_weights = torch.tensor([0.3, 0.7])
    monkeypatch.setattr(
        mod.ops,
        "moe_compute_token_index_ep_packed",
        lambda **kwargs: (
            torch.arange(2),
            None,
            routing_weights,
            torch.arange(3),
            None,
            torch.tensor(2),
        ),
    )
    monkeypatch.setattr(
        mod.ops, "moe_expand_input_ep_dynamic_scaled_int8", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        mod.ops,
        "activation_dynamic_scaled_int8",
        lambda **kwargs: (torch.zeros(2, 256, dtype=torch.int8), torch.ones(2)),
    )
    seen = []

    def gemm(weights, x, scale, prefix, layout, output=None, mode=None):
        assert weights is parameters
        seen.append(prefix)
        output.zero_()

    def reduce(**kwargs):
        assert kwargs["weights"] is routing_weights
        assert kwargs["output"] is destination
        return kwargs["output"].zero_()

    monkeypatch.setattr(compute, "gemm", gemm)
    monkeypatch.setattr(mod.ops, "moe_output_reduce_sum_ep", reduce)
    x = torch.zeros(2, 512, dtype=torch.bfloat16)
    destination = torch.empty_like(x)
    compute.run_standard(
        x,
        parameters,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        None,
        workspace=compute.workspace_spec(2, x.dtype).allocate("cpu"),
        output=destination,
    )
    assert seen == ["w13", "w2"]


def test_standard_adapter_passes_caller_output_without_copy():
    from types import SimpleNamespace

    import torch

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEWorkspaceSpec,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
        IluStandardComputeExperts,
    )

    x = torch.zeros(2, 8)
    output = torch.empty_like(x)

    def run(*args, output, workspace):
        output.fill_(7)
        # None deliberately detects adapters that copy a return value.

    adapter = object.__new__(IluStandardComputeExperts)
    adapter.weights = object()
    adapter.layer = None
    adapter.backend = SimpleNamespace(
        run_standard=run, workspace_spec=lambda *args, **kwargs: MoEWorkspaceSpec({})
    )
    result = adapter.apply(
        output,
        x,
        None,
        None,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        None,
        1,
        None,
        None,
        None,
        None,
        None,
    )
    assert result is output
    assert torch.equal(output, torch.full_like(output, 7))


@pytest.mark.parametrize("batched", [False, True])
def test_adapter_protocol_binding_and_scaling(monkeypatch, batched):
    from types import SimpleNamespace

    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
        TopKWeightAndReduceNoOP,
    )

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        MoEWorkspaceSpec,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
        bind_compute_experts,
    )

    monkeypatch.setattr(FusedMoEExpertsModular, "__init__", lambda self, *args: None)
    fmt = (
        FusedMoEActivationFormat.BatchedExperts
        if batched
        else FusedMoEActivationFormat.Standard
    )
    prepare = SimpleNamespace(
        activation_format=fmt,
        max_num_tokens_per_rank=lambda: 4,
        num_dispatchers=lambda: 1,
    )
    weights = object()
    counts = torch.tensor([1, 3], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75]])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    scale = torch.ones(2, 4, 1) if batched else None
    x = torch.zeros(2, 4, 8) if batched else torch.zeros(1, 8)
    output = torch.empty_like(x)
    seen = []

    def workspace_spec(rows, dtype, **kwargs):
        assert rows == (8 if batched else 2)
        assert kwargs.get("batched", False) == batched
        return MoEWorkspaceSpec({})

    def standard(a, w, tw, ti, expert_map, *, output, workspace):
        assert not batched and w is weights and tw is topk_weights and ti is topk_ids
        seen.append("standard")
        output.fill_(7)

    def batch(a, w, scales, valid, *, output, workspace):
        assert batched and w is weights and scales is scale and valid is counts
        seen.append("batched")
        output.fill_(11)

    backend = SimpleNamespace(
        output_scale=2.5,
        local_num_experts=2,
        intermediate=16,
        workspace_spec=workspace_spec,
        run_standard=standard,
        run_batched=batch,
        validate_batched_activation=lambda: None,
    )
    adapter = bind_compute_experts(backend, weights, None, None, prepare)
    assert adapter.activation_format() == fmt
    assert adapter.runner_scaling_factor == (2.5 if batched else 1.0)
    reduction = adapter.finalize_weight_and_reduce_impl()
    assert isinstance(
        reduction, TopKWeightAndReduceDelegate if batched else TopKWeightAndReduceNoOP
    )
    assert adapter.workspace_shapes(8 if batched else 1, 32, 8, 2, 2, 2, None, None)[
        2
    ] == tuple(x.shape)
    adapter.apply(
        output,
        x,
        None,
        None,
        topk_weights,
        topk_ids,
        None,
        2,
        None,
        scale,
        None,
        None,
        None,
        expert_tokens_meta=SimpleNamespace(expert_num_tokens=counts)
        if batched
        else None,
    )
    assert seen == ["batched" if batched else "standard"]
    assert torch.equal(output, torch.full_like(output, 11 if batched else 7))
    if batched:
        with pytest.raises(ValueError, match="valid token counts"):
            adapter.apply(
                output,
                x,
                None,
                None,
                topk_weights,
                topk_ids,
                None,
                2,
                None,
                scale,
                None,
                None,
                None,
            )


@pytest.mark.parametrize(
    "activation",
    ["silu", "swiglu", "swiglustep", "swigluoai", "swigluoai_uninterleave"],
)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("backend_name", ["ixinfer_w4a8", "ixmm_fp4fp8"])
def test_activation_at_binding(monkeypatch, activation, batched, backend_name):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )

    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_experts import (
        bind_compute_experts,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixinfer_w4a8 import (
        IxInferW4A8Backend,
    )
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_ixmm_fp4fp8 import (
        IxMMFP4FP8Backend,
    )

    cls = IxInferW4A8Backend if backend_name == "ixinfer_w4a8" else IxMMFP4FP8Backend
    if backend_name == "ixmm_fp4fp8" and activation == "swigluoai":
        pytest.skip("Activation is not supported by this backend")
    assert activation in cls.supported_activations
    backend = object.__new__(cls)
    backend.config = SimpleNamespace(activation=activation)
    backend.local_num_experts = 2
    fmt = (
        FusedMoEActivationFormat.BatchedExperts
        if batched
        else FusedMoEActivationFormat.Standard
    )
    prepare = SimpleNamespace(
        activation_format=fmt,
        max_num_tokens_per_rank=lambda: 4,
        num_dispatchers=lambda: 1,
    )
    monkeypatch.setattr(FusedMoEExpertsModular, "__init__", lambda self, *args: None)
    unsupported = activation.startswith("swigluoai") or (
        activation == "swiglustep" and backend_name == "ixinfer_w4a8"
    )
    if batched and unsupported:
        with pytest.raises(
            NotImplementedError, match=f"batched experts do not support {activation}"
        ):
            bind_compute_experts(backend, None, None, None, prepare)
    else:
        adapter = bind_compute_experts(backend, None, None, None, prepare)
        assert adapter.activation_format() == fmt
