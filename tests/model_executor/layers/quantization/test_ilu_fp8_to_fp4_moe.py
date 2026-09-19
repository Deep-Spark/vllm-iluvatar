# SPDX-License-Identifier: Apache-2.0
"""Load-time block-FP8 -> ix30 FP4 routed-expert loading.

The load-time path only pays off if it is *indistinguishable* from the
existing one, so almost every test here ends in a byte comparison against
``IluFp8MoEMethod``: same checkpoint in, same four parameters out. Anything
looser would let a sharding or pairing bug hide behind FP4's coarse grid.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8 import (
    IluFp8MoEMethod,
    _ixmm_backend_available,
    preprocess_fp8_block_weight_for_ixmm,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_to_fp4 import (
    Fp8SourceLayout,
    convert_to_ix30_fp4,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_to_fp4_moe import (
    IluFp8ToFp4OnlineMoEMethod,
)

BLOCK = (128, 128)
HIDDEN = 256
INTERMEDIATE = 256
NUM_EXPERTS = 4

# Every test converts with an explicit backend. "torch" is the accuracy
# baseline; "ixformer_native" must agree with it byte for byte (ixformer's own
# test_preprocess_fp8_block_weight_for_ixmm.py holds the kernel to that), so
# running both here catches a facade-level mistake such as a mis-shaped
# destination view that only one path would notice.
BACKENDS = ["torch", "ixformer_native"]


def _has_ixmm_ivcore30() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from vllm_iluvatar.ilu_util.arch import is_ivcore30

        return is_ivcore30() and _ixmm_backend_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_ixmm_ivcore30(), reason="requires an ivcore30 GPU with ixmm backend"
)


# ---------------------------------------------------------------------------
# A RoutedExperts stand-in
# ---------------------------------------------------------------------------


class _ExpertMapManager:
    def __init__(self, global_to_local: dict[int, int]):
        self._map = global_to_local

    def map_global_to_local(self, expert_id: int) -> int:
        return self._map.get(expert_id, -1)


class _StubRoutedExperts(RoutedExperts):
    """Real ``RoutedExperts`` loading methods, no engine required.

    Subclassing (rather than duck typing) is deliberate: the point of these
    tests is that the new loader delegates TP narrowing to community code, so
    the community code under test has to be the real thing.
    """

    def __init__(self, tp_size: int, tp_rank: int, local_experts: dict[int, int]):
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            is_act_and_mul=True,
            has_bias=False,
            hidden_dim=HIDDEN,
            hidden_dim_unpadded=HIDDEN,
            intermediate_size_per_partition_unpadded=None,
            num_local_experts=len(local_experts),
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_parallel_config=SimpleNamespace(tp_size=tp_size),
        )
        self.expert_map_manager = _ExpertMapManager(local_experts)
        self._expert_map = None
        self.rocm_aiter_fmoe_enabled = False
        self.quant_config = None
        self.quant_method = None
        self.activation = "silu"
        self.prefix = "model.layers.0.mlp.experts"
        self._loaded_expert_biases = set()


def _make_layer(method_cls, tp_size, tp_rank, local_experts, **method_kwargs):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    layer = _StubRoutedExperts(tp_size, tp_rank, local_experts)
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(BLOCK),
    )
    method = method_cls(quant_config, layer, **method_kwargs)
    layer.quant_method = method
    # Both create_weights implementations allocate with the ambient default
    # device, the way vLLM sets it up during engine init.
    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            num_experts=len(local_experts),
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE // tp_size,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
    return layer, method


# ---------------------------------------------------------------------------
# A synthetic checkpoint
# ---------------------------------------------------------------------------


def _make_checkpoint(num_experts=NUM_EXPERTS, seed=0):
    """One entry per (expert, param) in checkpoint (pre-TP) shapes."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def fp8(*shape):
        return (
            torch.randn(shape, generator=g, dtype=torch.float32, device="cuda")
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )

    def scale(*shape):
        return (
            torch.rand(shape, generator=g, dtype=torch.float32, device="cuda") * 0.75
            + 0.5
        )

    bn, bk = BLOCK
    ckpt = {}
    for e in range(num_experts):
        ckpt[(e, "w1", "weight")] = fp8(INTERMEDIATE, HIDDEN)
        ckpt[(e, "w1", "scale")] = scale(INTERMEDIATE // bn, HIDDEN // bk)
        ckpt[(e, "w3", "weight")] = fp8(INTERMEDIATE, HIDDEN)
        ckpt[(e, "w3", "scale")] = scale(INTERMEDIATE // bn, HIDDEN // bk)
        ckpt[(e, "w2", "weight")] = fp8(HIDDEN, INTERMEDIATE)
        ckpt[(e, "w2", "scale")] = scale(HIDDEN // bn, INTERMEDIATE // bk)
    return ckpt


_PARAM_OF = {
    ("w1", "weight"): "w13_weight",
    ("w3", "weight"): "w13_weight",
    ("w2", "weight"): "w2_weight",
    ("w1", "scale"): "w13_weight_scale_inv",
    ("w3", "scale"): "w13_weight_scale_inv",
    ("w2", "scale"): "w2_weight_scale_inv",
}


def _arrival_order(ckpt):
    """Checkpoint order: each weight immediately followed by its scale."""
    order = []
    experts = sorted({key[0] for key in ckpt})
    for e in experts:
        for shard in ("w1", "w3", "w2"):
            order.append((e, shard, "weight"))
            order.append((e, shard, "scale"))
    return order


def _feed(layer, ckpt, order):
    for expert_id, shard, kind in order:
        param_name = _PARAM_OF[(shard, kind)]
        param = getattr(layer, param_name)
        param.weight_loader(
            param=param,
            loaded_weight=ckpt[(expert_id, shard, kind)],
            weight_name=f"model.layers.0.mlp.experts.{param_name}",
            shard_id=shard,
            expert_id=expert_id,
            return_success=True,
        )


_FINAL_PARAMS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


def _assert_params_equal(expected_layer, actual_layer):
    for name in _FINAL_PARAMS:
        expected = getattr(expected_layer, name)
        actual = getattr(actual_layer, name)
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        mismatch = int((actual != expected).sum().item())
        assert mismatch == 0, (
            f"{name}: {mismatch}/{expected.numel()} elements differ between the "
            f"load-time FP4 path and IluFp8MoEMethod"
        )
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        assert not hasattr(actual_layer, name), (
            f"{name} must be dropped once every shard is converted"
        )


def _run_both_paths(
    monkeypatch, tp_size, tp_rank, local_experts, backend, order=None, **online_kwargs
):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    ckpt = _make_checkpoint()
    order = order or _arrival_order(ckpt)

    baseline_layer, baseline_method = _make_layer(
        IluFp8MoEMethod, tp_size, tp_rank, local_experts
    )
    _feed(baseline_layer, ckpt, order)
    baseline_method.process_weights_after_loading(baseline_layer)

    online_layer, online_method = _make_layer(
        IluFp8ToFp4OnlineMoEMethod,
        tp_size,
        tp_rank,
        local_experts,
        converter_backend=backend,
        **online_kwargs,
    )
    _feed(online_layer, ckpt, order)
    online_method.process_weights_after_loading(online_layer)

    return baseline_layer, online_layer, online_method


# ---------------------------------------------------------------------------
# T-0 / T-1: the conversion itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ixformer_op_matches_the_plugin_reference(backend):
    """T-0: moving the conversion into ixformer did not change the numbers."""
    ckpt = _make_checkpoint(num_experts=2, seed=3)
    weight = torch.stack([ckpt[(e, "w1", "weight")] for e in range(2)])
    scale = torch.stack([ckpt[(e, "w1", "scale")] for e in range(2)])

    ref_w, ref_s = preprocess_fp8_block_weight_for_ixmm(weight, scale, BLOCK)
    got_w, got_s = convert_to_ix30_fp4(
        weight,
        scale,
        source=Fp8SourceLayout(format="fp8-block", block_shape=BLOCK),
        backend=backend,
    )
    assert torch.equal(got_w, ref_w)
    assert torch.equal(got_s, ref_s)


@pytest.mark.parametrize("backend", BACKENDS)
def test_column_subset_conversion_matches_whole_tensor(backend):
    """T-1: converting the w1 half alone equals converting all 2I columns.

    This is the property the whole design rests on: ix30 groups never span
    columns, so a column subset can be quantized independently.
    """
    ckpt = _make_checkpoint(num_experts=1, seed=4)
    w1, s1 = ckpt[(0, "w1", "weight")], ckpt[(0, "w1", "scale")]
    w3, s3 = ckpt[(0, "w3", "weight")], ckpt[(0, "w3", "scale")]
    fused_w = torch.cat([w1, w3], dim=0).unsqueeze(0)
    fused_s = torch.cat([s1, s3], dim=0).unsqueeze(0)
    source = Fp8SourceLayout(format="fp8-block", block_shape=BLOCK)

    full_w, full_s = convert_to_ix30_fp4(
        fused_w, fused_s, source=source, backend=backend
    )

    out_w = torch.zeros_like(full_w)
    out_s = torch.zeros_like(full_s)
    for offset, (w, s) in ((0, (w1, s1)), (INTERMEDIATE, (w3, s3))):
        convert_to_ix30_fp4(
            w.unsqueeze(0),
            s.unsqueeze(0),
            source=source,
            out_weight=out_w.narrow(-1, offset, INTERMEDIATE),
            out_scale=out_s.narrow(-1, offset, INTERMEDIATE),
            backend=backend,
        )
    assert torch.equal(out_w, full_w)
    assert torch.equal(out_s, full_s)


# ---------------------------------------------------------------------------
# T-2 / T-3: end-to-end loader equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_loader_matches_ilu_fp8_moe_method(monkeypatch, backend):
    """T-2: same checkpoint, same four parameters."""
    local = {e: e for e in range(NUM_EXPERTS)}
    baseline, online, method = _run_both_paths(monkeypatch, 1, 0, local, backend)
    _assert_params_equal(baseline, online)
    assert method._conversions == 3 * NUM_EXPERTS


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (2, 0)])
def test_loader_matches_across_tp_ranks(monkeypatch, tp_size, tp_rank):
    """T-3a: w13 splits on n, w2 splits on k; both must land identically."""
    local = {e: e for e in range(NUM_EXPERTS)}
    baseline, online, _ = _run_both_paths(monkeypatch, tp_size, tp_rank, local, "torch")
    _assert_params_equal(baseline, online)


def test_loader_matches_with_expert_parallelism(monkeypatch):
    """T-3b: EP keeps only a subset of experts on this rank."""
    local = {2: 0, 3: 1}
    baseline, online, method = _run_both_paths(monkeypatch, 1, 0, local, "torch")
    _assert_params_equal(baseline, online)
    assert method._conversions == 3 * len(local)


# ---------------------------------------------------------------------------
# The model's "every parameter came from the checkpoint" audit
# ---------------------------------------------------------------------------


def test_only_the_derived_scales_are_exempt_from_the_checkpoint_audit(monkeypatch):
    """``IluGlmMoeDsaForCausalLM.load_weights`` refuses to start when a
    parameter was never written by a checkpoint tensor. The FP4 group scales
    have no checkpoint tensor at all -- the converter derives them -- so they
    carry a marker the audit honours. Everything else must stay auditable, or a
    genuinely missing shard would start serving uninitialised bytes.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, _ = _make_layer(
        IluFp8ToFp4OnlineMoEMethod, 1, 0, local, converter_backend="torch"
    )

    exempt = {
        name
        for name, param in layer.named_parameters()
        if getattr(param, "ilu_derived_at_load", False)
    }
    assert exempt == {"w13_weight_scale", "w2_weight_scale"}


# ---------------------------------------------------------------------------
# T-4: arrival order
# ---------------------------------------------------------------------------


def test_scale_before_weight_and_cross_expert_interleaving(monkeypatch):
    """T-4: the loader pairs by (expert, shard), not by arrival position."""
    ckpt = _make_checkpoint()
    # Scales ahead of their weights, and two experts in flight at once: six
    # half-loaded units, still inside the default budget.
    order = []
    experts = sorted({key[0] for key in ckpt})
    for lo in range(0, len(experts), 2):
        pair = experts[lo : lo + 2]
        for kind in ("scale", "weight"):
            for e in pair:
                for shard in ("w2", "w3", "w1"):
                    order.append((e, shard, kind))

    local = {e: e for e in range(NUM_EXPERTS)}
    baseline, online, method = _run_both_paths(
        monkeypatch, 1, 0, local, "torch", order=order
    )
    _assert_params_equal(baseline, online)
    assert method._pending_peak == 6


def test_worst_case_arrival_order_needs_a_raised_budget(monkeypatch):
    """T-4b: every scale before every weight still converts correctly.

    The default ``max_pending_shards`` rejects this order on purpose -- it
    pins one FP8 staging buffer per expert, which is the memory blowup this
    loader exists to prevent -- so it is only reachable by opting in.
    """
    ckpt = _make_checkpoint()
    natural = _arrival_order(ckpt)
    order = [item for item in natural if item[2] == "scale"]
    order += [item for item in natural if item[2] == "weight"]

    local = {e: e for e in range(NUM_EXPERTS)}
    baseline, online, method = _run_both_paths(
        monkeypatch,
        1,
        0,
        local,
        "torch",
        order=order,
        max_pending_shards=3 * NUM_EXPERTS,
    )
    _assert_params_equal(baseline, online)
    assert method._pending_peak == 3 * NUM_EXPERTS


def test_too_many_half_loaded_shards_is_an_error(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    ckpt = _make_checkpoint()
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, _ = _make_layer(
        IluFp8ToFp4OnlineMoEMethod,
        1,
        0,
        local,
        converter_backend="torch",
        max_pending_shards=2,
    )
    weights_only = [item for item in _arrival_order(ckpt) if item[2] == "weight"]
    with pytest.raises(RuntimeError, match="max_pending_shards"):
        _feed(layer, ckpt, weights_only)


# ---------------------------------------------------------------------------
# T-5 / T-6: negative paths
# ---------------------------------------------------------------------------


def test_missing_scale_is_caught_at_finalisation(monkeypatch):
    """T-5: a shard whose scale never arrived must not be silently accepted."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    ckpt = _make_checkpoint()
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, method = _make_layer(
        IluFp8ToFp4OnlineMoEMethod, 1, 0, local, converter_backend="torch"
    )
    order = [item for item in _arrival_order(ckpt) if item != (1, "w3", "scale")]
    _feed(layer, ckpt, order)
    with pytest.raises(RuntimeError, match="never converted"):
        method.process_weights_after_loading(layer)


def test_fused_3d_expert_tensor_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, _ = _make_layer(
        IluFp8ToFp4OnlineMoEMethod, 1, 0, local, converter_backend="torch"
    )
    param = layer.w13_weight
    with pytest.raises(ValueError, match="2-D expert tensor"):
        param.weight_loader(
            param=param,
            loaded_weight=torch.zeros(
                (NUM_EXPERTS, INTERMEDIATE, HIDDEN),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            weight_name="model.layers.0.mlp.experts.w13_weight",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_biased_experts_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer = _StubRoutedExperts(1, 0, local)
    layer.moe_config.has_bias = True
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    method = IluFp8ToFp4OnlineMoEMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=list(BLOCK),
        ),
        layer,
    )
    with pytest.raises(ValueError, match="biased routed experts"):
        method.create_weights(
            layer=layer,
            num_experts=len(local),
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )


def test_unimplemented_source_format_names_the_supported_set():
    with pytest.raises(NotImplementedError, match="fp8-channel"):
        Fp8SourceLayout(
            format="fp8-channel", block_shape=(1, HIDDEN)
        ).validate_implemented()
    with pytest.raises(ValueError, match="unknown FP8 source format"):
        Fp8SourceLayout(format="int4-block", block_shape=BLOCK)


# ---------------------------------------------------------------------------
# T-6: the additional_config contract
# ---------------------------------------------------------------------------


def _glm_payload():
    """The shipped GLM-5.2 example, so the docs and the code cannot diverge."""
    import json
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "iluvatar-runtime-quant"
        / "ZhipuAI"
        / "GLM-5.2"
        / "fp4-moe-runtime.json"
    )
    return json.loads(example.read_text())


def _glm_group(**overrides):
    from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_config import (  # noqa: E501
        parse_ilu_runtime_quant_config,
    )

    payload = _glm_payload()
    group = payload["config_groups"]["glm52_moe_fp8_to_ix30_fp4"]
    for path, value in overrides.items():
        node = group
        *parents, leaf = path.split(".")
        for part in parents:
            node = node[part]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    config = parse_ilu_runtime_quant_config(
        {"source": "inline", **payload},
        model_type="glm_moe_dsa",
        quant_method="fp8",
    )
    return config.config_groups[0]


def _context(group, layer, method):
    return SimpleNamespace(
        group=group,
        layer=layer,
        original_method=method,
        base_quant_config=method.quant_config,
        record_shape_result=lambda **_: None,
    )


def test_example_config_builds_the_method(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, baseline = _make_layer(IluFp8MoEMethod, 1, 0, local)

    method = IluFp8ToFp4OnlineMoEMethod.from_ilu_runtime_config(
        _context(_glm_group(), layer, baseline)
    )
    assert isinstance(method, IluFp8ToFp4OnlineMoEMethod)
    assert method.source.format == "fp8-block"
    assert method.source.block_shape == (128, 128)
    # "auto" means "let the ixformer registry pick", which the converter
    # expresses as backend=None rather than a literal string.
    assert method.converter_backend is None
    assert method.max_pending_shards == 8


def test_example_file_yaml_sha256_matches_the_shipped_json():
    import hashlib
    from pathlib import Path

    import yaml

    example_dir = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "iluvatar-runtime-quant"
        / "ZhipuAI"
        / "GLM-5.2"
    )
    payload = yaml.safe_load((example_dir / "fp4-moe-file.yaml").read_text())
    declared = payload["additional-config"]["ilu_quant"]["sha256"]
    actual = hashlib.sha256(
        (example_dir / "fp4-moe-runtime.json").read_bytes()
    ).hexdigest()
    assert declared == f"sha256:{actual}", (
        "fp4-moe-file.yaml's sha256 is stale; recompute it after editing "
        "fp4-moe-runtime.json"
    )


def test_inline_and_file_examples_describe_the_same_groups():
    import json
    from pathlib import Path

    import yaml

    example_dir = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "iluvatar-runtime-quant"
        / "ZhipuAI"
        / "GLM-5.2"
    )
    inline = yaml.safe_load((example_dir / "fp4-moe-inline.yaml").read_text())
    inline_quant = dict(inline["additional-config"]["ilu_quant"])
    inline_quant.pop("source")
    from_json = json.loads((example_dir / "fp4-moe-runtime.json").read_text())
    assert inline_quant == from_json


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"weights.loading.stage": "process_weights_after_loading"}, "weight_load"),
        ({"weights.source.format": "fp8-channel"}, "weights.source.format"),
        ({"weights.target.group_size": 32}, "group_size"),
        ({"weights.target.scale_format": "absolute"}, "scale_format"),
        ({"iluvatar.converter_backend": "cuda"}, "converter_backend"),
        ({"iluvatar.max_pending_shards": 0}, "max_pending_shards"),
        ({"require.original_disposition": "ignored"}, "original_disposition"),
        ({"input_activations.dynamic": False}, "dynamic per-token FP8"),
    ],
)
def test_invalid_contracts_are_rejected(monkeypatch, overrides, message):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer, baseline = _make_layer(IluFp8MoEMethod, 1, 0, local)
    with pytest.raises(ValueError, match=message):
        IluFp8ToFp4OnlineMoEMethod.from_ilu_runtime_config(
            _context(_glm_group(**overrides), layer, baseline)
        )
