# SPDX-License-Identifier: Apache-2.0
"""Native BF16 -> ix30 FP4 routed-expert loading (MiniMax-M3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_to_fp4 import (
    Fp8SourceLayout,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp16_to_fp4_moe import (
    IluFp16ToFp4OnlineMoEMethod,
    convert_native_to_ix30_fp4,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_base import (
    IluRuntimeQuantBaseConfig,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_config import (
    parse_ilu_runtime_quant_config,
)

BLOCK = (128, 128)
HIDDEN = 256
INTERMEDIATE = 256
NUM_EXPERTS = 4
EXAMPLE_DIR = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "iluvatar-runtime-quant"
    / "MiniMaxAI"
    / "MiniMax-M3"
)


class _ExpertMapManager:
    def __init__(self, global_to_local: dict[int, int]):
        self._map = global_to_local

    def map_global_to_local(self, expert_id: int) -> int:
        return self._map.get(expert_id, -1)


class _StubRoutedExperts(RoutedExperts):
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        local_experts: dict[int, int],
        hidden: int = HIDDEN,
        intermediate: int = INTERMEDIATE,
    ):
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            is_act_and_mul=True,
            w13_num_shards=2,
            tp_shard_with_padding=False,
            intermediate_size=intermediate,
            intermediate_size_per_partition=intermediate // tp_size,
            moe_backend="auto",
            has_bias=False,
            hidden_dim=hidden,
            hidden_dim_unpadded=hidden,
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
        self.activation = "swigluoai_uninterleave"
        self.prefix = "language_model.model.layers.3.block_sparse_moe.experts"
        self._loaded_expert_biases = set()


def _minimax_payload():
    import json

    return json.loads((EXAMPLE_DIR / "fp8-fp4-runtime.json").read_text())


def _moe_group(**overrides):
    payload = _minimax_payload()
    group = payload["config_groups"]["minimax_m3_bf16_moe_to_ix30_fp4"]
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
        model_type="minimax_m3_text",
        quant_method="ilu_runtime",
    )
    return [g for g in config.config_groups if g.group_id.endswith("moe_to_ix30_fp4")][
        0
    ]


def _context(group, layer):
    return SimpleNamespace(
        group=group,
        layer=layer,
        original_method=object.__new__(UnquantizedFusedMoEMethod),
        base_quant_config=IluRuntimeQuantBaseConfig(),
        record_shape_result=lambda **_: None,
    )


def test_example_config_builds_the_method():
    layer = _StubRoutedExperts(1, 0, {e: e for e in range(NUM_EXPERTS)})
    method = IluFp16ToFp4OnlineMoEMethod.from_ilu_runtime_config(
        _context(_moe_group(), layer)
    )
    assert type(method) is IluFp16ToFp4OnlineMoEMethod
    assert method.source.format == "fp8-block"
    assert method.source.block_shape == BLOCK
    assert method.converter_backend is None
    assert method.max_pending_shards == 8
    assert method.maybe_make_prepare_finalize() is None
    assert method.get_fused_moe_quant_config(layer) is None


def test_factory_requires_unquantized_original_method():
    layer = _StubRoutedExperts(1, 0, {0: 0})
    ctx = _context(_moe_group(), layer)
    ctx.original_method = object()
    with pytest.raises(TypeError, match="UnquantizedFusedMoEMethod"):
        IluFp16ToFp4OnlineMoEMethod.from_ilu_runtime_config(ctx)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"weights.loading.stage": "process_weights_after_loading"}, "weight_load"),
        ({"weights.source.format": "fp8-block"}, "weights.source.format"),
        ({"weights.source.dtypes": ["float16"]}, "weights.source.dtypes"),
        ({"require.original_disposition": "quantized"}, "original_disposition"),
        ({"weights.target.group_size": 32}, "group_size"),
        ({"iluvatar.converter_backend": "cuda"}, "converter_backend"),
        ({"iluvatar.fp8_block_shape": [128]}, "fp8_block_shape"),
    ],
)
def test_invalid_contracts_are_rejected(overrides, message):
    layer = _StubRoutedExperts(1, 0, {0: 0})
    with pytest.raises((ValueError, TypeError), match=message):
        IluFp16ToFp4OnlineMoEMethod.from_ilu_runtime_config(
            _context(_moe_group(**overrides), layer)
        )


def test_example_file_yaml_sha256_matches_the_shipped_json():
    import hashlib

    import yaml

    payload = yaml.safe_load((EXAMPLE_DIR / "fp8-fp4-file.yaml").read_text())
    declared = payload["additional-config"]["ilu_quant"]["sha256"]
    actual = hashlib.sha256(
        (EXAMPLE_DIR / "fp8-fp4-runtime.json").read_bytes()
    ).hexdigest()
    assert declared == f"sha256:{actual}"


def test_inline_and_file_examples_describe_the_same_groups():
    import json

    import yaml

    inline = yaml.safe_load((EXAMPLE_DIR / "fp8-fp4-inline.yaml").read_text())
    inline_quant = dict(inline["additional-config"]["ilu_quant"])
    inline_quant.pop("source")
    from_json = json.loads((EXAMPLE_DIR / "fp8-fp4-runtime.json").read_text())
    assert inline_quant == from_json


def _has_ixmm_ivcore30() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from vllm_iluvatar.ilu_util.arch import is_ivcore30
        from vllm_iluvatar.model_executor.layers.quantization.fp8 import (
            _ixmm_backend_available,
        )

        return is_ivcore30() and _ixmm_backend_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_ixmm_ivcore30(),
    reason="requires an ivcore30 GPU with ixmm backend",
)
@pytest.mark.parametrize("backend", ["torch", "ixformer_native"])
def test_load_time_bf16_matches_explicit_native_to_fp4(backend, monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_iluvatar.model_executor.layers.quantization.fp8 import (
        _ixmm_backend_available,
    )

    if not _ixmm_backend_available():
        pytest.skip("ixmm backend unavailable")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    local = {e: e for e in range(NUM_EXPERTS)}
    layer = _StubRoutedExperts(1, 0, local)
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(BLOCK),
    )
    method = IluFp16ToFp4OnlineMoEMethod(
        quant_config,
        layer,
        source=Fp8SourceLayout(format="fp8-block", block_shape=BLOCK),
        converter_backend=backend,
    )
    layer.quant_method = method
    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            num_experts=len(local),
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )

    g = torch.Generator(device="cuda").manual_seed(7)
    ckpt = {}
    for e in range(NUM_EXPERTS):
        ckpt[(e, "w1")] = torch.randn(
            INTERMEDIATE, HIDDEN, generator=g, dtype=torch.bfloat16, device="cuda"
        )
        ckpt[(e, "w3")] = torch.randn(
            INTERMEDIATE, HIDDEN, generator=g, dtype=torch.bfloat16, device="cuda"
        )
        ckpt[(e, "w2")] = torch.randn(
            HIDDEN, INTERMEDIATE, generator=g, dtype=torch.bfloat16, device="cuda"
        )

    for e in range(NUM_EXPERTS):
        for shard in ("w1", "w3", "w2"):
            param_name = "w13_weight" if shard in ("w1", "w3") else "w2_weight"
            param = getattr(layer, param_name)
            param.weight_loader(
                param=param,
                loaded_weight=ckpt[(e, shard)],
                weight_name=f"experts.{param_name}",
                shard_id=shard,
                expert_id=e,
                return_success=True,
            )
    # Community can invoke the loader twice for the same shard. A second
    # convert from a fresh zeros staging used to poison FP4 scales with Inf.
    for e in range(NUM_EXPERTS):
        for shard in ("w1", "w3", "w2"):
            param_name = "w13_weight" if shard in ("w1", "w3") else "w2_weight"
            param = getattr(layer, param_name)
            param.weight_loader(
                param=param,
                loaded_weight=ckpt[(e, shard)],
                weight_name=f"experts.{param_name}",
                shard_id=shard,
                expert_id=e,
                return_success=True,
            )
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight_scale_inv")

    for e in range(NUM_EXPERTS):
        for shard, native in (
            ("w1", ckpt[(e, "w1")]),
            ("w3", ckpt[(e, "w3")]),
            ("w2", ckpt[(e, "w2")]),
        ):
            ref_w, ref_s = convert_native_to_ix30_fp4(native, backend=backend)
            if shard == "w2":
                got_w = layer.w2_weight.data[e]
                got_s = layer.w2_weight_scale.data[e]
            else:
                size = INTERMEDIATE
                offset = 0 if shard == "w1" else size
                got_w = layer.w13_weight.data[e].narrow(1, offset, size)
                got_s = layer.w13_weight_scale.data[e].narrow(1, offset, size)
            assert torch.equal(got_w, ref_w), shard
            assert torch.equal(got_s, ref_s), shard


def _swigluoai_ref(gate_up, limit, alpha, beta):
    """Reference for the uninterleaved SwiGLU-OAI used by MiniMax-M3."""
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d], gate_up[..., d:]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def _bf16_moe_reference(x, ckpt, topk_weights, topk_ids, limit, alpha, beta):
    """Dense BF16 MoE using the original (un-quantized) expert weights."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            w = float(topk_weights[t, k])
            gate_up = torch.cat(
                [
                    x[t].float() @ ckpt[(e, "w1")].float().T,
                    x[t].float() @ ckpt[(e, "w3")].float().T,
                ],
                dim=-1,
            )
            act = _swigluoai_ref(gate_up, limit, alpha, beta)
            out[t] += w * (act @ ckpt[(e, "w2")].float().T)
    return out


@pytest.mark.skipif(
    not _has_ixmm_ivcore30(),
    reason="requires an ivcore30 GPU with ixmm backend",
)
@pytest.mark.parametrize(
    "hidden,inter,experts",
    [
        (HIDDEN, INTERMEDIATE, NUM_EXPERTS),
        # MiniMax-M3 routed-expert shapes (config hidden_size /
        # intermediate_size); EP keeps intermediate un-sharded.
        (6144, 3072, 2),
    ],
    ids=["small", "m3_shapes"],
)
def test_load_time_fp4_moe_matches_bf16_reference(hidden, inter, experts, monkeypatch):
    """The served FP4 MoE must track a BF16 reference, not just be self-consistent.

    ``test_load_time_bf16_matches_explicit_fp8_then_fp4`` only proves the loader
    reproduces ``convert_native_to_ix30_fp4`` byte for byte. That stays green even when
    the served numerics are wrong, which is how a garbage-output regression can
    hide. This drives the real ``apply()`` and compares against the original
    BF16 weights.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    torch.manual_seed(0)
    local = {e: e for e in range(experts)}
    layer = _StubRoutedExperts(1, 0, local, hidden=hidden, intermediate=inter)
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(BLOCK),
    )
    method = IluFp16ToFp4OnlineMoEMethod(
        quant_config,
        layer,
        source=Fp8SourceLayout(format="fp8-block", block_shape=BLOCK),
    )
    layer.quant_method = method
    with torch.device("cuda"):
        method.create_weights(
            layer=layer,
            num_experts=len(local),
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )

    g = torch.Generator(device="cuda").manual_seed(7)
    ckpt = {}
    for e in range(experts):
        for shard, shape in (
            ("w1", (inter, hidden)),
            ("w3", (inter, hidden)),
            ("w2", (hidden, inter)),
        ):
            ckpt[(e, shard)] = (
                torch.randn(shape, generator=g, dtype=torch.bfloat16, device="cuda")
                * 0.05
            )

    for e in range(experts):
        for shard in ("w1", "w3", "w2"):
            param_name = "w13_weight" if shard in ("w1", "w3") else "w2_weight"
            param = getattr(layer, param_name)
            param.weight_loader(
                param=param,
                loaded_weight=ckpt[(e, shard)],
                weight_name=f"experts.{param_name}",
                shard_id=shard,
                expert_id=e,
                return_success=True,
            )
    method.process_weights_after_loading(layer)

    # MiniMax-M3 activation parameters (config.swiglu_*).
    layer.swiglu_limit, layer.swiglu_alpha, layer.swiglu_beta = 7.0, 1.702, 1.0
    layer._expert_map = None
    layer._ilu_output_scale = 1.0

    num_tokens, top_k = 8, 2
    x = torch.randn(
        num_tokens, hidden, generator=g, dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = torch.randint(
        0,
        experts,
        (num_tokens, top_k),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        num_tokens, top_k, generator=g, dtype=torch.float32, device="cuda"
    )
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    got = method.apply(layer, x, topk_weights, topk_ids, None, None).float()
    ref = _bf16_moe_reference(
        x,
        ckpt,
        topk_weights,
        topk_ids,
        layer.swiglu_limit,
        layer.swiglu_alpha,
        layer.swiglu_beta,
    )

    assert torch.isfinite(got).all(), "FP4 MoE produced non-finite output"
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0
    ).item()
    rel = ((got - ref).norm() / ref.norm().clamp(min=1e-6)).item()
    assert cos > 0.95, f"FP4 MoE decorrelated from BF16 reference: cos={cos:.4f}"
    assert rel < 0.35, f"FP4 MoE relative error too large: {rel:.4f}"
