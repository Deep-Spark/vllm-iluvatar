# SPDX-License-Identifier: Apache-2.0
"""FP8 load-time quantization, TP/EP placement and NN graph GEMM contracts."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_routed_experts import (
    IluRoutedExperts,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.fp8_moe import (
    IluFp8MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import IluFp8Config
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.moe.fp8_to_w4a8_nn import (
    IluFp8ToW4A8NNOnlineMoEMethod,
    quantize_fp8_to_w4a8_nn,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

# IluW4A8NNMoEMethod 自带 require_ivcore30_nn() 硬校验；只有真正构造
# NN method 的用例需要 ivcore30（CQ），其余用例在 ivcore11 上照常跑。
_requires_ivcore30 = pytest.mark.skipif(
    not arch.is_ivcore30(),
    reason="requires ivcore30 (CQ); W4A8 NN MoE method",
)


def runtime_payload():
    # Self-contained test fixture; no dependency on local deployment files.
    return {
        "source": "inline",
        "schema_version": 1,
        "enabled": True,
        "applies_to": {"model_types": ["glm_moe_dsa"], "quant_methods": ["fp8"]},
        "ignore": [],
        "config_groups": {
            "glm52_fp8_w4a8_nn": {
                "priority": 100,
                "targets": [r"re:.*\.mlp\.experts$"],
                "require": {
                    "original_disposition": "quantized",
                    "checkpoint_dtypes": ["float8_e4m3fn"],
                },
                "weights": {
                    "source": {
                        "format": "fp8-block",
                        "dtypes": ["float8_e4m3fn"],
                        "block_shape": [128, 128],
                    },
                    "target": {
                        "format": "int4-packed-nn",
                        "dtype": "int8",
                        "strategy": "channel",
                        "pack_version": 1,
                    },
                    "loading": {"mode": "transform", "stage": "weight_load"},
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "dynamic": True,
                },
                "output_activations": None,
                "iluvatar": {
                    "method": "IluFp8ToW4A8NNOnlineMoEMethod",
                    "max_pending_shards": 8,
                },
            }
        },
    }


class Layer(IluRoutedExperts):
    def __init__(self, tp=1, rank=0, intermediate=2048):
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            has_bias=False,
            is_act_and_mul=True,
            w13_num_shards=2,
            tp_shard_with_padding=False,
            intermediate_size=intermediate,
            intermediate_size_per_partition=intermediate // tp,
            moe_backend="auto",
            hidden_dim=256,
            hidden_dim_unpadded=256,
            intermediate_size_per_partition_unpadded=None,
            num_local_experts=2,
            tp_size=tp,
            tp_rank=rank,
            moe_parallel_config=SimpleNamespace(tp_size=tp),
        )
        self.expert_map_manager = SimpleNamespace(
            map_global_to_local=lambda e: {1: 0, 3: 1}.get(e, -1)
        )
        self._expert_map = None
        self.rocm_aiter_fmoe_enabled = False
        self.quant_config = None
        self.quant_method = None
        self.activation = "silu"
        self.prefix = "model.layers.3.mlp.experts"
        self._loaded_expert_biases = set()


def select(tp=1, rank=0, config_mutation=None, intermediate=2048):
    layer = Layer(tp, rank, intermediate=intermediate)
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"), dtype=torch.bfloat16
    )
    config.additional_config = {"ilu_quant": runtime_payload()}
    if config_mutation:
        config_mutation(config.additional_config["ilu_quant"])
    quant = IluFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    with set_current_vllm_config(config):
        method = quant.get_quant_method(layer, layer.prefix)
    layer.quant_method = method
    return layer, method


@_requires_ivcore30
def test_dispatch():
    _, method = select()
    assert isinstance(method, IluFp8ToW4A8NNOnlineMoEMethod)
    assert method.gemm_format == "NN"
    assert not hasattr(method, "version")
    _, original = select(config_mutation=lambda c: c.update(enabled=False))
    assert isinstance(original, IluFp8MoEMethod)
    assert not isinstance(original, IluFp8ToW4A8NNOnlineMoEMethod)


def test_canonical_class_and_config_name():
    assert (
        IluvatarQuantRegistry.get_method("IluFp8ToW4A8NNOnlineMoEMethod")
        is IluFp8ToW4A8NNOnlineMoEMethod
    )


@_requires_ivcore30
def test_runtime_target_can_omit_pack_version():
    def mutate(config):
        del config["config_groups"]["glm52_fp8_w4a8_nn"]["weights"]["target"][
            "pack_version"
        ]

    _, method = select(config_mutation=mutate)
    assert isinstance(method, IluFp8ToW4A8NNOnlineMoEMethod)
    assert not hasattr(method, "version")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("source", "format", "int4-packed-tn"),
        ("target", "pack_version", 2),
        ("target", "typo", True),
        ("loading", "stage", "process_weights_after_loading"),
        ("input_activations", "type", "float"),
        ("input_activations", "dynamic", False),
        ("iluvatar", "decode_backend", "gemv"),
        ("iluvatar", "max_pending_shards", True),
    ],
)
def test_invalid_contract(section, key, value):
    def mutate(c):
        group = c["config_groups"]["glm52_fp8_w4a8_nn"]
        (
            group["weights"][section]
            if section in {"source", "target", "loading"}
            else group[section]
        )[key] = value

    with pytest.raises(ValueError):
        select(config_mutation=mutate)


def reference(weight, scale):
    w = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    s8 = w.abs().amax(-1, keepdim=True) / 127
    s8[s8 == 0] = 1
    q8 = torch.round(w / s8).clamp(-127, 127)
    s4 = torch.round(q8.abs().amax(-1, keepdim=True) / 7).clamp_min(1)
    q4 = torch.round(q8 / s4).clamp(-8, 7)
    return q4, (s8 * s4).T


def unpack_nn(packed):
    k, half_n = packed.shape
    n = 2 * half_n
    # Independent signed-nibble decoder for validating the online packer.
    tiles = packed.reshape(k, n // 32, 16).to(torch.int16)
    low = (tiles & 15) - ((tiles & 8) << 1)
    high = tiles >> 4
    swizzled = torch.cat((low, high), -1).reshape(k, n)
    return (
        swizzled.reshape(k // 32, 2, 16, n // 32, 2, 16)
        .permute(0, 4, 2, 3, 1, 5)
        .reshape(k, n)
        .T
    )


@pytest.mark.parametrize("zero", [False, True])
def test_quantizer_reference(zero):
    torch.manual_seed(9)
    w = (torch.zeros(128, 256) if zero else torch.randn(128, 256)).to(
        torch.float8_e4m3fn
    )
    blocks = torch.rand(1, 2) * 0.03
    packed, scale = quantize_fp8_to_w4a8_nn(w, blocks, (128, 128))
    q4, expected_scale = reference(w, blocks)
    torch.testing.assert_close(unpack_nn(packed).float(), q4, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


def test_tp_keeps_both_global_scales():
    # The first shard has neither the full-channel amax nor the INT8 amax=127.
    # A local NN/v1 repacker would recompute its second scale and change q4.
    w = torch.ones(128, 512)
    w[:, 256:] *= 32
    w[0] = 0  # also pin the all-zero channel behavior
    w = w.to(torch.float8_e4m3fn)
    blocks = torch.ones(1, 4)
    full_q, full_s = reference(w, blocks)
    for rank in range(4):
        packed, scale = quantize_fp8_to_w4a8_nn(
            w, blocks, (128, 128), k_start=rank * 128, k_size=128, pad_k=256
        )
        expected = torch.nn.functional.pad(
            full_q[:, rank * 128 : (rank + 1) * 128], (0, 128)
        )
        torch.testing.assert_close(unpack_nn(packed).float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scale, full_s, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.getenv("ILU_W4A8_FP8_MODEL"),
    reason="opt-in real checkpoint and independent NN quantizer comparison",
)
@pytest.mark.parametrize("tp", [8, 16])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_real_glm_tp_against_native_nn_quantizer(tp, projection):
    import ixformer.inference.functions as ops
    from safetensors import safe_open

    root = Path(os.environ["ILU_W4A8_FP8_MODEL"])
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = f"model.layers.3.mlp.experts.0.{projection}.weight"
    with safe_open(str(root / index[name]), framework="pt", device="cuda") as f:
        weight = f.get_tensor(name)
    with safe_open(
        str(root / index[name + "_scale_inv"]), framework="pt", device="cuda"
    ) as f:
        blocks = f.get_tensor(name + "_scale_inv")
    # Full-channel FP32 -> INT8 reference, then native direct NN/v1 packing.
    # No TN checkpoint or plugin relayout participates in the reference.
    value = weight.float() * blocks.repeat_interleave(128, 0).repeat_interleave(128, 1)
    scale = value.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q8 = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    packed_nn, extra, _ = ops.quant_repack_int4(q8[None], -1, 1, "NN", False)
    nn = packed_nn[0]
    scale = (scale[:, 0] * extra[0])[None]
    for rank in range(tp):
        if projection == "down_proj":
            size = weight.shape[1] // tp
            packed, actual_scale = quantize_fp8_to_w4a8_nn(
                weight,
                blocks,
                (128, 128),
                k_start=rank * size,
                k_size=size,
                pad_k=max(256, size),
            )
            expected = torch.nn.functional.pad(
                nn[rank * size : (rank + 1) * size], (0, 0, 0, max(256, size) - size)
            )
            expected_scale = scale
        else:
            size = weight.shape[0] // tp
            packed, actual_scale = quantize_fp8_to_w4a8_nn(
                weight[rank * size : (rank + 1) * size],
                blocks[rank * size // 128 : (rank + 1) * size // 128],
                (128, 128),
            )
            expected = nn[:, rank * size // 2 : (rank + 1) * size // 2]
            expected_scale = scale[:, rank * size : (rank + 1) * size]
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


def feed(layer, tensor, shard, expert, scale=False):
    group = "w2" if shard == "w2" else "w13"
    name = f"{group}_weight" + ("_scale_inv" if scale else "")
    param = getattr(layer, name)
    return param.weight_loader(param, tensor, name, shard, expert, return_success=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1), (4, 3), (8, 7), (16, 15)])
@pytest.mark.parametrize("scale_first", [False, True])
@_requires_ivcore30
def test_streaming_tp_ep_and_graph(tp, rank, scale_first, monkeypatch):
    import ixformer.inference.functions as ops

    layer, method = select(tp, rank)
    i = 2048 // tp
    with torch.device("cuda"):
        method.create_weights(
            layer, 2, 256, i, torch.bfloat16, weight_loader=layer.weight_loader
        )
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight_scale.ilu_derived_at_load
    with pytest.raises(RuntimeError, match="missing FP8"):
        method.process_weights_after_loading(layer)
    refs = {}
    torch.manual_seed(23)
    for expert in range(4):
        for shard in ("w3", "w2", "w1"):
            shape = (256, 2048) if shard == "w2" else (2048, 256)
            w = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
            s = torch.rand(shape[0] // 128, shape[1] // 128, device="cuda") * 0.01
            if scale_first:
                ok = feed(layer, s, shard, expert, True)
                feed(layer, w, shard, expert)
            else:
                ok = feed(layer, w, shard, expert)
                feed(layer, s, shard, expert, True)
            assert ok == (expert in (1, 3))
            if not ok:
                continue
            if shard == "w2":
                # Offline reference quantizes the FULL K, then slices INT4.
                q, sc = reference(w, s)
                q = q[:, rank * i : (rank + 1) * i]
                q = torch.nn.functional.pad(q, (0, max(256, i) - i))
                refs[expert, shard] = (q, sc)
            else:
                w, s = (
                    w[rank * i : (rank + 1) * i],
                    s[rank * i // 128 : (rank + 1) * i // 128],
                )
                refs[expert, shard] = reference(w, s)
    method.process_weights_after_loading(layer)
    assert not method._weights and not method._scales
    assert not hasattr(layer, "w13_weight_scale_inv")
    method._runtime_context.assert_validation_complete()
    for group, info in (("w13", method.gemm1_info), ("w2", method.gemm2_info)):
        weight = getattr(layer, group + "_weight")
        scale = getattr(layer, group + "_weight_scale")
        qrefs, srefs = [], []
        for local, e in enumerate((1, 3)):
            if group == "w13":
                q = torch.cat([refs[e, s][0] for s in ("w1", "w3")])
                sc = torch.cat([refs[e, s][1] for s in ("w1", "w3")], dim=1)
            else:
                q, sc = refs[e, "w2"]
            torch.testing.assert_close(
                unpack_nn(weight[local]).float(), q, rtol=0, atol=0
            )
            torch.testing.assert_close(scale[local], sc, rtol=0, atol=0)
            qrefs.append(q)
            srefs.append(sc)
        a = torch.randint(-127, 128, (7, info.pad_k), device="cuda", dtype=torch.int8)
        sa = torch.rand(7, device="cuda") * 0.01
        layout = torch.tensor([[0, 3], [3, 7]], device="cuda", dtype=torch.int32)
        calls = []
        original = ops.moe_w4a8_group_gemm_ixmm

        def wrapped(*args, layout=layout, calls=calls, original=original, **kwargs):
            assert kwargs["backend"] == "ixmm"
            assert kwargs["group_layout"] is layout
            calls.append(True)
            return original(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("decode must not execute GEMV")

        with monkeypatch.context() as patch:
            patch.setattr(ops, "moe_w4a8_group_gemm_ixmm", wrapped)
            patch.setattr(ops, "moe_w4a8_group_gemv", forbidden)

            def launch(
                info=info, a=a, sa=sa, weight=weight, scale=scale, layout=layout
            ):
                return method._group_gemm(
                    info, a, sa, weight, scale, layout, 7, torch.bfloat16
                )

            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = launch()
            assert len(calls) == 2
        for changed in (False, True):
            if changed:
                a.random_(-127, 128)
                layout.copy_(
                    torch.tensor([[0, 0], [0, 7]], device="cuda", dtype=torch.int32)
                )
            graph.replay()
            torch.cuda.synchronize()
            for e, (lo, hi) in enumerate(layout.tolist()):
                expected = (a[lo:hi].float() @ qrefs[e].T) * sa[lo:hi, None] * srefs[e]
                torch.testing.assert_close(
                    out[lo:hi], expected.to(out.dtype), atol=0.008, rtol=0.008
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@_requires_ivcore30
def test_invalid_dtype_and_pending():
    layer, method = select(intermediate=512)
    with torch.device("cuda"):
        method.create_weights(
            layer, 2, 256, 512, torch.bfloat16, weight_loader=layer.weight_loader
        )
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        feed(layer, torch.zeros(512, 256, device="cuda"), "w1", 1)
    w = torch.zeros(512, 256, device="cuda").to(torch.float8_e4m3fn)
    method.max_pending_shards = 1
    feed(layer, w, "w1", 1)
    with pytest.raises(ValueError, match="duplicate"):
        feed(layer, w, "w1", 1)
    with pytest.raises(RuntimeError, match="max_pending"):
        feed(layer, w, "w3", 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("pack_version", [None, 1])
@_requires_ivcore30
def test_static_nn_load_matches_online(tp, rank, pack_version, monkeypatch):
    import ixformer.inference.functions as ops

    from vllm_iluvatar.model_executor.layers.quantization.backends.w4a8 import (
        nn_moe as w4a8,
    )

    online, method = select(tp, rank, intermediate=512)
    static = Layer(tp, rank, intermediate=512)
    ilu_params = {"gemm_format": "NN"}
    if pack_version is not None:
        ilu_params["pack_version"] = pack_version
    static_method = w4a8.IluW4A8NNMoEMethod(
        method.weight_quant,
        method.input_quant,
        static.moe_config,
        ilu_params=ilu_params,
    )
    static.quant_method = static_method
    with torch.device("cuda"):
        for layer, m in ((online, method), (static, static_method)):
            m.create_weights(
                layer,
                2,
                256,
                512 // tp,
                torch.bfloat16,
                weight_loader=layer.weight_loader,
            )
    torch.manual_seed(8)
    for e in (1, 3):
        for shard in ("w1", "w2", "w3"):
            n, k = (256, 512) if shard == "w2" else (512, 256)
            w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            blocks = torch.rand(n // 128, k // 128, device="cuda") * 0.01
            value = w.float() * blocks.repeat_interleave(128, 0).repeat_interleave(
                128, 1
            )
            s = value.abs().amax(-1, keepdim=True) / 127
            q8 = torch.round(value / s).clamp(-127, 127).to(torch.int8)
            packed, extra, _ = ops.quant_repack_int4(q8[None], -1, 1, "NN", False)
            group = "w2" if shard == "w2" else "w13"
            for suffix, tensor in (
                ("weight", packed[0]),
                ("weight_scale", (s[:, 0] * extra[0])[None]),
            ):
                name = group + "_" + suffix
                param = getattr(static, name)
                assert param.weight_loader(
                    param, tensor, name, shard, e, return_success=True
                )
            feed(online, w, shard, e)
            feed(online, blocks, shard, e, True)
    for layer, m in ((online, method), (static, static_method)):
        m.process_weights_after_loading(layer)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        torch.testing.assert_close(
            getattr(static, name), getattr(online, name), rtol=0, atol=0
        )
    assert not hasattr(static_method, "version")
