# SPDX-License-Identifier: Apache-2.0
"""MXFP4/TN numeric layout and bounded loader contracts (CPU unless marked)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_iluvatar.model_executor.layers.quantization.ilu_deepseek_v4_fp8 import (
    IluDeepseekV4FP8Config,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 import IluMxfp4MoEMethod
from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4_to_w4a8_moe import (
    IluMxfp4ToW4A8TNMoEMethod,
    quantize_mxfp4_to_w4a8_tn,
)


def source(n=256, k=256):
    generator = torch.Generator().manual_seed(56075618)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    scale = torch.randint(
        119, 128, (n, k // 32), dtype=torch.uint8, generator=generator
    )
    return weight, scale


def oracle(weight, scale):
    # Independent scalar E2M1 decoding, then explicit TN byte addressing.
    table = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    rows, packed = [], []
    for w, s in zip(weight.tolist(), scale.tolist()):
        row = []
        for j, byte in enumerate(w):
            for code in (byte & 15, byte >> 4):
                row.append(
                    table[code & 7]
                    * (-1 if code & 8 else 1)
                    * 2.0 ** (s[j // 16] - 127)
                )
        channel = max(map(abs, row)) / 127 or 1.0
        q4 = [max(-8, min(7, round(round(value / channel) / 18))) for value in row]
        packed.append(
            [
                (q4[k + j] & 15) | ((q4[k + j + 16] & 15) << 4)
                for k in range(0, len(row), 32)
                for j in range(16)
            ]
        )
        rows.append(channel)
    return torch.tensor(packed, dtype=torch.uint8).view(torch.int8), torch.tensor(
        [rows]
    )


@pytest.mark.parametrize("n,k", [(7, 32), (19, 96), (2048, 4096), (4096, 2048)])
def test_numeric_tn_layout(n, k):
    w, s = source(n, k)
    original = (w.clone(), s.clone())
    signed = quantize_mxfp4_to_w4a8_tn(w.view(torch.int8), s)
    packed, channel = quantize_mxfp4_to_w4a8_tn(w, s)
    torch.testing.assert_close(signed[0], packed, atol=0, rtol=0)
    torch.testing.assert_close(signed[1], channel, atol=0, rtol=0)
    # Scalar oracle on diverse rows; full-size GPU compute is tested separately.
    selected = [0, n // 2, n - 1]
    expected, expected_scale = oracle(w[selected], s[selected])
    torch.testing.assert_close(packed[selected], expected, atol=0, rtol=0)
    torch.testing.assert_close(channel[:, selected], expected_scale, atol=0, rtol=1e-7)
    assert torch.equal(w, original[0]) and torch.equal(s, original[1])
    # Negative control: TN/v2 scale is INT8 scale; factor 18 belongs to kernel.
    assert not torch.allclose(channel[:, selected] * 18, expected_scale)


def test_zero_and_every_e2m1_code():
    w = torch.arange(0, 256, dtype=torch.uint8).reshape(16, 16)
    s = torch.arange(119, 135, dtype=torch.uint8).reshape(16, 1)
    got = quantize_mxfp4_to_w4a8_tn(w, s)
    expected = oracle(w, s)
    torch.testing.assert_close(got[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(got[1], expected[1], atol=0, rtol=1e-7)
    w.zero_()
    packed, channel = quantize_mxfp4_to_w4a8_tn(w, s)
    assert not packed.count_nonzero()
    assert torch.equal(channel, torch.ones_like(channel))


@pytest.mark.parametrize("n,k", [(4096, 4096), (4096, 2048)])
def test_device_conversion_and_tn_group_gemm(n, k):
    # TP1 V4 W13/W2 dimensions. Unequal expert counts expose scale/row mapping.
    import ixformer.functions as ixf
    from ixformer.inference.backend_registry import registry

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    assert "ixinfer" in registry.available_backends("moe_w4a8_group_gemm")
    weights, scales, decoded = [], [], []
    for expert in range(2):
        w, s = source(n, k)
        s = s - expert
        cpu_packed, cpu_scale = quantize_mxfp4_to_w4a8_tn(w, s)
        gpu_packed, gpu_scale = quantize_mxfp4_to_w4a8_tn(
            w.view(torch.int8).cuda(), s.cuda()
        )
        torch.testing.assert_close(gpu_packed.cpu(), cpu_packed, atol=0, rtol=0)
        torch.testing.assert_close(gpu_scale.cpu(), cpu_scale, atol=0, rtol=0)
        # Independent TN addressing: each 16-byte tile contains 32 K values.
        tiles = cpu_packed.view(torch.uint8).reshape(n, k // 32, 16)
        low, high = (tiles & 15).int(), (tiles >> 4).int()
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        decoded.append(torch.cat((low, high), -1).reshape(n, k).float())
        weights.append(gpu_packed)
        scales.append(gpu_scale)
    generator = torch.Generator().manual_seed(20261026)
    inputs = torch.randint(-127, 128, (20, k), generator=generator).to(torch.int8)
    input_scales = torch.linspace(0.001, 0.007, 20)
    expected = torch.cat(
        [
            (inputs[a:b].float() @ decoded[e].T)
            * input_scales[a:b, None]
            * scales[e].cpu()
            * 18
            for e, (a, b) in enumerate(((0, 3), (3, 20)))
        ]
    ).to(torch.bfloat16)
    got = ixf.moe_w4a8_group_gemm(
        inputs.cuda(),
        torch.stack(weights),
        input_scales.cuda(),
        torch.stack(scales),
        torch.bfloat16,
        torch.tensor([3, 17], dtype=torch.int32),
        format=2,
        version=2,
        group_size=-1,
        backend="ixinfer",
    )
    assert torch.isfinite(got).all()
    # Integer dot products are exact in FP32 here; allow one BF16 ULP for
    # kernel scale multiplication/output rounding, including near-zero floor.
    torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1 / 128)
    assert not torch.allclose(got.float().cpu(), expected.float() / 18)


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "overflow"])
def test_reject_invalid_source(bad):
    w, s = source(2, 32)
    if bad == "dtype":
        w = w.float()
    elif bad == "shape":
        s = s[:, :0]
    elif bad == "nan":
        s[0, 0] = 255
    else:
        w.fill_(0x77)
        s.fill_(254)
    with pytest.raises(ValueError):
        quantize_mxfp4_to_w4a8_tn(w, s)


class Layer(torch.nn.Module):
    def __init__(self, tp=1, ep=1):
        super().__init__()
        self.moe_config = SimpleNamespace(
            has_bias=False,
            is_act_and_mul=True,
            tp_size=tp,
            ep_size=ep,
            ep_rank=1 if ep > 1 else 0,
            hidden_dim_unpadded=256,
            intermediate_size_per_partition_unpadded=256,
        )
        self.expert_map = None
        if ep > 1:
            self.expert_map = torch.full((2 * ep,), -1, dtype=torch.int32)
            self.expert_map[2:4] = torch.arange(2, dtype=torch.int32)

    def _map_global_expert_id_to_local_expert_id(self, expert):
        if self.expert_map is not None:
            return (
                int(self.expert_map[expert])
                if 0 <= expert < len(self.expert_map)
                else -1
            )
        return {2: 0, 5: 1}.get(expert, -1)


def method(tp=1, pending=8, ep=1):
    layer = Layer(tp, ep)
    impl = IluMxfp4ToW4A8TNMoEMethod(layer, max_pending_shards=pending)
    with set_current_vllm_config(VllmConfig()):
        impl.create_weights(layer, 2, 256, 256, torch.bfloat16)
    return layer, impl


def load(layer, shard, expert, *, scale=False, tensor=None):
    group = "w2" if shard == "w2" else "w13"
    name = f"{group}_weight" + ("_scale" if scale else "")
    param = getattr(layer, name)
    tensor = source()[int(scale)] if tensor is None else tensor
    return param.weight_loader(param, tensor, name, shard, expert, return_success=True)


@pytest.mark.parametrize(
    "scale_first,ep", [(False, 1), (True, 1), (False, 8), (True, 8)]
)
def test_pair_order_placement_and_finalization(scale_first, ep):
    layer, impl = method(pending=1, ep=ep)
    assert not load(layer, "w1", 99)
    assert not impl._pending
    for expert in (3, 2) if ep > 1 else (5, 2):
        for shard in ("w3", "w2", "w1"):
            for scale in (scale_first, not scale_first):
                assert load(layer, shard, expert, scale=scale)
            assert not impl._pending
    assert len(impl._converted) == 6
    expected, scales = quantize_mxfp4_to_w4a8_tn(*source())
    torch.testing.assert_close(layer.w13_weight[0, :256], expected, atol=0, rtol=0)
    torch.testing.assert_close(layer.w13_weight[1, 256:], expected, atol=0, rtol=0)
    torch.testing.assert_close(layer.w2_weight_scale[1], scales, atol=0, rtol=0)
    impl.process_weights_after_loading(layer)
    assert impl._finalized
    with pytest.raises(RuntimeError, match="unfinalized"):
        impl.process_weights_after_loading(layer)
    with pytest.raises(ValueError, match="duplicate/finalized"):
        load(layer, "w1", 2)


def test_missing_duplicate_bound_and_nonlocal():
    layer, impl = method(pending=1)
    assert not load(layer, "w1", 99)
    assert not impl._pending
    load(layer, "w1", 2)
    with pytest.raises(ValueError, match="duplicate"):
        load(layer, "w1", 2)
    with pytest.raises(RuntimeError, match="max_pending"):
        load(layer, "w3", 2)
    with pytest.raises(RuntimeError, match="missing MXFP4"):
        impl.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="not been finalized"):
        impl.apply(layer, None, None, None, None, None)


def test_tp_rejected_before_allocating():
    with pytest.raises(ValueError, match="TP1"):
        method(tp=2)


@pytest.mark.parametrize("invalid", ["uneven", "noncontiguous", "missing", "eplb"])
def test_ep_placement_rejected_before_allocating(invalid):
    layer = Layer(ep=8)
    if invalid == "uneven":
        layer.expert_map = layer.expert_map[:-1]
    elif invalid == "noncontiguous":
        layer.expert_map[3], layer.expert_map[5] = -1, 1
    elif invalid == "missing":
        layer.expert_map = None
    else:
        layer.enable_eplb = True
    impl = IluMxfp4ToW4A8TNMoEMethod(layer)
    with set_current_vllm_config(VllmConfig()), pytest.raises(ValueError):
        impl.create_weights(layer, 2, 256, 256, torch.bfloat16)
    assert not list(layer.parameters())


def test_validation_checked_before_compute():
    layer, impl = method()
    impl._finalized = True
    impl._runtime_context = SimpleNamespace(
        assert_validation_complete=Mock(side_effect=RuntimeError("counts mismatch"))
    )
    with pytest.raises(RuntimeError, match="counts mismatch"):
        impl.apply(layer, None, None, None, None, None)


def runtime_payload():
    return {
        "source": "inline",
        "schema_version": 1,
        "enabled": True,
        "applies_to": {
            "model_types": ["deepseek_v4"],
            "quant_methods": ["deepseek_v4_fp8"],
        },
        "ignore": [],
        "config_groups": {
            "tn": {
                "priority": 100,
                "targets": [r"re:.*\.ffn\.experts$"],
                "require": {
                    "original_disposition": "quantized",
                    "checkpoint_dtypes": ["uint8"],
                },
                "weights": {
                    "source": {
                        "format": "mxfp4",
                        "dtypes": ["uint8"],
                        "block_size": 32,
                        "scale_format": "ue8m0",
                    },
                    "target": {
                        "format": "int4-packed-tn",
                        "dtype": "int8",
                        "strategy": "channel",
                        "pack_version": 2,
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
                    "method": "IluMxfp4ToW4A8TNMoEMethod",
                    "max_pending_shards": 8,
                },
            }
        },
    }


def select(payload):
    # Exercise the actual config override and registry with a real RoutedExperts
    # type; construction itself is covered by the device model-load smoke test.
    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(has_bias=False, max_capture_size=1)
    cfg = VllmConfig()
    hf = SimpleNamespace(model_type="deepseek_v4", expert_dtype="fp4")
    cfg.model_config = SimpleNamespace(hf_config=hf, hf_text_config=hf)
    cfg.additional_config = {"ilu_quant": payload} if payload is not None else {}
    quant = IluDeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    with set_current_vllm_config(cfg):
        return quant.get_quant_method(layer, "model.layers.2.ffn.experts")


def test_config_dispatch_and_disabled_preserves_mxfp4():
    assert isinstance(select(runtime_payload()), IluMxfp4ToW4A8TNMoEMethod)
    assert isinstance(select(None), IluMxfp4MoEMethod)
    payload = runtime_payload()
    payload["enabled"] = False
    assert isinstance(select(payload), IluMxfp4MoEMethod)


@pytest.mark.parametrize("invalid", ["symmetric", "typo", "layout", "stage", "pending"])
def test_invalid_runtime_contract(invalid):
    p = runtime_payload()
    g = p["config_groups"]["tn"]
    if invalid == "symmetric":
        g["input_activations"]["symmetric"] = False
    elif invalid == "typo":
        g["weights"]["source"]["typo"] = True
    elif invalid == "layout":
        g["weights"]["target"]["format"] = "int4-packed-nn"
    elif invalid == "stage":
        g["weights"]["loading"]["stage"] = "process_weights_after_loading"
    else:
        g["iluvatar"]["max_pending_shards"] = True
    with pytest.raises((ValueError, TypeError)):
        select(p)
