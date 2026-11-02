# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 FP8 checkpoint -> the offline W4A8 NN runtime layout.

Covers the two Linear source contracts (block-FP8 and native BF16), the W8A16
``kv_b_proj`` conversion, and the group/target coverage of the shipped example
YAMLs. The quantizer and schema tests need no device; constructing a method
builds an :class:`IluCompressedTensorsW8A8Int8`, which reads the arch.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase

from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.fp8 import IluFp8Config
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_to_w8a8_linear import (
    IluFp8ToW8A8Int8LinearMethod,
    IluNativeToW8A8Int8LinearMethod,
    quantize_fp8_block_to_w8a8_int8,
    quantize_native_to_w8a8_int8,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_config import (
    parse_ilu_runtime_quant_config,
)

# The W8A8 scheme resolves ixformer kernels for the local arch at __init__.
_requires_device = pytest.mark.skipif(
    not (arch.is_ivcore30() or arch.is_ivcore11()),
    reason="constructing the W8A8 scheme requires an Iluvatar device",
)

EXAMPLES = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "iluvatar-runtime-quant"
    / "ZhipuAI"
    / "GLM-5.2"
)

# Every runtime module prefix the GLM-5.2 decoder builds that the offline
# GLM-5.2-w4a8-NN checkpoint quantizes, and the group that must claim it.
# fused_qkv_a_proj is the only runtime form of the checkpoint's q_a_proj and
# kv_a_proj_with_mqa, and indexer.wk/weights_proj stay BF16 in both paths.
PREFIX_GROUPS = {
    "model.layers.3.mlp.experts": "glm52_moe_fp8_to_w4a8_nn",
    "model.layers.3.self_attn.kv_b_proj": "glm52_kv_b_fp8_to_w8a16",
    "model.layers.3.self_attn.fused_qkv_a_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.3.self_attn.q_b_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.3.self_attn.o_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.3.self_attn.indexer.wq_b": "glm52_linear_fp8_to_w8a8",
    "model.layers.0.mlp.gate_up_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.0.mlp.down_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.3.mlp.shared_experts.gate_up_proj": "glm52_linear_fp8_to_w8a8",
    "model.layers.3.mlp.shared_experts.down_proj": "glm52_linear_fp8_to_w8a8",
}

UNCLAIMED_PREFIXES = (
    "model.layers.3.self_attn.indexer.wk_weights_proj",
    "model.layers.3.self_attn.indexer.weights_proj",
    "model.layers.3.mlp.gate",
    "lm_head",
)


def load_example(name):
    text = (EXAMPLES / name).read_text()
    return yaml.safe_load(text)["additional-config"]["ilu_quant"]


def parse_example(name):
    return parse_ilu_runtime_quant_config(
        load_example(name), model_type="glm_moe_dsa", quant_method="fp8"
    )


class Linear(LinearBase):
    """Minimal LinearBase stand-in; only the class and prefix are matched on."""

    def __init__(self, prefix):
        torch.nn.Module.__init__(self)
        self.prefix = prefix


class Experts(torch.nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix


def fp8_config():
    """``IluFp8Config`` as the official checkpoint's own config.json builds it.

    Going through ``from_config`` matters: its ``none_moe_*`` defaults differ
    from the constructor's, and they decide whether a dense Linear starts on
    the block-scaled ``IluFp8LinearMethod`` that these groups replace.
    """
    return IluFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            # The official checkpoint lists the BF16 MTP eh_proj here.
            "modules_to_not_convert": ["model.layers.78.eh_proj"],
        }
    )


def select_method(prefix, name="w4a8-nn-moe-inline.yaml", layer=None):
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"), dtype=torch.bfloat16
    )
    config.additional_config = {"ilu_quant": load_example(name)}
    quant = fp8_config()
    with set_current_vllm_config(config):
        return quant.get_quant_method(layer or Linear(prefix), prefix)


# --- weight conversion ----------------------------------------------------


@pytest.mark.parametrize("zero", [False, True])
def test_fp8_block_quantizer_reference(zero):
    torch.manual_seed(19)
    # 130x250 deliberately straddles the block grid: the last row block has 2
    # of 128 rows and the last column block 122 of 128.
    weight = (torch.zeros(130, 250) if zero else torch.randn(130, 250)).to(
        torch.float8_e4m3fn
    )
    blocks = torch.rand(2, 2, dtype=torch.float32) * 0.03
    actual_q, actual_s = quantize_fp8_block_to_w8a8_int8(weight, blocks, (128, 128))

    expanded = blocks.repeat_interleave(128, 0).repeat_interleave(128, 1)
    value = weight.float() * expanded[:130, :250]
    expected_s = value.abs().amax(dim=1, keepdim=True) / 127.0
    expected_s = torch.where(expected_s > 0, expected_s, torch.ones_like(expected_s))
    expected_q = torch.round(value / expected_s).clamp(-127, 127).to(torch.int8)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    assert actual_s.shape == (130, 1)


def test_fp8_block_quantizer_rejects_mismatched_scale():
    weight = torch.randn(256, 256).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="block scale shape"):
        quantize_fp8_block_to_w8a8_int8(
            weight, torch.rand(1, 2, dtype=torch.float32), (128, 128)
        )


def test_fp8_block_quantizer_rejects_non_fp8():
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        quantize_fp8_block_to_w8a8_int8(
            torch.randn(128, 128).bfloat16(),
            torch.rand(1, 1, dtype=torch.float32),
            (128, 128),
        )


@pytest.mark.parametrize("zero", [False, True])
def test_native_quantizer_reference(zero):
    torch.manual_seed(23)
    weight = (torch.zeros(96, 130) if zero else torch.randn(96, 130)).bfloat16()
    actual_q, actual_s = quantize_native_to_w8a8_int8(weight)

    value = weight.float()
    expected_s = value.abs().amax(dim=1, keepdim=True) / 127.0
    expected_s = torch.where(expected_s > 0, expected_s, torch.ones_like(expected_s))
    expected_q = torch.round(value / expected_s).clamp(-127, 127).to(torch.int8)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)


def test_native_quantizer_rejects_fp8_source():
    with pytest.raises(ValueError, match="BF16/FP16"):
        quantize_native_to_w8a8_int8(torch.randn(8, 8).to(torch.float8_e4m3fn))


def test_zero_channel_scale_is_finite():
    weight = torch.zeros(4, 128).bfloat16()
    weight[1, :] = 1.0
    q, s = quantize_native_to_w8a8_int8(weight)
    assert torch.isfinite(s).all() and (s > 0).all()
    assert q[0].eq(0).all() and q[2].eq(0).all()


# --- group selection ------------------------------------------------------


@pytest.mark.parametrize("prefix,group_id", sorted(PREFIX_GROUPS.items()))
def test_example_targets_claim_every_quantized_module(prefix, group_id):
    config = parse_example("w4a8-nn-moe-inline.yaml")
    layer = Experts(prefix) if prefix.endswith(".experts") else Linear(prefix)
    group = config.select_group(layer, prefix, fp8_config().packed_modules_mapping)
    assert group is not None, f"{prefix} matched no group"
    assert group.group_id == group_id


@pytest.mark.parametrize("prefix", UNCLAIMED_PREFIXES)
def test_example_leaves_bf16_modules_alone(prefix):
    config = parse_example("w4a8-nn-moe-inline.yaml")
    layer = Linear(prefix)
    assert (
        config.select_group(layer, prefix, fp8_config().packed_modules_mapping) is None
    )


@pytest.mark.parametrize(
    "name", ["w4a8-nn-moe-inline.yaml", "w4a8-nn-moe-mtp-inline.yaml"]
)
def test_every_example_target_is_reachable(name):
    """A target that claims no module aborts the first forward; catch it here."""
    config = parse_example(name)
    prefixes = dict(PREFIX_GROUPS)
    if "mtp" in name:
        prefixes["model.layers.78.eh_proj"] = "glm52_eh_proj_native_to_w8a8"
    mapping = fp8_config().packed_modules_mapping
    for prefix in prefixes:
        layer = Experts(prefix) if prefix.endswith(".experts") else Linear(prefix)
        group = config.select_group(layer, prefix, mapping)
        assert group is not None, f"{name}: {prefix} matched no group"
    for group in config.config_groups:
        unmatched = [
            target
            for target, hits in config._audits[group.group_id].target_hits.items()
            if hits == 0
        ]
        assert not unmatched, f"{name}: {group.group_id} has dead targets {unmatched}"


def test_mtp_example_only_adds_eh_proj():
    base = load_example("w4a8-nn-moe-inline.yaml")
    mtp = load_example("w4a8-nn-moe-mtp-inline.yaml")
    added = set(mtp["config_groups"]) - set(base["config_groups"])
    assert added == {"glm52_eh_proj_native_to_w8a8"}
    for group_id, group in base["config_groups"].items():
        assert mtp["config_groups"][group_id] == group


def test_eh_proj_group_is_absent_without_mtp():
    """MTP off must not declare the eh_proj target: nothing would match it."""
    base = load_example("w4a8-nn-moe-inline.yaml")
    targets = [t for g in base["config_groups"].values() for t in g["targets"]]
    assert not any("eh_proj" in t for t in targets)


# --- method construction --------------------------------------------------


@_requires_device
@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.3.self_attn.fused_qkv_a_proj",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.3.mlp.shared_experts.down_proj",
    ],
)
def test_block_fp8_linear_dispatch(prefix):
    method = select_method(prefix)
    assert isinstance(method, IluFp8ToW8A8Int8LinearMethod)
    assert method.block_shape == (128, 128)
    assert method.scheme.format == "TN"
    assert method.scheme.perf_mode == "PERF"
    assert not method.scheme.is_static_input_scheme
    assert method.scheme.input_symmetric


@_requires_device
def test_native_eh_proj_dispatch():
    prefix = "model.layers.78.eh_proj"
    method = select_method(prefix, "w4a8-nn-moe-mtp-inline.yaml")
    assert isinstance(method, IluNativeToW8A8Int8LinearMethod)


@_requires_device
def test_kv_b_dispatch_is_the_w8a16_method():
    from vllm_iluvatar.models.glm_moe_dsa.model import IluFp8ToGlmKVBProjW8A16Method

    method = select_method("model.layers.3.self_attn.kv_b_proj")
    assert isinstance(method, IluFp8ToGlmKVBProjW8A16Method)
    assert method.block_shape == (128, 128)


# --- fail-fast ------------------------------------------------------------


def mutate(name, group_id, mutation):
    payload = load_example(name)
    mutation(payload["config_groups"][group_id])
    return payload


@_requires_device
@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda g: g["weights"]["loading"].update(stage="weight_load"),
            "process_weights_after_loading",
        ),
        (
            lambda g: g["require"].update(original_disposition="ignored"),
            "requires original disposition 'ignored'",
        ),
        (lambda g: g["iluvatar"].update(gemm_format="NN"), "gemm_format=TN"),
        (lambda g: g["iluvatar"].update(unknown_option=1), "unsupported options"),
        (
            lambda g: g["weights"]["source"].update(block_shape=[64, 64]),
            "block_shape",
        ),
        (lambda g: g.update(input_activations=None), "per-token INT8"),
    ],
)
def test_block_fp8_linear_rejects_bad_contract(mutation, message):
    payload = mutate("w4a8-nn-moe-inline.yaml", "glm52_linear_fp8_to_w8a8", mutation)
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"), dtype=torch.bfloat16
    )
    config.additional_config = {"ilu_quant": payload}
    prefix = "model.layers.3.self_attn.q_b_proj"
    with (
        set_current_vllm_config(config),
        pytest.raises((ValueError, TypeError), match=message),
    ):
        fp8_config().get_quant_method(Linear(prefix), prefix)


@_requires_device
@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda g: g.update(
                input_activations={
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "dynamic": True,
                }
            ),
            "activation",
        ),
        (
            lambda g: g["weights"]["target"].update(format="int8-channel"),
            "weight target",
        ),
        (
            lambda g: g["weights"]["loading"].update(stage="weight_load"),
            "process_weights_after_loading",
        ),
    ],
)
def test_kv_b_rejects_bad_contract(mutation, message):
    payload = mutate("w4a8-nn-moe-inline.yaml", "glm52_kv_b_fp8_to_w8a16", mutation)
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"), dtype=torch.bfloat16
    )
    config.additional_config = {"ilu_quant": payload}
    prefix = "model.layers.3.self_attn.kv_b_proj"
    with (
        set_current_vllm_config(config),
        pytest.raises((ValueError, TypeError), match=message),
    ):
        fp8_config().get_quant_method(Linear(prefix), prefix)
