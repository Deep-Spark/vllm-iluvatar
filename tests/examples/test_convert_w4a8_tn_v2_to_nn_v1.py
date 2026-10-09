# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline TN/v2 -> NN/v1 W4A8 checkpoint converter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe_w4a8_int8 as w4a8_moe,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "iluvatar-compressed-tensors"
    / "convert_w4a8_tn_v2_to_nn_v1.py"
)
_SPEC = importlib.util.spec_from_file_location("convert_w4a8_tn_v2_to_nn_v1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
conv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conv)


def test_classify_dsv4_expert_names():
    assert conv.classify_expert_tensor("layers.0.ffn.experts.0.w1.weight") == "weight"
    assert (
        conv.classify_expert_tensor("layers.0.ffn.experts.0.w1.weight_scale")
        == "weight_scale"
    )
    assert (
        conv.classify_expert_tensor("mtp.0.ffn.experts.10.w2.i8_weight_scale")
        == "i8_weight_scale"
    )
    assert conv.classify_expert_tensor("layers.0.attn.wq_b.weight") is None
    assert conv.classify_expert_tensor("layers.0.ffn.shared_experts.w1.weight") is None


def test_relayout_matches_runtime():
    torch.manual_seed(0)
    values = torch.randint(-8, 8, (2, 64, 64), dtype=torch.int8)
    packed_tn = torch.stack([w4a8_moe._pack_int4(values[e]) for e in range(2)])
    torch.testing.assert_close(
        conv.relayout_int4_tn_to_nn(packed_tn),
        w4a8_moe._relayout_int4_tn_to_nn(packed_tn),
    )
    torch.testing.assert_close(
        conv.relayout_int4_tn_to_nn(packed_tn[0]),
        w4a8_moe._relayout_int4_tn_to_nn(packed_tn[0].unsqueeze(0)).squeeze(0),
    )
    # Same-shape 2D batch used by the shard converter.
    batched = conv.relayout_int4_tn_to_nn(torch.stack([packed_tn[0], packed_tn[1]]))
    one_by_one = torch.stack(
        [
            conv.relayout_int4_tn_to_nn(packed_tn[0]),
            conv.relayout_int4_tn_to_nn(packed_tn[1]),
        ]
    )
    torch.testing.assert_close(batched, one_by_one)


def test_resolve_device_auto_and_explicit():
    cpu = conv.resolve_device("cpu")
    assert cpu.type == "cpu"
    auto = conv.resolve_device("auto")
    assert auto.type in {"cpu", "cuda"}
    if torch.cuda.is_available():
        assert auto.type == "cuda"
        assert conv.resolve_device("cuda").type == "cuda"


def test_fold_scale_uses_v2_alpha_or_i8_scale():
    scale = torch.tensor([[0.5, 1.0]])
    torch.testing.assert_close(conv.fold_v2_scale(scale), scale * 18.0)
    i8 = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(conv.fold_v2_scale(scale, i8), scale * i8)


def test_convert_shard_relayouts_weights_and_drops_i8():
    torch.manual_seed(1)
    raw = torch.randint(-8, 8, (64, 64), dtype=torch.int8)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": w4a8_moe._pack_int4(raw),
        "layers.0.ffn.experts.0.w1.weight_scale": torch.ones((1, 64)),
        "layers.0.ffn.experts.0.w1.i8_weight_scale": torch.full((1, 64), 18.0),
        "layers.0.attn.q_norm.weight": torch.ones(8),
    }
    converted, stats = conv.convert_shard_tensors(tensors, device="cpu")
    assert stats["weight"] == 1
    assert stats["scale"] == 1
    assert stats["dropped_i8"] == 1
    assert stats["kept"] == 1
    assert "layers.0.ffn.experts.0.w1.i8_weight_scale" not in converted
    assert converted["layers.0.ffn.experts.0.w1.weight"].shape == (64, 32)
    torch.testing.assert_close(
        converted["layers.0.ffn.experts.0.w1.weight_scale"],
        torch.full((1, 64), 18.0),
    )
    torch.testing.assert_close(
        converted["layers.0.attn.q_norm.weight"], tensors["layers.0.attn.q_norm.weight"]
    )


def test_rewrite_legacy_and_compressed_tensors_config():
    legacy = conv.rewrite_quant_config(
        {
            "quant_method": "ilu_deepseek_v4_w4a8",
            "version": 2,
            "routed_experts": {"gemm_format": "TN", "pack_version": 2},
        }
    )
    assert legacy["version"] == 1
    assert legacy["routed_experts"] == {"gemm_format": "NN", "pack_version": 1}

    compressed = conv.rewrite_quant_config(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "routed_experts": {
                    "iluvatar": {"gemm_format": "TN", "pack_version": 2},
                },
                "dense_linears": {"iluvatar": {"gemm_format": "TN"}},
            },
        }
    )
    assert compressed["config_groups"]["routed_experts"]["iluvatar"] == {
        "gemm_format": "NN",
        "pack_version": 1,
    }
    assert compressed["config_groups"]["dense_linears"]["iluvatar"] == {
        "gemm_format": "TN"
    }


def test_convert_writes_nn_v1_checkpoint(tmp_path: Path):
    torch.manual_seed(2)
    raw = torch.randint(-8, 8, (64, 64), dtype=torch.int8)
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    save_file(
        {
            "layers.0.ffn.experts.0.w1.weight": w4a8_moe._pack_int4(raw),
            "layers.0.ffn.experts.0.w1.weight_scale": torch.ones((1, 64)),
            "embed.weight": torch.ones(4, 4),
        },
        str(src / "model-00001-of-00001.safetensors"),
    )
    (src / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "ilu_deepseek_v4_w4a8",
                    "version": 2,
                    "routed_experts": {"gemm_format": "TN", "pack_version": 2},
                }
            }
        )
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00001.safetensors",
                    "layers.0.ffn.experts.0.w1.weight_scale": "model-00001-of-00001.safetensors",
                    "embed.weight": "model-00001-of-00001.safetensors",
                },
            }
        )
    )
    (src / "tokenizer.json").write_text("{}")

    result = conv.convert(src, dst, device="cpu")
    assert result["stats"]["weight"] == 1
    written = json.loads((dst / "config.json").read_text())
    assert written["quantization_config"]["routed_experts"]["gemm_format"] == "NN"
    assert written["quantization_config"]["routed_experts"]["pack_version"] == 1
    tensors = load_file(str(dst / "model-00001-of-00001.safetensors"))
    assert tensors["layers.0.ffn.experts.0.w1.weight"].shape == (64, 32)
    torch.testing.assert_close(
        tensors["layers.0.ffn.experts.0.w1.weight_scale"],
        torch.full((1, 64), 18.0),
    )
    assert (dst / "tokenizer.json").is_file()
