# SPDX-License-Identifier: Apache-2.0
"""Tests for the model-agnostic online-FP8 derived-config generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[4]
    / "examples"
    / "iluvatar-compressed-tensors"
    / "qwen38_flash_next_fp8_online.py"
)
SPEC = importlib.util.spec_from_file_location("fp8_online_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_auto_detects_nested_quant_config_and_ignore_key():
    config = {
        "text_config": {
            "vendor_quant": {
                "quant_method": "fp8",
                "modules_to_not_convert": ["model.a", "model.b", "lm_head"],
            }
        }
    }

    report = MODULE.patch_quant_config(
        config,
        quant_config_path=None,
        ignore_key=None,
        remove_patterns=[r"\.(?:a|b)$"],
        quant_updates={"serialized_dense": False},
    )

    quant = config["text_config"]["vendor_quant"]
    assert report["quant_config_path"] == "text_config.vendor_quant"
    assert report["removed_count"] == 2
    assert quant["modules_to_not_convert"] == ["lm_head"]
    assert quant["serialized_dense"] is False


def test_explicit_paths_support_renamed_model_and_ignore_keys():
    config = {
        "renamed_model_config": {
            "renamed_quant_config": {
                "quant_method": "fp8",
                "do_not_quantize": ["layers.0.proj", "layers.0.keep"],
            }
        }
    }

    report = MODULE.patch_quant_config(
        config,
        quant_config_path="renamed_model_config.renamed_quant_config",
        ignore_key="do_not_quantize",
        remove_patterns=[r"\.proj$"],
        quant_updates={"online_dense": True},
    )

    quant = config["renamed_model_config"]["renamed_quant_config"]
    assert report["ignore_key"] == "do_not_quantize"
    assert report["runtime_ignore_key"] == "ignored_layers"
    assert quant["do_not_quantize"] == ["layers.0.keep"]
    assert quant["ignored_layers"] == ["layers.0.keep"]
    assert quant["online_dense"] is True


def test_auto_detection_rejects_ambiguous_quant_configs():
    config = {
        "first": {"quant_method": "fp8", "ignored_layers": ["a"]},
        "second": {"quant_method": "fp8", "ignored_layers": ["b"]},
    }

    with pytest.raises(ValueError, match="exactly one"):
        MODULE.patch_quant_config(
            config,
            quant_config_path=None,
            ignore_key=None,
            remove_patterns=["a"],
            quant_updates={},
        )


def test_qwen_preset_selects_shape_dependent_layers_for_runtime_fallback():
    names = [
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
    ]
    config = {
        "quantization_config": {
            "quant_method": "fp8",
            "modules_to_not_convert": names,
        }
    }

    MODULE.patch_quant_config(
        config,
        quant_config_path=None,
        ignore_key=None,
        remove_patterns=MODULE.QWEN38_REMOVE_PATTERNS,
        quant_updates=MODULE.QWEN38_QUANT_UPDATES,
    )

    remaining = config["quantization_config"]["modules_to_not_convert"]
    assert remaining == []


def test_overlay_links_checkpoint_files_but_not_config(tmp_path):
    input_dir = tmp_path / "model"
    output_dir = tmp_path / "derived"
    input_dir.mkdir()
    (input_dir / "config.json").write_text("{}", encoding="utf-8")
    (input_dir / "model.safetensors").write_bytes(b"weights")
    (input_dir / "tokenizer").mkdir()
    (input_dir / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    output_dir.mkdir()

    MODULE._link_checkpoint_files(input_dir, output_dir, Path("config.json"))
    # Rerunning is idempotent when existing symlinks still target the source.
    MODULE._link_checkpoint_files(input_dir, output_dir, Path("config.json"))

    assert not (output_dir / "config.json").exists()
    assert (output_dir / "model.safetensors").is_symlink()
    assert (output_dir / "model.safetensors").resolve() == (
        input_dir / "model.safetensors"
    )
    assert (output_dir / "tokenizer" / "vocab.json").is_symlink()


def test_overlay_refuses_unsafe_output_and_real_file_overwrite(tmp_path):
    input_dir = tmp_path / "model"
    input_dir.mkdir()
    (input_dir / "config.json").write_text("{}", encoding="utf-8")
    (input_dir / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="must not be nested"):
        MODULE._ensure_output_is_safe(input_dir, input_dir / "derived")

    output_dir = tmp_path / "derived"
    output_dir.mkdir()
    (output_dir / "model.safetensors").write_bytes(b"do not replace")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        MODULE._link_checkpoint_files(input_dir, output_dir, Path("config.json"))
