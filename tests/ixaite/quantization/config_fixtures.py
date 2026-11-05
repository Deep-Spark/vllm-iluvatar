# SPDX-License-Identifier: Apache-2.0
"""Shared quantization inputs, independent of baseline test-module state."""

from __future__ import annotations

_TARGET = "model.layers.0.linear"


# Keep private config tests independent of imports that register baseline classes.
def _make_config(
    *,
    weight_bits: int = 8,
    iluvatar: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": [_TARGET],
                "weights": {
                    "num_bits": weight_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "group_size": None,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "group_size": None,
                    "dynamic": True,
                },
                "output_activations": None,
                "iluvatar": iluvatar,
            }
        },
        "ignore": [],
    }


_DEEPSEEK_LEGACY = {
    "quant_method": "ilu_deepseek_v4_w4a8",
    "format": "ilu-mixed-w4a8",
    "version": 2,
    "routed_experts": {
        "targets": ["re:.*\\.experts\\.\\d+\\.w[123]$"],
        "num_bits": 4,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "pack_factor": 2,
        "gemm_format": "TN",
        "pack_version": 2,
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "shared_experts": {
        "targets": ["re:.*\\.shared_experts\\.w[123]$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "dense_linears": {
        "targets": [
            "re:.*\\.attn\\.(fused_wqa_wkv|wq_b|wo_b)$",
            "re:.*\\.attn\\.indexer\\.wq_b$",
            "re:.*\\.e_proj$",
            "re:.*\\.h_proj$",
        ],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "bmm_w8a16_layers": {
        "targets": ["re:.*\\.attn\\.wo_a$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": None,
        "gemm_format": "k16n16",
        "input_dtype": "bfloat16",
    },
    "bf16_layers": ["re:.*\\.attn\\.compressor\\..*"],
    "ignore": ["embed", "head", "re:.*\\.norm$"],
}
