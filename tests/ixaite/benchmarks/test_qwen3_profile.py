# SPDX-License-Identifier: Apache-2.0
"""Check the Qwen3 workload geometry used by the ixaite benchmark."""

from pathlib import Path

import yaml


def test_qwen3_8b_qkv_geometry_is_single_6144x4096_gemm():
    # GQA 32/8, D=128, H=4096 -> packed N = 32*128 + 8*128 + 8*128 = 6144.
    head_dim = 128
    n_q = 32 * head_dim
    n_kv = 8 * head_dim
    assert n_q + n_kv + n_kv == 6144
    profile = Path(__file__).resolve().parent / "models" / "qwen3_8b.yaml"
    linear = yaml.safe_load(profile.read_text(encoding="utf-8"))["l2"]["linear"]
    assert set(linear) >= {
        "attn_qkv_proj",
        "attn_o_proj",
        "mlp_gate_up_proj",
        "mlp_down_proj",
        "lm_head",
    }
    assert "attn_q_proj" not in linear
    assert "attn_k_proj" not in linear
    assert "attn_v_proj" not in linear
    assert (linear["attn_qkv_proj"]["N"], linear["attn_qkv_proj"]["K"]) == (6144, 4096)
