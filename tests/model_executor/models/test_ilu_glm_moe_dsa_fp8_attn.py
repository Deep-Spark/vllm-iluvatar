# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for GLM DSA fp8 packed MLA wiring (scale / kv_spec / dispatch)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.glm_moe_dsa import attention as attn_mod
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
)


@pytest.fixture(autouse=True)
def _clear_fp8_scale_cache():
    attn_mod._load_glm_dsa_fp8_kv_scale_table.cache_clear()
    yield
    attn_mod._load_glm_dsa_fp8_kv_scale_table.cache_clear()


def test_backend_supports_bf16_and_fp8_packed_head_sizes():
    assert IluFlashMLASparseBackend.get_supported_head_sizes() == [576, 640]


def test_resolve_fp8_kv_scale_global_env_float(monkeypatch):
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE", "0.03125", raising=False
    )
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH", "", raising=False
    )

    scale, source = attn_mod._resolve_glm_dsa_fp8_kv_scale(
        "model.layers.3.self_attn.attn"
    )

    assert scale == pytest.approx(0.03125)
    assert source == "VLLM_ILU_GLM_DSA_FP8_KV_SCALE"


def test_resolve_fp8_kv_scale_json_table_prefers_prefix(tmp_path, monkeypatch):
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {
                "model.layers.1.self_attn.attn": {"kv": 0.02},
                "layers.2.self_attn.attn": 0.04,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE", "", raising=False
    )
    monkeypatch.setattr(
        attn_mod.envs,
        "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH",
        str(path),
        raising=False,
    )

    scale1, source1 = attn_mod._resolve_glm_dsa_fp8_kv_scale(
        "model.layers.1.self_attn.attn"
    )
    scale2, source2 = attn_mod._resolve_glm_dsa_fp8_kv_scale(
        "model.layers.2.self_attn.attn"
    )

    assert scale1 == pytest.approx(0.02)
    assert source1.endswith("model.layers.1.self_attn.attn")
    assert scale2 == pytest.approx(0.04)
    assert source2.endswith("layers.2.self_attn.attn")


def test_resolve_fp8_kv_scale_missing_json_key_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"other.layer": 0.1}), encoding="utf-8")
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE", "", raising=False
    )
    monkeypatch.setattr(
        attn_mod.envs,
        "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH",
        str(path),
        raising=False,
    )

    scale, source = attn_mod._resolve_glm_dsa_fp8_kv_scale(
        "model.layers.0.self_attn.attn"
    )

    assert scale is None
    assert source == "online_amax_missing_json_key"


def test_resolve_fp8_kv_scale_online_default(monkeypatch):
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE", "", raising=False
    )
    monkeypatch.setattr(
        attn_mod.envs, "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH", "", raising=False
    )

    scale, source = attn_mod._resolve_glm_dsa_fp8_kv_scale(
        "model.layers.0.self_attn.attn"
    )

    assert scale is None
    assert source == "online_amax"


def test_load_fp8_kv_scale_table_rejects_bad_shape(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        attn_mod._load_glm_dsa_fp8_kv_scale_table(str(path))


def _sparse_mla_stub(*, num_heads: int, use_fp8: bool, prefix: str):
    """Minimal stand-in that exercises get_kv_cache_spec / fp8 dispatch flags."""
    mla = object.__new__(attn_mod.IluGlmMoeDsaSparseMLA)
    mla.num_heads = num_heads
    mla.head_size = 576
    mla.use_fp8_attn = use_fp8
    mla.kv_cache_dtype = "fp8" if use_fp8 else "bfloat16"
    mla._fp8_q_heads = (
        attn_mod._GLM_DSA_FP8_Q_HEADS
        if num_heads > 32
        else max(16, (num_heads + 15) // 16 * 16)
    )
    mla.prefix = prefix
    return mla


def test_fp8_kv_cache_spec_uses_640_uint8_rows():
    mla = _sparse_mla_stub(
        num_heads=8, use_fp8=True, prefix="model.layers.0.self_attn.attn"
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(),
    )

    spec = mla.get_kv_cache_spec(vllm_config)

    assert spec.head_size == attn_mod._GLM_DSA_FP8_ROW_BYTES == 640
    assert spec.dtype == torch.uint8
    assert spec.cache_dtype_str == "fp8"
    assert spec.num_kv_heads == 1


def test_bf16_kv_cache_spec_keeps_576_latent(monkeypatch):
    mla = _sparse_mla_stub(
        num_heads=8, use_fp8=False, prefix="model.layers.0.self_attn.attn"
    )
    monkeypatch.setattr(
        attn_mod,
        "kv_cache_dtype_str_to_dtype",
        lambda *_args, **_kwargs: torch.bfloat16,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, cache_dtype="bfloat16"),
        model_config=SimpleNamespace(),
    )

    spec = mla.get_kv_cache_spec(vllm_config)

    assert spec.head_size == 576
    assert spec.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("num_heads", "expect_q_heads"),
    [
        (8, 16),  # TP8 local heads, 16-aligned for DSV4
        (32, 32),
        (64, 64),  # fmha 64-head tile
    ],
)
def test_fp8_q_heads_padding(num_heads, expect_q_heads):
    mla = _sparse_mla_stub(
        num_heads=num_heads,
        use_fp8=True,
        prefix="model.layers.0.self_attn.attn",
    )

    assert mla._fp8_q_heads == expect_q_heads


def test_maybe_init_fp8_kv_scale_freezes_from_amax(monkeypatch):
    mla = _sparse_mla_stub(
        num_heads=8, use_fp8=True, prefix="model.layers.0.self_attn.attn"
    )
    mla._fp8_kv_scale_frozen = False
    mla._k_scale = torch.full((1,), 1.0, dtype=torch.float32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    kv_c = torch.full((4, 512), 89.6, dtype=torch.float32)  # 89.6/448 = 0.2
    mla._maybe_init_fp8_kv_scale(kv_c)

    assert mla._fp8_kv_scale_frozen is True
    assert float(mla._k_scale.item()) == pytest.approx(0.2)


def test_maybe_init_fp8_kv_scale_skips_during_capture(monkeypatch):
    mla = _sparse_mla_stub(
        num_heads=8, use_fp8=True, prefix="model.layers.0.self_attn.attn"
    )
    mla._fp8_kv_scale_frozen = False
    mla._k_scale = torch.full((1,), 1.0, dtype=torch.float32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    mla._maybe_init_fp8_kv_scale(torch.ones(2, 512))

    assert mla._fp8_kv_scale_frozen is False
    assert float(mla._k_scale.item()) == pytest.approx(1.0)


def _resolve_rms_rope_fp8_kv_fusion(
    *,
    env_enabled: bool,
    use_fp8_attn: bool,
    scale_frozen: bool,
    op_available: bool,
    num_heads: int = 8,
) -> bool:
    """Mirror IluGlmMoeDsaSparseMLA init-time fusion arming rules."""
    return bool(
        env_enabled
        and use_fp8_attn
        and scale_frozen
        and op_available
        and num_heads in (8, 16, 32)
    )


@pytest.mark.parametrize(
    ("env_enabled", "use_fp8", "frozen", "op_ok", "num_heads", "expect"),
    [
        (True, True, True, True, 8, True),
        (True, True, True, True, 16, True),
        (True, True, True, True, 32, True),
        (True, True, True, True, 64, False),
        (True, True, False, True, 8, False),
        (True, False, True, True, 8, False),
        (False, True, True, True, 8, False),
        (True, True, True, False, 8, False),
    ],
)
def test_rms_rope_fp8_kv_fusion_arm_matrix(
    env_enabled, use_fp8, frozen, op_ok, num_heads, expect
):
    assert (
        _resolve_rms_rope_fp8_kv_fusion(
            env_enabled=env_enabled,
            use_fp8_attn=use_fp8,
            scale_frozen=frozen,
            op_available=op_ok,
            num_heads=num_heads,
        )
        is expect
    )


def test_wrapper_skips_external_kv_rmsnorm_when_fusion_armed():
    """Mirror IluGlmMoeDsaMLAAttention forward branch: fusion skips external RMS."""
    called = {"norm": False, "mla": None}

    class _Norm:
        weight = torch.ones(512)
        variance_epsilon = 1e-6

        def __call__(self, *_args, **_kwargs):
            called["norm"] = True
            raise AssertionError("external kv_a_layernorm must be skipped")

    class _MlaAttn:
        use_rms_rope_fp8_kv_fusion = True

        def __call__(self, q, kv, k_pe, positions, kv_rms_weight=None, kv_rms_eps=None):
            called["mla"] = (
                kv_rms_weight is kv_a_layernorm.weight,
                kv_rms_eps == kv_a_layernorm.variance_epsilon,
                kv is kv_a,
            )
            return torch.zeros(1, 8)

    kv_a_layernorm = _Norm()
    mla_attn = _MlaAttn()
    kv_a = torch.randn(2, 512)
    q = torch.randn(2, 8, 576)
    k_pe = torch.randn(2, 64)
    positions = torch.arange(2)

    if mla_attn.use_rms_rope_fp8_kv_fusion:
        out = mla_attn(
            q,
            kv_a,
            k_pe,
            positions,
            kv_a_layernorm.weight,
            kv_a_layernorm.variance_epsilon,
        )
    else:
        kv_a_layernorm(kv_a)
        out = None
    assert out is not None
    assert called["norm"] is False
    assert called["mla"] == (True, True, True)
