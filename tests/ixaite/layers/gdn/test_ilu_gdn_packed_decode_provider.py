# SPDX-License-Identifier: Apache-2.0
"""GDN binding, immutable geometry and recurrent-state adapter contracts."""

from unittest.mock import Mock

import pytest

from vllm_iluvatar.backends.ixaite.layers import gdn as module

_SERVE_GDN_GEOMETRIES = (
    ("qwen35_9b_w8a8_tp1", 16, 32, 128, 128),
    ("qwen35_9b_tp2", 8, 16, 128, 128),
    ("qwen35_9b_tp4", 4, 8, 128, 128),
    ("qwen36_27b_w8a8_tp2", 8, 24, 128, 128),
    ("qwen36_27b_w8a8_tp4", 4, 12, 128, 128),
)


# Restore bindings and cached geometry decisions after every test.
@pytest.fixture
def gdn(monkeypatch):
    for name in (
        "_IXAITE_POST_CONV_PREP",
        "_IXAITE_POST_CONV_PREP_GEOMETRY",
        "_IXAITE_PACKED_DECODE",
        "_IXAITE_PACKED_DECODE_GEOMETRY",
        "_IXAITE_SPEC_DELTA_UPDATE",
        "_IXAITE_SPEC_DELTA_UPDATE_GEOMETRY",
        "_GDN_CHUNK",
        "_IXAITE_APC_CHUNK",
    ):
        monkeypatch.setattr(module, name, None)
    for fn in (
        module._fused_prep_supports_geometry,
        module._packed_decode_supports_geometry,
        module._spec_delta_update_supports_geometry,
    ):
        fn.cache_clear()
    yield module
    for fn in (
        module._fused_prep_supports_geometry,
        module._packed_decode_supports_geometry,
        module._spec_delta_update_supports_geometry,
    ):
        fn.cache_clear()


def test_geometry_gate_routes_by_local_head_count(gdn, monkeypatch) -> None:
    def is_geometry_supported(*, num_k_heads, head_k_dim, num_v_heads, head_v_dim):
        if num_k_heads % 4:
            return False, f"num_k_heads must be a multiple of 4, got {num_k_heads}"
        return True, None

    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        gdn,
        "load_ixaite_python_op",
        Mock(
            side_effect=[
                (Mock(), "supported on ivcore11"),
                (is_geometry_supported, "supported on ivcore11"),
            ]
        ),
    )

    gdn.configure_gdn_packed_decode_provider()

    # Qwen3.5-9B has 16 k heads: TP=4 shards them to 4, TP=8 to 2.
    assert gdn._packed_decode_supports_geometry(4, 128, 8, 128)
    assert not gdn._packed_decode_supports_geometry(2, 128, 4, 128)


def _packed_decode_geometry_ok(*, num_k_heads, head_k_dim, num_v_heads, head_v_dim):
    if num_k_heads < 1 or num_v_heads < 1:
        return False, f"head counts must be positive, got {num_k_heads}/{num_v_heads}"
    if num_v_heads % num_k_heads != 0:
        return False, "num_v_heads must be a multiple of num_k_heads (GQA)"
    if head_k_dim % 64 != 0:
        return False, f"head_k_dim must be a multiple of 64, got {head_k_dim}"
    if head_v_dim % 32 != 0:
        return False, f"head_v_dim must be a multiple of 32, got {head_v_dim}"
    return True, None


def _chunk_geometry_ok(*, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    if num_k_heads <= 0 or num_v_heads <= 0:
        return False, "head counts must be positive"
    if num_v_heads % num_k_heads != 0:
        return False, "num_v_heads must be divisible by num_k_heads"
    if head_k_dim not in (64, 128, 192, 256):
        return False, f"head_k_dim must be one of (64, 128, 192, 256), got {head_k_dim}"
    if head_v_dim <= 0:
        return False, f"head_v_dim must be positive, got {head_v_dim}"
    return True, None


@pytest.mark.parametrize(
    ("slug", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim"),
    _SERVE_GDN_GEOMETRIES,
)
def test_serve_geometries_pass_ixaite_packed_decode_gate(
    gdn, monkeypatch, slug, num_k_heads, num_v_heads, head_k_dim, head_v_dim
) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        gdn, "_IXAITE_PACKED_DECODE_GEOMETRY", _packed_decode_geometry_ok
    )
    gdn._packed_decode_supports_geometry.cache_clear()

    supported, reason = _packed_decode_geometry_ok(
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )
    assert supported, f"{slug}: {reason}"
    chunk_ok, chunk_reason = _chunk_geometry_ok(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    assert chunk_ok, f"{slug}: {chunk_reason}"
    assert gdn._packed_decode_supports_geometry(
        num_k_heads, head_k_dim, num_v_heads, head_v_dim
    )


def test_spec_dispatch_uses_bound_ixaite_for_supported_geometry(
    gdn, monkeypatch
) -> None:
    implementation = Mock()
    monkeypatch.setattr(gdn, "_IXAITE_SPEC_DELTA_UPDATE", implementation)
    monkeypatch.setattr(
        gdn,
        "_IXAITE_SPEC_DELTA_UPDATE_GEOMETRY",
        Mock(return_value=(True, None)),
    )

    selected = gdn._select_spec_delta_update(16, 128, 32, 128)

    assert selected is implementation
