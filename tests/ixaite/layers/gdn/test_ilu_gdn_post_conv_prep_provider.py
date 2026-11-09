# SPDX-License-Identifier: Apache-2.0
"""GDN binding, immutable geometry and recurrent-state adapter contracts."""

from unittest.mock import Mock

import pytest
import torch

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

    gdn.configure_gdn_post_conv_prep_provider()

    # Qwen3.5-9B has 16 k heads: TP=4 shards them to 4, TP=8 to 2.
    assert gdn._fused_prep_supports_geometry(4, 128, 8, 128)
    assert not gdn._fused_prep_supports_geometry(2, 128, 4, 128)


def _prep_geometry_ok(*, num_k_heads, head_k_dim, num_v_heads, head_v_dim):
    if head_k_dim % 64 != 0:
        return False, f"head_k_dim must be a multiple of 64, got {head_k_dim}"
    if (num_k_heads * head_k_dim) % 256 != 0:
        return False, "num_k_heads * head_k_dim must be a multiple of 256"
    if (num_v_heads * head_v_dim) % 256 != 0:
        return False, "num_v_heads * head_v_dim must be a multiple of 256"
    if not 0 < num_v_heads <= 256:
        return False, f"num_v_heads must be in (0, 256], got {num_v_heads}"
    return True, None


@pytest.mark.parametrize(
    ("slug", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim"),
    _SERVE_GDN_GEOMETRIES,
)
def test_serve_geometries_pass_ixaite_prep_gate(
    gdn, monkeypatch, slug, num_k_heads, num_v_heads, head_k_dim, head_v_dim
) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(gdn, "_IXAITE_POST_CONV_PREP_GEOMETRY", _prep_geometry_ok)
    gdn._fused_prep_supports_geometry.cache_clear()

    supported, reason = _prep_geometry_ok(
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
    )
    assert supported, f"{slug}: {reason}"
    assert gdn._fused_prep_supports_geometry(
        num_k_heads, head_k_dim, num_v_heads, head_v_dim
    )


def test_ixaite_gdn_chunk_adapts_cache_state_and_output_indices(gdn) -> None:
    seen = {}

    def implementation(q, k, v, g, beta, **kwargs):
        del k, g, beta
        seen.update(kwargs)
        result = v + 1
        if kwargs["out"] is not None:
            kwargs["out"].copy_(result)
            result = kwargs["out"]
        # Emulate the kernel's read-mask and in-place pool contract.
        pool = kwargs["initial_state"]
        for slot, read in zip(
            kwargs["initial_state_indices"], kwargs["has_initial_state"]
        ):
            if not read:
                pool[slot].zero_()
            pool[slot].add_(2)
        return result, pool

    adapter = gdn._adapt_ixaite_gdn_chunk(implementation)
    # Keep this mock-only wiring test independent of earlier tests that change
    # PyTorch's process-wide default device.
    device = torch.device("cpu")
    q = torch.zeros(3, 1, 4, dtype=torch.bfloat16, device=device)
    k = torch.zeros_like(q)
    v = torch.arange(6, dtype=torch.bfloat16, device=device).reshape(3, 1, 2)
    g = torch.zeros(3, 1, device=device)
    beta = torch.ones(3, 1, dtype=torch.bfloat16, device=device)
    out = torch.zeros_like(v)
    state = torch.arange(48, dtype=torch.float32, device=device).reshape(4, 1, 2, 6)
    state_before = state.clone()
    has_initial_state = torch.tensor([1, 0], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    # A distinct object makes the identity check verify the CPU hint argument.
    cu_seqlens_cpu = cu_seqlens.clone()
    state_indices = torch.tensor([2, 1], dtype=torch.int32, device=device)
    output_indices = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)

    adapter(
        q,
        k,
        v,
        out,
        g,
        beta,
        state,
        has_initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        state_indices,
        0.5,
        output_indices,
    )

    assert seen["scale"] == 0.5
    assert seen["chunk_size"] == 64
    assert seen["use_qk_l2norm_in_kernel"] is False
    assert seen["out"] is None
    # Passing the original fp32 pool prevents per-step gather/cast/scatter.
    assert seen["initial_state"] is state
    assert seen["initial_state_indices"] is state_indices
    assert seen["has_initial_state"] is has_initial_state
    assert seen["cu_seqlens_cpu"] is cu_seqlens_cpu
    torch.testing.assert_close(out[output_indices.long()], v + 1)
    torch.testing.assert_close(state[2], state_before[2] + 2)
    torch.testing.assert_close(state[1], torch.full_like(state[1], 2))
    torch.testing.assert_close(state[0], state_before[0])
    torch.testing.assert_close(state[3], state_before[3])
