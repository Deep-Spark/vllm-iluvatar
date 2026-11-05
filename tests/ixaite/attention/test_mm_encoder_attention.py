# SPDX-License-Identifier: Apache-2.0
"""Check the selected encoder ABI and private varlen binding."""

import pytest
import torch

from vllm_iluvatar.model_executor.layers.attention import mm_encoder_attention as ilu_mm


def _make_attn(
    *,
    num_heads: int = 2,
    head_size: int = 3,
    opt_level: int = 0,
    block_idx: int = 3,
    monkeypatch: pytest.MonkeyPatch,
    module=ilu_mm,
) -> ilu_mm.IluMMEncoderAttention:
    monkeypatch.setattr(
        module.envs,
        "VLLM_ILU_VIT_ATTN_OPT_LEVEL",
        opt_level,
        raising=False,
    )
    monkeypatch.setattr(
        module.envs,
        "VLLM_ILU_VIT_ATTN_LAYER",
        "",
        raising=False,
    )

    attn = object.__new__(module.IluMMEncoderAttention)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_heads
    attn.head_size = head_size
    attn.scale = head_size**-0.5
    attn.is_flash_attn_backend = True
    attn._vit_block_idx = block_idx
    return attn


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


# The serving BF16 path must preserve its native scale and declare the padding to attention.
@pytest.mark.parametrize("opt_level,layer_filter", [(0, ""), (1, "0"), (2, "0")])
def test_bf16_encoder_passes_logical_width(monkeypatch, opt_level, layer_filter):
    from vllm_iluvatar.backends.ixaite.attention import mm_encoder_attention as own

    captured = {}

    # Filling the supplied buffer verifies that the adapter keeps the existing out-parameter ABI.
    def attention(q, k, v, cuq, cuk, sq, sk, **kwargs):
        captured.update(kwargs)
        assert q.shape == k.shape == v.shape == (3, 2, 128)
        assert torch.count_nonzero(q[..., 72:]) == 0
        return kwargs["out"].zero_()

    monkeypatch.setattr(own, "_BF16_FLASH_ATTN_VARLEN", attention)
    attn = _make_attn(
        num_heads=2, head_size=72, opt_level=opt_level, monkeypatch=monkeypatch, module=own
    )
    # Layer 3 stays BF16 even when another layer is selected for quantized attention.
    monkeypatch.setattr(own.envs, "VLLM_ILU_VIT_ATTN_LAYER", layer_filter)
    assert attn._effective_vit_opt_level() == 0
    q = torch.randn((1, 3, 2, 72), dtype=torch.bfloat16)
    output = attn._forward_fa(q, q, q)
    assert output.shape == q.shape and captured["valid_head_dim"] == 72
    assert captured["softmax_scale"] == 72**-.5


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    for idx in range(cu_q.numel() - 1):
        q_start, q_end = int(cu_q[idx]), int(cu_q[idx + 1])
        kv_start, kv_end = int(cu_kv[idx]), int(cu_kv[idx + 1])
        out = _reference_attention(
            q[q_start:q_end].unsqueeze(0),
            k[kv_start:kv_end].unsqueeze(0),
            v[kv_start:kv_end].unsqueeze(0),
            scale,
        )
        outputs.append(out.squeeze(0))
    return torch.cat(outputs, dim=0)


def test_forward_fa_uses_bound_varlen_without_ixformer(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_iluvatar.backends.ixaite.attention import mm_encoder_attention as own

    calls = {"bound": 0}

    def bound_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max_seqlen_q,
        max_seqlen_k,
        *,
        softmax_scale,
        causal,
        out=None,
        valid_head_dim=None,
    ):
        del max_seqlen_q, max_seqlen_k
        assert causal is False
        assert valid_head_dim is None  # Legacy float32 test callers keep the old ABI.
        assert out is not None
        assert out.shape == q.shape
        calls["bound"] += 1
        result = _varlen_attention(q, k, v, cu_q, cu_kv, softmax_scale)
        out.copy_(result)
        return out

    monkeypatch.setattr(own, "_BF16_FLASH_ATTN_VARLEN", bound_flash_attn_varlen_func)
    attn = _make_attn(opt_level=0, monkeypatch=monkeypatch, module=own)
    query = torch.randn(1, 3, attn.num_heads, attn.head_size)
    key = torch.randn(1, 3, attn.num_kv_heads, attn.head_size)
    value = torch.randn(1, 3, attn.num_kv_heads, attn.head_size)

    output = attn._forward_fa(query, key, value)

    expected = _reference_attention(query, key, value, attn.scale)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert calls["bound"] == 1


# Both independent implementations retain the same encoder geometry.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_d72_pads_to_128_for_both_providers(monkeypatch, mode):
    from vllm_iluvatar.backends.ixaite.attention import mm_encoder_attention as own

    module = ilu_mm if mode == "baseline" else own
    attn = _make_attn(head_size=72, monkeypatch=monkeypatch, module=module)
    assert attn.padded_head_dim == 128


# Check real preparation and attention against independent BF16-prepared, FP32 attention math.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("head_size", [72, 80])
@pytest.mark.parametrize("num_heads", [16, 8])
@pytest.mark.parametrize("cache_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("opt_level", [0, 1])
def test_vit_encoder_matches_reference_on_gpu(
    monkeypatch, head_size, num_heads, cache_dtype, opt_level
):
    import ixaite

    from vllm_iluvatar.backends.ixaite.attention import mm_encoder_attention as own
    from vllm_iluvatar.backends.ixaite.attention import vit

    ixaite.initialize("ivcore11")
    monkeypatch.setattr(vit, "_VIT_QKV_LAYOUT", ixaite.vit_qkv_layout)
    monkeypatch.setattr(vit, "_VIT_QKV_ROPE", ixaite.vit_qkv_rope)
    monkeypatch.setattr(own, "_BF16_FLASH_ATTN_VARLEN", ixaite.flash_attn_varlen_func)
    attn = _make_attn(
        num_heads=num_heads, head_size=head_size, opt_level=opt_level,
        monkeypatch=monkeypatch, module=own,
    )
    monkeypatch.setattr(own.envs, "VLLM_ILU_VIT_ATTN_LAYER", "0")
    torch.manual_seed(3965)
    # Unequal segments and tail lengths catch cross-image attention and padded-channel errors.
    offsets = [0, 17, 48, 113]
    total = offsets[-1]
    packed = torch.randn(
        total, 1, 3 * num_heads * head_size, device="cuda", dtype=torch.bfloat16
    )
    reference = list(packed.view(total, 1, 3, num_heads, head_size)
                     .permute(1, 0, 2, 3, 4).unbind(2))
    cos = sin = None
    if cache_dtype is not None:
        angles = torch.randn(total, head_size // 2, device="cuda")
        cos, sin = angles.cos().to(cache_dtype), angles.sin().to(cache_dtype)
        c, s = cos[None, :, None, :].float(), sin[None, :, None, :].float()
        # Split the NeoX pairs, rotate in FP32, then round once to the BF16 serving input.
        for index in (0, 1):
            first, second = reference[index].float().chunk(2, dim=-1)
            reference[index] = torch.cat(
                (first * c - second * s, second * c + first * s), dim=-1
            ).to(torch.bfloat16)

    prepared = vit.vit_fused_qkv_preproc(
        packed, cos, sin, 1, total, num_heads, head_size, 128
    )
    for actual, expected in zip(prepared, reference):
        torch.testing.assert_close(actual[..., :head_size], expected, atol=0.03, rtol=0.03)
        assert torch.count_nonzero(actual[..., head_size:]) == 0

    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    actual = attn._forward_fa(*prepared, cu_seqlens=cu, max_seqlen=65)
    expected = _varlen_attention(
        *(tensor.squeeze(0).float() for tensor in reference), cu, cu, head_size**-0.5
    ).unsqueeze(0)
    assert actual.shape == (1, total, num_heads, head_size)
    torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.03)
