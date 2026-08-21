# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the generic Attention output-shape wrapper."""

from __future__ import annotations

import pytest
import torch

from vllm_iluvatar.model_executor.layers.attention.attention import (
    make_forward,
    make_init,
    make_process_weights_after_loading,
)


def test_attention_init_restores_opaque_boundary_for_ixformer_torch_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("IXFORMER_USE_TORCH_OPS", "1")

    class Attention:
        use_direct_call = True

    def original_init(self):
        self.use_direct_call = True

    wrapped = make_init(original_init)
    attention = Attention()
    wrapped(attention)

    assert attention.use_direct_call is False


def test_attention_init_preserves_default_direct_call_without_ixformer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("IXFORMER_USE_TORCH_OPS", raising=False)
    monkeypatch.setattr(
        "vllm_iluvatar.model_executor.layers.attention.attention."
        "_ixformer_torch_ops_enabled",
        lambda: False,
    )

    class Attention:
        use_direct_call = False

    def original_init(self):
        self.use_direct_call = True

    wrapped = make_init(original_init)
    attention = Attention()
    wrapped(attention)

    assert attention.use_direct_call is True


def test_attention_wrapper_supplies_tuple_output_shape():
    captured: dict[str, object] = {}

    def original(self, query, key, value, output_shape=None, output_dtype=None):
        captured["shape"] = output_shape
        captured["dtype"] = output_dtype
        return query

    class Attention:
        num_heads = 2
        head_size_v = 3

    wrapped = make_forward(original)
    query = torch.zeros(4, 6)
    result = wrapped(Attention(), query, query, query)

    assert result is query
    assert captured == {"shape": (4, 6), "dtype": None}


def test_attention_wrapper_preserves_model_supplied_output_shape():
    captured: dict[str, object] = {}

    def original(self, query, key, value, output_shape=None, output_dtype=None):
        captured["shape"] = output_shape
        return query

    class Attention:
        num_heads = 2
        head_size_v = 3

    shape = (4, 5)
    wrapped = make_forward(original)
    wrapped(Attention(), torch.zeros(4, 6), None, None, output_shape=shape)

    assert captured["shape"] is shape


def test_attention_post_load_preserves_community_path_without_external_scales(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH", raising=False)
    calls: list[str] = []

    def original(self, act_dtype):
        calls.append("community")

    class Impl:
        supports_ilu_external_fp8_scales = True

        def process_weights_after_loading(self, act_dtype, *, layer):
            calls.append("ilu")
            return True

    class Attention:
        impl = Impl()

    make_process_weights_after_loading(original)(Attention(), torch.bfloat16)
    assert calls == ["community"]


def test_attention_post_load_binds_external_scales_before_community_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    scale_path = tmp_path / "fp8_scales.pt"
    scale_path.touch()
    monkeypatch.setenv(
        "VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH",
        str(scale_path),
    )
    calls: list[tuple[str, object]] = []

    def original(self, act_dtype):
        calls.append(("community", self))

    class Impl:
        supports_ilu_external_fp8_scales = True

        def process_weights_after_loading(self, act_dtype, *, layer):
            calls.append(("ilu", layer))
            return True

    class Attention:
        impl = Impl()

    attention = Attention()
    make_process_weights_after_loading(original)(attention, torch.bfloat16)
    assert calls == [("ilu", attention)]
