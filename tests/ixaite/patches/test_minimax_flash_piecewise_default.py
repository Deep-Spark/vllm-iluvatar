# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU-only tests for the MiniMax-M3.1-Flash PIECEWISE decode default."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.config.compilation import CUDAGraphMode

from vllm_iluvatar.backends.ixaite.platform import (
    apply_minimax_flash_piecewise_default,
    is_minimax_flash_hf_config,
)


def _hf_config(*, vl: bool, nvfp4: bool, sparse: bool = True):
    quant_cfg = {"moe_quant_algo": "NVFP4" if nvfp4 else "MXFP4"}
    inner = SimpleNamespace(
        sparse_attention_config={"sparse_num_index_heads": 4} if sparse else None,
    )
    if vl:
        return SimpleNamespace(
            model_type="minimax_m3_vl", text_config=inner, quantization_config=quant_cfg
        )
    return SimpleNamespace(
        model_type="minimax_m3", text_config=inner, quantization_config=quant_cfg
    )


def _vllm_config(hf_config, cudagraph_mode, *, enforce_eager=False):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        model_config=SimpleNamespace(hf_config=hf_config, enforce_eager=enforce_eager),
    )


def test_minimax_flash_detection():
    assert is_minimax_flash_hf_config(_hf_config(vl=False, nvfp4=True))
    assert is_minimax_flash_hf_config(_hf_config(vl=True, nvfp4=True))


def test_non_flash_checkpoints_are_not_detected():
    assert not is_minimax_flash_hf_config(_hf_config(vl=False, nvfp4=False))
    assert not is_minimax_flash_hf_config(_hf_config(vl=True, nvfp4=True, sparse=False))
    other = _hf_config(vl=True, nvfp4=True)
    other.model_type = "qwen3_5_moe"
    assert not is_minimax_flash_hf_config(other)


def test_default_mode_flips_to_piecewise():
    config = _vllm_config(
        _hf_config(vl=True, nvfp4=True), CUDAGraphMode.FULL_AND_PIECEWISE
    )
    apply_minimax_flash_piecewise_default(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_explicit_full_mode_is_kept():
    config = _vllm_config(_hf_config(vl=True, nvfp4=True), CUDAGraphMode.FULL)
    apply_minimax_flash_piecewise_default(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL


def test_enforce_eager_keeps_default():
    config = _vllm_config(
        _hf_config(vl=True, nvfp4=True),
        CUDAGraphMode.FULL_AND_PIECEWISE,
        enforce_eager=True,
    )
    apply_minimax_flash_piecewise_default(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_non_flash_model_keeps_default():
    config = _vllm_config(
        SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(sparse_attention_config=None),
            quantization_config={},
        ),
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    apply_minimax_flash_piecewise_default(config)
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
