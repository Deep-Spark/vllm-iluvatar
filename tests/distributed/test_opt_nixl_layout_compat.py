from __future__ import annotations

from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    byte_compatible_kv_layouts,
)


def _singleton_mla() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
    )


def _singleton_swa_mla() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
    )


def test_exact_layout_is_compatible_without_registered_specs() -> None:
    assert byte_compatible_kv_layouts("LBNHC", "LBNHC", [])


def test_singleton_head_mla_accepts_certified_layout_pairs() -> None:
    for left, right in (
        ("LBNHC", "BLHNC"),
        ("LBNHC", "LBHNC"),
        ("LBHNC", "BLHNC"),
    ):
        assert byte_compatible_kv_layouts(left, right, [_singleton_mla()])
        assert byte_compatible_kv_layouts(right, left, [_singleton_mla()])
        assert byte_compatible_kv_layouts(left, right, [_singleton_swa_mla()])
        assert byte_compatible_kv_layouts(left, right, [])


def test_mixed_layout_compatibility_remains_fail_closed() -> None:
    multi_head_mla = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=512,
        dtype=torch.bfloat16,
    )

    for left, right in (
        ("LBNHC", "BLHNC"),
        ("LBNHC", "LBHNC"),
        ("LBHNC", "BLHNC"),
    ):
        assert not byte_compatible_kv_layouts(left, right, [multi_head_mla])
        assert not byte_compatible_kv_layouts(
            left,
            right,
            [_singleton_mla(), MagicMock()],
        )

    assert not byte_compatible_kv_layouts(
        "LBNHC",
        "UNKNOWN",
        [_singleton_mla()],
    )
