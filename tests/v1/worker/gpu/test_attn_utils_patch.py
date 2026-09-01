# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the V2 attention/Mamba KV-cache layout patch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_iluvatar.ilu_patch.general_patches import patch_attn_utils
from vllm_iluvatar.v1.worker.gpu import attn_utils as ilu_attn_utils


class _Backend:
    def __init__(self, block_dim: int) -> None:
        self.block_dim = block_dim

    def get_kv_cache_block_dim(self, *args: Any, **kwargs: Any) -> int:
        return self.block_dim


def _attention_spec() -> AttentionSpec:
    return AttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float16,),
    )


def _group(
    kv_cache_spec: AttentionSpec | MambaSpec,
    layer_names: list[str],
    *,
    block_dim: int = 0,
    kv_cache_group_id: int = 0,
):
    return SimpleNamespace(
        kv_cache_spec=kv_cache_spec,
        layer_names=layer_names,
        kv_cache_group_id=kv_cache_group_id,
        backend=_Backend(block_dim),
    )


def test_align_wrapper_delegates_without_mamba():
    called = False

    def orig_align(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    align = ilu_attn_utils.make_align_mixed_attention_kv_cache_views(orig_align)
    kv_cache = torch.empty(2, 3, 5, 7)

    align(
        [_group(_attention_spec(), ["attn"], block_dim=1)],
        {"attn": kv_cache},
        [16],
        "auto",
        SimpleNamespace(),
    )

    assert called


def test_align_wrapper_restrides_kv_first_attention_when_mamba_present():
    def orig_align(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Mamba hybrid layout should not use community path")

    align = ilu_attn_utils.make_align_mixed_attention_kv_cache_views(orig_align)
    kv_cache = torch.empty(2, 3, 5, 7)

    align(
        [
            _group(_attention_spec(), ["attn"], block_dim=1, kv_cache_group_id=0),
            _group(_mamba_spec(), ["mamba"], kv_cache_group_id=1),
        ],
        {
            "attn": kv_cache,
            "mamba": torch.empty(3, 1, 1, 128, dtype=torch.uint8),
        },
        [16, 16],
        "auto",
        SimpleNamespace(),
    )

    assert kv_cache.stride() == (35, 70, 7, 1)


def test_align_wrapper_leaves_blocks_first_attention_when_mamba_present():
    def orig_align(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Mamba hybrid layout should not use community path")

    align = ilu_attn_utils.make_align_mixed_attention_kv_cache_views(orig_align)
    kv_cache = torch.empty(3, 2, 5, 7)
    original_stride = kv_cache.stride()

    align(
        [
            _group(_attention_spec(), ["attn"], block_dim=0, kv_cache_group_id=0),
            _group(_mamba_spec(), ["mamba"], kv_cache_group_id=1),
        ],
        {
            "attn": kv_cache,
            "mamba": torch.empty(3, 1, 1, 128, dtype=torch.uint8),
        },
        [16, 16],
        "auto",
        SimpleNamespace(),
    )

    assert kv_cache.stride() == original_stride


def test_patch_register_binds_align_wrapper():
    from vllm.v1.worker.gpu import attn_utils as community_attn_utils

    patch_attn_utils.register()

    assert hasattr(
        community_attn_utils._align_mixed_attention_kv_cache_views,
        "__wrapped__",
    )
