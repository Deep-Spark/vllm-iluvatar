# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ixformer align layout must describe the cache exactly as vLLM does.

``Qwen4ExpModelState`` hands the ixformer CUDA ops raw device addresses derived
from the state tensors, replacing the Triton kernels that read the equivalent
arrays out of ``MambaSpecDecodeGPUContext``. Nothing at runtime can catch a
mis-derived stride or a group/state ordering slip -- the copies would just move
the wrong bytes -- so this pins both derivations against each other.
"""

from __future__ import annotations

import pytest
import torch
from ixformer.inference.functions.qwen38 import build_mamba_align_layout

from conftest import requires_cuda
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
from vllm_iluvatar.models.qwen4_exp.model_state import collect_mamba_states

pytestmark = requires_cuda

CONV_DIM = 64
CONV_WIDTH = 3
V_HEADS, HEAD_V, HEAD_K = 4, 16, 16
NUM_BLOCKS = 8
MAX_REQS = 4
BLOCK_SIZE = 64


class _FakeAttention:
    def __init__(
        self,
        kv_cache: list[torch.Tensor],
        *,
        hosts_ple_window: bool = False,
    ) -> None:
        self.kv_cache = kv_cache
        self.ple_window_shape = (
            tuple(kv_cache[-1].shape[1:]) if len(kv_cache) == 3 else None
        )
        self.hosts_ple_window = hosts_ple_window


def _make_layer_states(device, dim_first: bool, page_padded: bool):
    if dim_first:
        conv = torch.zeros(
            (NUM_BLOCKS, CONV_DIM, CONV_WIDTH), dtype=torch.bfloat16, device=device
        )
    else:
        conv = torch.zeros(
            (NUM_BLOCKS, CONV_WIDTH, CONV_DIM), dtype=torch.bfloat16, device=device
        )
    if page_padded:
        # Real caches are as_strided over a padded page, which makes the block
        # stride larger than the data. Both derivations must size the copy from
        # the data, not the stride.
        inner = V_HEADS * HEAD_V * HEAD_K
        storage = torch.zeros(
            (NUM_BLOCKS, inner + 8), dtype=torch.float32, device=device
        )
        temporal = torch.as_strided(
            storage,
            (NUM_BLOCKS, V_HEADS, HEAD_V, HEAD_K),
            (inner + 8, HEAD_V * HEAD_K, HEAD_K, 1),
        )
    else:
        temporal = torch.zeros(
            (NUM_BLOCKS, V_HEADS, HEAD_V, HEAD_K), dtype=torch.float32, device=device
        )
    return [conv, temporal]


def _kv_cache_config(num_groups: int, layers_per_group: int) -> KVCacheConfig:
    """Two mamba groups plus a full-attention group, as the real model has."""
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((CONV_WIDTH, CONV_DIM), (V_HEADS, HEAD_V, HEAD_K)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_K,
        dtype=torch.bfloat16,
    )
    groups = [
        KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
    ]
    for group in range(num_groups):
        groups.append(
            KVCacheGroupSpec(
                layer_names=[f"gdn.{group}.{i}" for i in range(layers_per_group)],
                kv_cache_spec=mamba_spec,
            )
        )
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
    )


@pytest.mark.parametrize("page_padded", [False, True])
def test_layout_matches_vllm_metadata(page_padded):
    device = torch.device("cuda")
    dim_first = is_conv_state_dim_first()
    num_groups, layers_per_group = 2, 2
    kv_cache_config = _kv_cache_config(num_groups, layers_per_group)
    # Group 0 is full attention; the mamba groups follow.
    mamba_group_ids = [1 + i for i in range(num_groups)]
    copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)

    forward_context = {}
    for group_id in mamba_group_ids:
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
            forward_context[layer_name] = _FakeAttention(
                _make_layer_states(device, dim_first, page_padded)
            )
    block_tables = [
        torch.arange(MAX_REQS * 4, dtype=torch.int32, device=device).reshape(
            MAX_REQS, 4
        )
        % NUM_BLOCKS
        for _ in mamba_group_ids
    ]

    states, groups, is_conv = collect_mamba_states(
        kv_cache_config, mamba_group_ids, copy_funcs, forward_context
    )
    assert len(states) == num_groups * layers_per_group * len(copy_funcs)
    assert is_conv == [True, False] * (num_groups * layers_per_group)
    assert groups == [0, 0, 0, 0, 1, 1, 1, 1]

    layout = build_mamba_align_layout(states, groups, is_conv, block_tables, dim_first)

    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=MAX_REQS,
        kv_cache_config=kv_cache_config,
        num_state_types=len(copy_funcs),
        device=device,
        make_buffer=lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=device),
    )
    ctx.initialize_from_forward_context(
        kv_cache_config, forward_context, copy_funcs, block_tables
    )

    assert layout.block_table_stride_req == ctx.block_table_stride_req
    for field in (
        "block_table_ptrs",
        "state_base_addrs",
        "state_block_strides",
        "state_elem_sizes",
        "state_inner_sizes",
        "state_conv_widths",
        "state_group_indices",
        "state_dim_row_count",
        "state_dim_row_stride",
    ):
        mine = getattr(layout, field)
        theirs = getattr(ctx, field)
        torch.testing.assert_close(mine, theirs, atol=0, rtol=0, msg=field)


def test_layout_only_copies_the_used_ple_window() -> None:
    device = torch.device("cuda")
    kv_cache_config = _kv_cache_config(num_groups=1, layers_per_group=2)
    mamba_group_ids = [1]
    copy_funcs = (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        get_temporal_copy_spec,
    )
    forward_context = {}
    for index, layer_name in enumerate(kv_cache_config.kv_cache_groups[1].layer_names):
        states = _make_layer_states(device, is_conv_state_dim_first(), False)
        states.append(
            torch.zeros((NUM_BLOCKS, 32, 9), dtype=torch.bfloat16, device=device)
        )
        forward_context[layer_name] = _FakeAttention(
            states, hosts_ple_window=index == 0
        )

    states, groups, is_conv = collect_mamba_states(
        kv_cache_config, mamba_group_ids, copy_funcs, forward_context
    )

    assert len(states) == 5
    assert groups == [0] * 5
    assert is_conv == [True, False, False, True, False]
