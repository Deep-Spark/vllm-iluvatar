# SPDX-License-Identifier: Apache-2.0
"""CUDA launchers for V1 worker and DFlash metadata preparation."""

from __future__ import annotations

from typing import Any

import torch

from vllm_iluvatar.v1.worker.gpu._kernel_launcher import KernelLauncher


def _prepare_dflash(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    torch.ops._C.triton_prepare_dflash_inputs(
        *args,
        kwargs.pop("SAMPLE_FROM_ANCHOR"),
        kwargs.pop("PAD_SLOT_ID"),
        kwargs.pop("CP_SIZE"),
        kwargs.pop("CP_INTERLEAVE"),
    )
    if kwargs:
        raise TypeError(f"unexpected prepare_dflash launch options: {kwargs}")


def _zero_kv_blocks(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    if kwargs:
        raise TypeError(f"unexpected zero_kv_blocks launch options: {kwargs}")
    torch.ops._C.triton_zero_kv_blocks(*args)


def _prepare_uniform_decode(grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    if kwargs:
        raise TypeError(f"unexpected uniform-decode launch options: {kwargs}")
    torch.ops._C.triton_prepare_uniform_decode(*args, int(grid[0]))


def _token_request_mapping(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("num_warps", None)
    if kwargs:
        raise TypeError(f"unexpected token-mapping launch options: {kwargs}")
    torch.ops._C.triton_token_request_mapping(*args)


def _gather_block_tables(grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    if kwargs:
        raise TypeError(f"unexpected block-table gather launch options: {kwargs}")
    torch.ops._C.triton_gather_block_tables(*args, int(grid[1]))


def _compute_slot_mappings(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("TRITON_BLOCK_SIZE")
    torch.ops._C.triton_compute_slot_mappings(
        *args,
        kwargs.pop("CP_SIZE"),
        kwargs.pop("CP_INTERLEAVE"),
        kwargs.pop("PAD_ID"),
    )
    if kwargs:
        raise TypeError(f"unexpected slot-mapping launch options: {kwargs}")


def _apply_write(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    multi_group = kwargs.pop("MULTI_GROUP")
    if kwargs:
        raise TypeError(f"unexpected staged-write launch options: {kwargs}")
    (
        output,
        output_stride,
        write_indices,
        write_starts,
        contents,
        cu_lens,
        group_ids,
    ) = args
    torch.ops._C.triton_apply_write(
        output,
        0 if multi_group else int(output_stride),
        write_indices,
        write_starts,
        contents,
        cu_lens,
        group_ids if multi_group else None,
        output_stride if multi_group else None,
        multi_group,
    )


def _prepare_prefill_inputs(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    kwargs.pop("LOOKAHEAD_BLOCK")
    if kwargs:
        raise TypeError(f"unexpected prefill-input launch options: {kwargs}")
    torch.ops._C.triton_prepare_prefill_inputs(*args)


def _prepare_pos_seq_lens(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    if kwargs:
        raise TypeError(f"unexpected position launch options: {kwargs}")
    torch.ops._C.triton_prepare_pos_seq_lens(*args)


def _combine_sampled_and_draft(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    num_new = kwargs.pop("NUM_NEW_SAMPLED_TOKENS")
    if kwargs:
        raise TypeError(f"unexpected token-combine launch options: {kwargs}")
    torch.ops._C.triton_combine_sampled_and_draft_tokens(*args, num_new)


def _get_num_sampled_and_rejected(
    _grid: Any, *args: Any, **kwargs: Any
) -> None:
    if kwargs:
        raise TypeError(f"unexpected sample-count launch options: {kwargs}")
    torch.ops._C.triton_get_num_sampled_and_rejected(*args)


def _post_update(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("num_warps", None)
    if kwargs:
        raise TypeError(f"unexpected post-update launch options: {kwargs}")
    (
        idx_mapping,
        num_computed_tokens,
        last_sampled,
        output_bin_counts,
        output_bin_stride,
        sampled_tokens,
        sampled_stride,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids_stride,
        total_len,
    ) = args
    torch.ops._C.triton_post_update(
        idx_mapping,
        num_computed_tokens,
        last_sampled,
        output_bin_counts,
        output_bin_stride,
        sampled_tokens,
        sampled_stride,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids_stride,
        total_len,
    )


def _expand_idx_mapping(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("BLOCK_SIZE")
    if kwargs:
        raise TypeError(f"unexpected mapping-expand launch options: {kwargs}")
    torch.ops._C.triton_expand_idx_mapping(*args)


prepare_dflash_inputs_kernel = KernelLauncher(_prepare_dflash)
zero_kv_blocks_kernel = KernelLauncher(_zero_kv_blocks)
prepare_uniform_decode_kernel = KernelLauncher(_prepare_uniform_decode)
token_request_mapping_kernel = KernelLauncher(_token_request_mapping)
gather_block_tables_kernel = KernelLauncher(_gather_block_tables)
compute_slot_mappings_kernel = KernelLauncher(_compute_slot_mappings)
apply_write_kernel = KernelLauncher(_apply_write)
prepare_prefill_inputs_kernel = KernelLauncher(_prepare_prefill_inputs)
prepare_pos_seq_lens_kernel = KernelLauncher(_prepare_pos_seq_lens)
combine_sampled_and_draft_tokens_kernel = KernelLauncher(
    _combine_sampled_and_draft
)
get_num_sampled_and_rejected_kernel = KernelLauncher(
    _get_num_sampled_and_rejected
)
post_update_kernel = KernelLauncher(_post_update)
expand_idx_mapping_kernel = KernelLauncher(_expand_idx_mapping)
