# SPDX-License-Identifier: Apache-2.0
"""Route V1 worker metadata preparation to CUDA extension kernels."""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="V1 worker metadata preparation uses CUDA extension kernels",
    )


def _apply() -> None:
    from vllm.v1.attention.ops import metadata
    from vllm.v1.worker import utils
    from vllm.v1.worker.gpu import block_table, buffer_utils, input_batch
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash

    from vllm_iluvatar.v1.attention.backends.mla import ilu_indexer
    from vllm_iluvatar.v1.worker.gpu import metadata_kernels

    replacements = (
        (
            metadata,
            "_token_request_mapping_kernel",
            metadata_kernels.token_request_mapping_kernel,
        ),
        (
            utils,
            "_zero_kv_blocks_kernel",
            metadata_kernels.zero_kv_blocks_kernel,
        ),
        (
            block_table,
            "_gather_block_tables_kernel",
            metadata_kernels.gather_block_tables_kernel,
        ),
        (
            block_table,
            "_compute_slot_mappings_kernel",
            metadata_kernels.compute_slot_mappings_kernel,
        ),
        (
            buffer_utils,
            "_apply_write_kernel",
            metadata_kernels.apply_write_kernel,
        ),
        (
            input_batch,
            "_prepare_prefill_inputs_kernel",
            metadata_kernels.prepare_prefill_inputs_kernel,
        ),
        (
            input_batch,
            "_prepare_pos_seq_lens_kernel",
            metadata_kernels.prepare_pos_seq_lens_kernel,
        ),
        (
            input_batch,
            "_combine_sampled_and_draft_tokens_kernel",
            metadata_kernels.combine_sampled_and_draft_tokens_kernel,
        ),
        (
            input_batch,
            "_get_num_sampled_and_rejected_kernel",
            metadata_kernels.get_num_sampled_and_rejected_kernel,
        ),
        (
            input_batch,
            "_post_update_kernel",
            metadata_kernels.post_update_kernel,
        ),
        (
            input_batch,
            "_expand_idx_mapping_kernel",
            metadata_kernels.expand_idx_mapping_kernel,
        ),
        (
            dflash,
            "_prepare_dflash_inputs_kernel",
            metadata_kernels.prepare_dflash_inputs_kernel,
        ),
        (
            ilu_indexer,
            "_prepare_uniform_decode_kernel",
            metadata_kernels.prepare_uniform_decode_kernel,
        ),
    )
    for module, name, replacement in replacements:
        p.rebind_module_func(module, name, replacement)
