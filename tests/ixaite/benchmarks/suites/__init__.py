# SPDX-License-Identifier: Apache-2.0
"""Import suites so registration side effects run."""

from __future__ import annotations

from tests.ixaite.benchmarks.suites import (
    activation_dynamic_scaled_int8 as activation_dynamic_scaled_int8,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import attention as attention  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    attn_output_sigmoid_gate as attn_output_sigmoid_gate,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import causal_conv1d as causal_conv1d  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    chunk_gated_delta_rule as chunk_gated_delta_rule,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import distributed as distributed  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    ds4_combine_topk_swa as ds4_combine_topk_swa,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_compressor as ds4_compressor,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_compute_global_topk as ds4_compute_global_topk,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_compute_swa_decode_indices as ds4_compute_swa_decode_indices,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_compute_swa_prefill_gather_lens as ds4_compute_swa_prefill_gather_lens,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_dense_compressed_attn_fwd as ds4_dense_compressed_attn_fwd,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_flash_mla_sparse_fwd as ds4_flash_mla_sparse_fwd,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_flash_mla_with_kvcache as ds4_flash_mla_with_kvcache,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_fused_indexer_q_rope as ds4_fused_indexer_q_rope,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_fused_qnorm_rope_kv_insert as ds4_fused_qnorm_rope_kv_insert,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_gather_k_cache as ds4_gather_k_cache,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import ds4_inv_rope as ds4_inv_rope  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    ds4_merge_attn_outputs as ds4_merge_attn_outputs,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import ds4_mhc as ds4_mhc  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    ds4_paged_sparse_attn_fwd as ds4_paged_sparse_attn_fwd,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    ds4_swa_attn_fwd as ds4_swa_attn_fwd,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    dsa_indexer_mqa_logits_bf16 as dsa_indexer_mqa_logits_bf16,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    dsa_update_topk_indices as dsa_update_topk_indices,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    embedding_gather as embedding_gather,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    fused_add_rms_norm as fused_add_rms_norm,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    fused_post_conv_prep as fused_post_conv_prep,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    fused_qk_rmsnorm as fused_qk_rmsnorm,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    gdn_packed_decode as gdn_packed_decode,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    gdn_spec_delta_update as gdn_spec_delta_update,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import gelu as gelu  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    gemm_bf16_bf16_fp32 as gemm_bf16_bf16_fp32,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    gemma_rms_norm as gemma_rms_norm,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import layer_norm as layer_norm  # noqa: F401
from tests.ixaite.benchmarks.suites import linear as linear  # noqa: F401
from tests.ixaite.benchmarks.suites import marlin_w8a16 as marlin_w8a16  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    moe_compute_token_index as moe_compute_token_index,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    moe_expand_input_dynamic_scaled_int8 as moe_expand_input_dynamic_scaled_int8,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    moe_fused_topk_bias as moe_fused_topk_bias,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    moe_output_reduce_sum as moe_output_reduce_sum,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    moe_w4a8_group_gemm as moe_w4a8_group_gemm,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    moe_w4a8_group_gemv as moe_w4a8_group_gemv,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    reshape_and_cache_flash as reshape_and_cache_flash,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    reshape_and_cache_int8 as reshape_and_cache_int8,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import rms_norm as rms_norm  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    rms_norm_gated as rms_norm_gated,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import rms_norm_qk as rms_norm_qk  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    scaled_int8_quant as scaled_int8_quant,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import (
    scaled_int8_quant_asym as scaled_int8_quant_asym,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import silu_and_mul as silu_and_mul  # noqa: F401
from tests.ixaite.benchmarks.suites import video_evs as video_evs  # noqa: F401
from tests.ixaite.benchmarks.suites import (
    vit_qkv_layout as vit_qkv_layout,  # noqa: F401
)
from tests.ixaite.benchmarks.suites import vit_qkv_rope as vit_qkv_rope  # noqa: F401
from tests.ixaite.benchmarks.suites import w8a8 as w8a8  # noqa: F401
