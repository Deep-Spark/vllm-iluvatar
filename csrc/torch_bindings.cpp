// SPDX-License-Identifier: Apache-2.0
// torch.ops._C bindings for the Iluvatar plugin.
//
// ixformer's native library used to occupy the ``_C`` torch namespace; it has
// since been renamed to ``_ixformer_C`` (opt_ds4_rebase), freeing ``_C`` for the
// plugin to register the vLLM-community-signature operators here. Importing the
// compiled module (``import vllm_iluvatar._C``) triggers these static
// TORCH_LIBRARY registrations, so community ``torch.ops._C.*`` call sites resolve
// to the Iluvatar implementations.
#include <torch/library.h>

#include "build_info.h"
#include "core/registration.h"
#include "community/ops.h"
#include "plugin/ops.h"

// Schemas are copied verbatim from vLLM upstream:
//   * rms_norm: csrc/libtorch_stable/torch_bindings.cpp
//   * rotary_embedding: csrc/libtorch_stable/torch_bindings.cpp
//   * weak_ref_tensor / get_cuda_view_from_cpu_tensor: csrc/torch_bindings.cpp
// TORCH_LIBRARY_FRAGMENT (rather than TORCH_LIBRARY) is used so the plugin only
// extends the ``_C`` namespace and never conflicts with a stub definition.
TORCH_LIBRARY_FRAGMENT_EXPAND(_C, ops) {
  ops.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rope_dim_offset=0, bool inverse=False) -> ()");
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");
  // Plugin-internal query op: arch-gated operators (e.g. the FP8 pair, which
  // only carries kernels on ivcore30) are bound on every arch, so callers and
  // tests need a way to ask which build this is. Device-independent, hence a
  // catch-all kernel registered together with the schema.
  ops.def("compiled_cuda_arch() -> str", &vllm_iluvatar::compiled_cuda_arch);
  ops.def(
      "reshape_and_cache_fp8("
      "Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, "
      "Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale"
      ") -> ()");
  // Plugin-internal op (NOT a community torch.ops._C op): backs the
  // ApplyRotaryEmb OOT override; consumed only by the plugin, never by
  // community source. Kept in the ``_C`` namespace alongside the other native
  // ops for a single load point (import vllm_iluvatar._C).
  ops.def(
      "apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) "
      "-> Tensor");
  ops.def(
      "fused_qk_rmsnorm_rope_gate("
      "Tensor q_gate, Tensor k, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, Tensor positions, float epsilon, "
      "int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim"
      ") -> (Tensor, Tensor, Tensor)");
  ops.def(
      "fused_qk_rmsnorm_gate("
      "Tensor q_gate, Tensor k, Tensor q_weight, Tensor k_weight, "
      "float epsilon, int num_q_heads, int num_kv_heads, int head_dim"
      ") -> (Tensor, Tensor, Tensor)");
  ops.def(
      "concat_and_cache_dsa_int8("
      "Tensor kv_int8, Tensor nope_scale, Tensor rope_scale, "
      "Tensor! kv_cache, Tensor! nope_scale_cache, Tensor! rope_scale_cache, "
      "Tensor slot_mapping) -> ()");
  ops.def(
      "reshape_and_cache_with_scale("
      "Tensor key, Tensor value, Tensor key_scale, Tensor! key_cache, "
      "Tensor! value_cache, Tensor! key_scale_cache, Tensor slot_mapping) "
      "-> ()");
  ops.def(
      "reshape_and_cache_mix("
      "Tensor key, Tensor value, Tensor key_scale, Tensor! key_cache, "
      "Tensor! value_cache, Tensor! key_scale_cache, Tensor slot_mapping) "
      "-> ()");
  // MiniMax QK-Norm two-stage ops (plugin-internal, TP>1 path).
  ops.def("minimax_qk_var(Tensor qkv, int q_size, int kv_size) -> Tensor");
  ops.def(
      "minimax_qk_rms_apply(Tensor qkv, Tensor qk_var, Tensor q_weight, "
      "Tensor k_weight, int q_size, int kv_size, int tp_world, "
      "float epsilon) -> (Tensor, Tensor)");
  // DeepSeek-V4.1 compressor / indexer ops (plugin-internal). Slots are
  // compressed-domain row indices, so none of these take a block table.
  ops.def(
      "v41_compress_norm_cr1("
      "Tensor kv_score, Tensor slot_mapping, Tensor rms_norm_weight, "
      "float rms_norm_eps, Tensor! latent_out) -> ()");
  ops.def(
      "v41_compress_pool_norm("
      "Tensor kv_score, Tensor positions, Tensor! state_cache, "
      "Tensor slot_mapping, Tensor query_start_loc, "
      "Tensor token_to_req_indices, Tensor rms_norm_weight, "
      "float rms_norm_eps, Tensor! latent_out) -> ()");
  ops.def(
      "v41_rope_insert("
      "Tensor latent, Tensor positions, Tensor cos_sin_cache, "
      "Tensor! kv_cache, Tensor slot_mapping, int compress_ratio, "
      "Tensor(a!)? row_out=None) -> ()");
  ops.def(
      "v41_qkv_rope_insert("
      "Tensor! q, Tensor! kv_cache, Tensor! kv, Tensor slot_mapping, "
      "Tensor positions, Tensor cos_sin_cache) -> ()");
  ops.def(
      "v41_indexer_k_norm_rope_store("
      "Tensor k_pre, Tensor positions, Tensor cos_sin_cache, "
      "Tensor rms_norm_weight, float rms_norm_eps, Tensor! k_cache, "
      "Tensor kv_slot_mapping, int compress_ratio, bool store_fp8, "
      "Tensor(a!)? row_out=None) -> ()");
  ops.def(
      "v41_move_rows_to_paged_cache("
      "Tensor rows, Tensor! kv_cache, Tensor slot_mapping, "
      "Tensor positions, int compress_ratio) -> ()");
  ops.def(
      "v41_candidate_block_scores("
      "Tensor logits, Tensor? row_ks, Tensor row_ke, int block_size, "
      "int row_repeat) -> Tensor");
  ops.def(
      "v41_apply_candidate_mask("
      "Tensor! logits, Tensor? row_ks, Tensor row_ke, "
      "Tensor candidate_blocks, int block_size, int row_repeat) -> ()");
  ops.def(
      "v41_indexer_q_rope("
      "Tensor! q, Tensor positions, Tensor cos_sin_cache, Tensor weights_in, "
      "Tensor! weights_out, float weights_scale) -> ()");
  ops.def(
      "v41_indexer_q_rope_fp8("
      "Tensor q, Tensor positions, Tensor cos_sin_cache, Tensor weights_in, "
      "float weights_scale) -> (Tensor, Tensor)");
  ops.def(
      "v41_fill_short_context_topk("
      "Tensor! topk_idxs, Tensor! topk_lens, Tensor positions, "
      "int compress_ratio) -> ()");
  ops.def(
      "v41_swa_indices_and_lens("
      "Tensor! swa_indices, Tensor! swa_lens, int window_size, int index_width, "
      "Tensor left_visible, Tensor right_visible, Tensor query_start_loc, "
      "Tensor seq_lens, Tensor token_to_req_indices, Tensor is_valid_token, "
      "Tensor block_table, int block_size, Tensor replay_start, "
      "int num_tokens, int token_offset, bool has_image) -> ()");
  ops.def(
      "v41_global_topk_indices_and_lens("
      "Tensor topk_indices, Tensor token_to_req_indices, Tensor block_table, "
      "int block_size, Tensor is_valid_token, Tensor? positions, "
      "int compress_ratio, int slot_limit, bool pad_walk_tail) "
      "-> (Tensor, Tensor)");
  ops.def(
      "v41_swa_prefill_gather_lens("
      "Tensor! prefill_gather_lens, Tensor seq_lens, Tensor query_start_loc, "
      "Tensor replay_start, int num_prefills, int num_decodes, "
      "int window_size) -> ()");
  ops.def(
      "triton_prepare_dflash_inputs("
      "Tensor! out_input_ids, Tensor! out_query_positions, "
      "Tensor! out_query_start_loc, Tensor! out_seq_lens, "
      "Tensor! out_query_slot_mapping, Tensor! out_context_positions, "
      "Tensor! out_context_slot_mapping, Tensor! out_sample_indices, "
      "Tensor! out_sample_pos, Tensor! out_sample_idx_mapping, "
      "Tensor! out_temperature, Tensor! out_seeds, Tensor target_positions, "
      "Tensor target_query_start_loc, Tensor idx_mapping, Tensor last_sampled, "
      "Tensor next_prefill_tokens, Tensor num_sampled, Tensor num_rejected, "
      "Tensor temperature, Tensor seeds, Tensor block_table, "
      "int block_table_stride, int parallel_drafting_token_id, int block_size, "
      "int num_query_per_req, int num_speculative_steps, int max_num_reqs, "
      "int max_num_tokens, int max_model_len, int cp_rank, "
      "bool sample_from_anchor, int pad_slot_id, int cp_size, "
      "int cp_interleave) -> ()");
  ops.def(
      "triton_zero_kv_blocks(Tensor seg_addrs, Tensor seg_block_strides, "
      "Tensor seg_page_sizes, Tensor block_ids) -> ()");
  ops.def(
      "triton_prepare_uniform_decode("
      "Tensor seq_lens, Tensor! decode_seq_lens, Tensor block_table, "
      "int block_table_stride, Tensor! expanded_block_table, "
      "int expanded_stride, Tensor! decode_lens, int max_decode_len, "
      "int num_tokens) -> ()");
  ops.def(
      "triton_token_request_mapping(Tensor query_start_loc, Tensor! output, "
      "int num_reqs, int num_mapped, int num_tokens) -> ()");
  ops.def(
      "triton_gather_block_tables("
      "Tensor batch_to_req, Tensor src_ptrs, Tensor dst_ptrs, Tensor strides, "
      "Tensor num_blocks, int num_blocks_stride, int num_reqs, "
      "int num_reqs_padded) -> ()");
  ops.def(
      "triton_compute_slot_mappings("
      "int max_num_tokens, Tensor idx_mapping, Tensor query_start_loc, "
      "Tensor positions, Tensor block_table_ptrs, Tensor block_table_strides, "
      "Tensor block_sizes, Tensor kernel_block_sizes, Tensor mapping_enabled, "
      "Tensor! slot_mappings, int slot_mappings_stride, int cp_rank, "
      "int cp_size, int cp_interleave, int pad_id) -> ()");
  ops.def(
      "triton_apply_write("
      "Tensor(a!) output, int output_stride, Tensor write_indices, "
      "Tensor write_starts, Tensor contents, Tensor cu_lens, "
      "Tensor? group_ids, Tensor? output_strides, bool multi_group) -> ()");
  ops.def(
      "triton_prepare_prefill_inputs("
      "Tensor! input_ids, Tensor! next_prefill_tokens, "
      "int next_prefill_stride, int num_lookahead, Tensor idx_mapping, "
      "Tensor query_start_loc, Tensor all_token_ids, int all_token_ids_stride, "
      "Tensor prefill_lens, Tensor num_computed_tokens) -> ()");
  ops.def(
      "triton_prepare_pos_seq_lens("
      "Tensor! positions, Tensor! seq_lens, Tensor idx_mapping, "
      "Tensor query_start_loc, Tensor num_computed_tokens, "
      "int max_num_reqs) -> ()");
  ops.def(
      "triton_combine_sampled_and_draft_tokens("
      "Tensor! input_ids, Tensor idx_mapping, Tensor last_sampled, "
      "Tensor query_start_loc, Tensor seq_lens, Tensor prefill_lens, "
      "Tensor draft_tokens, int draft_stride, Tensor cu_num_logits, "
      "Tensor! logits_indices, int num_new_sampled_tokens) -> ()");
  ops.def(
      "triton_get_num_sampled_and_rejected("
      "Tensor! num_sampled, Tensor! num_rejected, Tensor seq_lens, "
      "Tensor cu_num_logits, Tensor idx_mapping, Tensor prefill_lens) -> ()");
  ops.def(
      "triton_post_update("
      "Tensor idx_mapping, Tensor! num_computed_tokens, Tensor! last_sampled, "
      "Tensor? output_bin_counts, int output_bin_stride, "
      "Tensor sampled_tokens, int sampled_stride, Tensor num_sampled, "
      "Tensor num_rejected, Tensor? query_start_loc, Tensor! all_token_ids, "
      "int all_token_ids_stride, Tensor! total_len) -> ()");
  ops.def(
      "triton_expand_idx_mapping("
      "Tensor idx_mapping, Tensor! expanded_idx_mapping, "
      "Tensor! expanded_local_pos, Tensor cu_num_logits) -> ()");
  ops.def(
      "triton_gumbel_sample("
      "Tensor! local_argmax, int local_argmax_stride, Tensor! local_max, "
      "int local_max_stride, Tensor? logits_cache, "
      "int logits_cache_stride_0, int logits_cache_stride_1, "
      "Tensor? logits_cache_col, Tensor logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor seeds, Tensor positions, "
      "Tensor temperatures, int vocab_size, int block_size, "
      "bool is_drafting, bool apply_temperature, bool use_fp64, "
      "bool per_token_col) -> ()");
  ops.def(
      "triton_compute_local_logits_stats("
      "Tensor! target_local_argmax, int target_argmax_stride, "
      "Tensor! target_local_max, int target_max_stride, "
      "Tensor! target_local_sumexp, int target_sumexp_stride, "
      "Tensor! draft_local_max, int draft_max_stride, "
      "Tensor! draft_local_sumexp, int draft_sumexp_stride, "
      "Tensor target_logits, int target_logits_stride, Tensor? draft_logits, "
      "int draft_stride_0, int draft_stride_1, Tensor expanded_idx_mapping, "
      "Tensor expanded_local_pos, Tensor temperatures, int vocab_size, "
      "int num_speculative_steps, int vocab_block_size, "
      "bool has_draft_logits) -> ()");
  ops.def(
      "triton_rejection("
      "Tensor! sampled, int sampled_stride, Tensor! rejected_steps, "
      "Tensor! target_rejected_lse, Tensor! draft_rejected_lse, "
      "Tensor target_logits, int target_logits_stride, "
      "Tensor target_local_argmax, int target_argmax_stride, "
      "Tensor target_local_max, int target_max_stride, "
      "Tensor target_local_sumexp, int target_sumexp_stride, "
      "Tensor draft_sampled, Tensor? draft_logits, int draft_stride_0, "
      "int draft_stride_1, Tensor draft_local_max, int draft_max_stride, "
      "Tensor draft_local_sumexp, int draft_sumexp_stride, "
      "Tensor cu_num_logits, Tensor idx_mapping, Tensor temperatures, "
      "Tensor seeds, Tensor positions, Tensor? synthetic_rates, "
      "int vocab_num_blocks, bool has_draft_logits, bool synthetic_mode, "
      "bool use_block_verification) -> ()");
  ops.def(
      "triton_resample("
      "Tensor! local_argmax, int argmax_stride, Tensor! local_max, "
      "int max_stride, Tensor target_logits, int target_logits_stride, "
      "Tensor target_rejected_lse, Tensor? draft_logits, int draft_stride_0, "
      "int draft_stride_1, Tensor draft_rejected_lse, Tensor rejected_step, "
      "Tensor cu_num_logits, Tensor expanded_idx_mapping, "
      "Tensor draft_sampled, Tensor temperatures, Tensor seeds, "
      "Tensor positions, Tensor? cumulative_log_p, int vocab_size, "
      "int block_size, bool has_draft_logits, bool use_fp64, "
      "bool use_block_verification) -> ()");
  ops.def(
      "triton_insert_resampled("
      "Tensor! sampled, int sampled_stride, Tensor! num_sampled, "
      "Tensor local_argmax, int argmax_stride, Tensor local_max, "
      "int max_stride, int resample_num_blocks, Tensor cu_num_logits, "
      "Tensor expanded_idx_mapping, Tensor temperatures, "
      "bool use_fp64) -> ()");
  ops.def(
      "greedy_rejection_sample("
      "Tensor target_logits, Tensor draft_sampled, Tensor cu_num_logits, "
      "int max_speculative_steps) -> (Tensor, Tensor)");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  // DeepSeek-V4.1 Engram: slot-cache write, fused lookback+ngram hash,
  // fp8 row lookup, and a no-copy host registration for the mmap table.
  ops.def(
      "engram_write_hash_cache(Tensor input_ids, Tensor token_map, "
      "Tensor slot_mapping, Tensor! cache, int nslots) -> ()");
  ops.def(
      "engram_ngram_hash(Tensor input_ids, Tensor positions, Tensor token_map, "
      "Tensor cache, Tensor block_table, Tensor token_to_req, "
      "Tensor query_start_loc, Tensor multipliers, Tensor primes, "
      "Tensor offsets, int pad_id, int nslots, int block_size, int num_reqs, "
      "bool paged, Tensor? lookback_token_ids=None) -> Tensor");
  ops.def(
      "engram_lookup_fp8(Tensor indices, Tensor weight, Tensor scale, "
      "int vocab_start, int vocab_end) -> Tensor");
  ops.def("engram_register_host(Tensor cpu_tensor) -> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("rms_norm", &vllm_iluvatar::rms_norm);
  ops.impl("rotary_embedding", &vllm_iluvatar::rotary_embedding);
  ops.impl("static_scaled_fp8_quant",
           &vllm_iluvatar::static_scaled_fp8_quant);
  ops.impl("reshape_and_cache_fp8",
           &vllm_iluvatar::reshape_and_cache_fp8);
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb);
  ops.impl("fused_qk_rmsnorm_rope_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_rope_gate);
  ops.impl("fused_qk_rmsnorm_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_gate);
  ops.impl("concat_and_cache_dsa_int8",
           &vllm_iluvatar::concat_and_cache_dsa_int8);
  ops.impl("reshape_and_cache_with_scale",
           &vllm_iluvatar::reshape_and_cache_with_scale);
  ops.impl("reshape_and_cache_mix",
           &vllm_iluvatar::reshape_and_cache_mix);
  ops.impl("minimax_qk_var", &vllm_iluvatar::minimax_qk_var);
  ops.impl("minimax_qk_rms_apply", &vllm_iluvatar::minimax_qk_rms_apply);
  ops.impl("v41_compress_norm_cr1", &vllm_iluvatar::v41_compress_norm_cr1);
  ops.impl("v41_compress_pool_norm", &vllm_iluvatar::v41_compress_pool_norm);
  ops.impl("v41_rope_insert", &vllm_iluvatar::v41_rope_insert);
  ops.impl("v41_qkv_rope_insert", &vllm_iluvatar::v41_qkv_rope_insert);
  ops.impl("v41_indexer_k_norm_rope_store",
           &vllm_iluvatar::v41_indexer_k_norm_rope_store);
  ops.impl("v41_move_rows_to_paged_cache",
           &vllm_iluvatar::v41_move_rows_to_paged_cache);
  ops.impl("v41_candidate_block_scores",
           &vllm_iluvatar::v41_candidate_block_scores);
  ops.impl("v41_apply_candidate_mask",
           &vllm_iluvatar::v41_apply_candidate_mask);
  ops.impl("v41_indexer_q_rope", &vllm_iluvatar::v41_indexer_q_rope);
  ops.impl("v41_indexer_q_rope_fp8",
           &vllm_iluvatar::v41_indexer_q_rope_fp8);
  ops.impl("v41_fill_short_context_topk",
           &vllm_iluvatar::v41_fill_short_context_topk);
  ops.impl("v41_swa_indices_and_lens",
           &vllm_iluvatar::v41_swa_indices_and_lens);
  ops.impl("v41_swa_prefill_gather_lens",
           &vllm_iluvatar::v41_swa_prefill_gather_lens);
  ops.impl("v41_global_topk_indices_and_lens",
           &vllm_iluvatar::v41_global_topk_indices_and_lens);
  ops.impl("triton_prepare_dflash_inputs",
           &vllm_iluvatar::triton_prepare_dflash_inputs);
  ops.impl("triton_zero_kv_blocks",
           &vllm_iluvatar::triton_zero_kv_blocks);
  ops.impl("triton_prepare_uniform_decode",
           &vllm_iluvatar::triton_prepare_uniform_decode);
  ops.impl("triton_token_request_mapping",
           &vllm_iluvatar::triton_token_request_mapping);
  ops.impl("triton_gather_block_tables",
           &vllm_iluvatar::triton_gather_block_tables);
  ops.impl("triton_compute_slot_mappings",
           &vllm_iluvatar::triton_compute_slot_mappings);
  ops.impl("triton_apply_write", &vllm_iluvatar::triton_apply_write);
  ops.impl("triton_prepare_prefill_inputs",
           &vllm_iluvatar::triton_prepare_prefill_inputs);
  ops.impl("triton_prepare_pos_seq_lens",
           &vllm_iluvatar::triton_prepare_pos_seq_lens);
  ops.impl("triton_combine_sampled_and_draft_tokens",
           &vllm_iluvatar::triton_combine_sampled_and_draft_tokens);
  ops.impl("triton_get_num_sampled_and_rejected",
           &vllm_iluvatar::triton_get_num_sampled_and_rejected);
  ops.impl("triton_post_update", &vllm_iluvatar::triton_post_update);
  ops.impl("triton_expand_idx_mapping",
           &vllm_iluvatar::triton_expand_idx_mapping);
  ops.impl("triton_gumbel_sample", &vllm_iluvatar::triton_gumbel_sample);
  ops.impl("triton_compute_local_logits_stats",
           &vllm_iluvatar::triton_compute_local_logits_stats);
  ops.impl("triton_rejection", &vllm_iluvatar::triton_rejection);
  ops.impl("triton_resample", &vllm_iluvatar::triton_resample);
  ops.impl("triton_insert_resampled",
           &vllm_iluvatar::triton_insert_resampled);
  ops.impl("greedy_rejection_sample",
           &vllm_iluvatar::greedy_rejection_sample);
  // Aliasing a CUDA tensor's storage; registered on the CUDA key like upstream.
  ops.impl("weak_ref_tensor", &vllm_iluvatar::weak_ref_tensor);
  ops.impl("engram_write_hash_cache", &vllm_iluvatar::engram_write_hash_cache);
  ops.impl("engram_ngram_hash", &vllm_iluvatar::engram_ngram_hash);
  ops.impl("engram_lookup_fp8", &vllm_iluvatar::engram_lookup_fp8);
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CPU, ops) {
  // Input is a CPU tensor; registered on the CPU key like upstream.
  ops.impl("get_cuda_view_from_cpu_tensor",
           &vllm_iluvatar::get_cuda_view_from_cpu_tensor);
  ops.impl("engram_register_host", &vllm_iluvatar::engram_register_host);
}

TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops) {
  // Shape propagation for the out-of-place apply_rotary_emb (torch.compile /
  // FakeTensor / opcheck): output mirrors x's shape.
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb_meta);
  ops.impl("fused_qk_rmsnorm_rope_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_rope_gate_meta);
  ops.impl("fused_qk_rmsnorm_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_gate_meta);
  ops.impl("minimax_qk_var", &vllm_iluvatar::minimax_qk_var_meta);
  ops.impl("minimax_qk_rms_apply",
           &vllm_iluvatar::minimax_qk_rms_apply_meta);
  ops.impl("engram_ngram_hash", &vllm_iluvatar::engram_ngram_hash_meta);
  ops.impl("engram_lookup_fp8", &vllm_iluvatar::engram_lookup_fp8_meta);
  ops.impl("v41_indexer_q_rope_fp8",
           &vllm_iluvatar::v41_indexer_q_rope_fp8_meta);
}

REGISTER_EXTENSION(_C)
