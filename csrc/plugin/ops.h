// SPDX-License-Identifier: Apache-2.0
// Declarations for plugin-internal torch.ops._C operators that have NO upstream
// vLLM counterpart. These ops serve the plugin's OOT overrides exclusively.
#pragma once

#include <torch/all.h>

#include <optional>
#include <tuple>

namespace vllm_iluvatar {

// Native replacements for the Triton kernels exercised by the V2 GPU runner
// and DSpark. Python kernel-like launchers preserve the community call sites.
void triton_prepare_dflash_inputs(
    torch::Tensor& out_input_ids, torch::Tensor& out_query_positions,
    torch::Tensor& out_query_start_loc, torch::Tensor& out_seq_lens,
    torch::Tensor& out_query_slot_mapping,
    torch::Tensor& out_context_positions,
    torch::Tensor& out_context_slot_mapping,
    torch::Tensor& out_sample_indices, torch::Tensor& out_sample_pos,
    torch::Tensor& out_sample_idx_mapping, torch::Tensor& out_temperature,
    torch::Tensor& out_seeds, torch::Tensor& target_positions,
    torch::Tensor& target_query_start_loc, torch::Tensor& idx_mapping,
    torch::Tensor& last_sampled, torch::Tensor& next_prefill_tokens,
    torch::Tensor& num_sampled, torch::Tensor& num_rejected,
    torch::Tensor& temperature, torch::Tensor& seeds,
    torch::Tensor& block_table, int64_t block_table_stride,
    int64_t parallel_drafting_token_id, int64_t block_size,
    int64_t num_query_per_req, int64_t num_speculative_steps,
    int64_t max_num_reqs, int64_t max_num_tokens, int64_t max_model_len,
    int64_t cp_rank, bool sample_from_anchor, int64_t pad_slot_id,
    int64_t cp_size, int64_t cp_interleave);

void triton_zero_kv_blocks(torch::Tensor& seg_addrs,
                           torch::Tensor& seg_block_strides,
                           torch::Tensor& seg_page_sizes,
                           torch::Tensor& block_ids);

void triton_prepare_uniform_decode(
    torch::Tensor& seq_lens, torch::Tensor& decode_seq_lens,
    torch::Tensor& block_table, int64_t block_table_stride,
    torch::Tensor& expanded_block_table, int64_t expanded_stride,
    torch::Tensor& decode_lens, int64_t max_decode_len, int64_t num_tokens);

void triton_token_request_mapping(torch::Tensor& query_start_loc,
                                  torch::Tensor& output, int64_t num_reqs,
                                  int64_t num_mapped, int64_t num_tokens);

void triton_gather_block_tables(
    torch::Tensor& batch_to_req, torch::Tensor& src_ptrs,
    torch::Tensor& dst_ptrs, torch::Tensor& strides,
    torch::Tensor& num_blocks, int64_t num_blocks_stride, int64_t num_reqs,
    int64_t num_reqs_padded);

void triton_compute_slot_mappings(
    int64_t max_num_tokens, torch::Tensor& idx_mapping,
    torch::Tensor& query_start_loc, torch::Tensor& positions,
    torch::Tensor& block_table_ptrs, torch::Tensor& block_table_strides,
    torch::Tensor& block_sizes, torch::Tensor& kernel_block_sizes,
    torch::Tensor& mapping_enabled, torch::Tensor& slot_mappings,
    int64_t slot_mappings_stride, int64_t cp_rank, int64_t cp_size,
    int64_t cp_interleave, int64_t pad_id);

void triton_apply_write(
    torch::Tensor& output, int64_t output_stride,
    torch::Tensor& write_indices, torch::Tensor& write_starts,
    torch::Tensor& contents, torch::Tensor& cu_lens,
    const std::optional<torch::Tensor>& group_ids,
    const std::optional<torch::Tensor>& output_strides, bool multi_group);

void triton_prepare_prefill_inputs(
    torch::Tensor& input_ids, torch::Tensor& next_prefill_tokens,
    int64_t next_prefill_stride, int64_t num_lookahead,
    torch::Tensor& idx_mapping, torch::Tensor& query_start_loc,
    torch::Tensor& all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor& prefill_lens, torch::Tensor& num_computed_tokens);

void triton_prepare_pos_seq_lens(
    torch::Tensor& positions, torch::Tensor& seq_lens,
    torch::Tensor& idx_mapping, torch::Tensor& query_start_loc,
    torch::Tensor& num_computed_tokens, int64_t max_num_reqs);

void triton_combine_sampled_and_draft_tokens(
    torch::Tensor& input_ids, torch::Tensor& idx_mapping,
    torch::Tensor& last_sampled, torch::Tensor& query_start_loc,
    torch::Tensor& seq_lens, torch::Tensor& prefill_lens,
    torch::Tensor& draft_tokens, int64_t draft_stride,
    torch::Tensor& cu_num_logits, torch::Tensor& logits_indices,
    int64_t num_new_sampled_tokens);

void triton_get_num_sampled_and_rejected(
    torch::Tensor& num_sampled, torch::Tensor& num_rejected,
    torch::Tensor& seq_lens, torch::Tensor& cu_num_logits,
    torch::Tensor& idx_mapping, torch::Tensor& prefill_lens);

void triton_post_update(
    torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    torch::Tensor& last_sampled,
    const std::optional<torch::Tensor>& output_bin_counts,
    int64_t output_bin_stride, torch::Tensor& sampled_tokens,
    int64_t sampled_stride, torch::Tensor& num_sampled,
    torch::Tensor& num_rejected,
    const std::optional<torch::Tensor>& query_start_loc,
    torch::Tensor& all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor& total_len);

void triton_expand_idx_mapping(torch::Tensor& idx_mapping,
                               torch::Tensor& expanded_idx_mapping,
                               torch::Tensor& expanded_local_pos,
                               torch::Tensor& cu_num_logits);

void triton_gumbel_sample(
    torch::Tensor& local_argmax, int64_t local_argmax_stride,
    torch::Tensor& local_max, int64_t local_max_stride,
    const std::optional<torch::Tensor>& logits_cache,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const std::optional<torch::Tensor>& logits_cache_col,
    torch::Tensor& logits, int64_t logits_stride,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& seeds,
    torch::Tensor& positions, torch::Tensor& temperatures, int64_t vocab_size,
    int64_t block_size, bool is_drafting, bool apply_temperature,
    bool use_fp64, bool per_token_col);

void triton_compute_local_logits_stats(
    torch::Tensor& target_local_argmax, int64_t target_argmax_stride,
    torch::Tensor& target_local_max, int64_t target_max_stride,
    torch::Tensor& target_local_sumexp, int64_t target_sumexp_stride,
    torch::Tensor& draft_local_max, int64_t draft_max_stride,
    torch::Tensor& draft_local_sumexp, int64_t draft_sumexp_stride,
    torch::Tensor& target_logits, int64_t target_logits_stride,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& expanded_idx_mapping,
    torch::Tensor& expanded_local_pos, torch::Tensor& temperatures,
    int64_t vocab_size, int64_t num_speculative_steps,
    int64_t vocab_block_size, bool has_draft_logits);

void triton_rejection(
    torch::Tensor& sampled, int64_t sampled_stride,
    torch::Tensor& rejected_steps, torch::Tensor& target_rejected_lse,
    torch::Tensor& draft_rejected_lse, torch::Tensor& target_logits,
    int64_t target_logits_stride, torch::Tensor& target_local_argmax,
    int64_t target_argmax_stride, torch::Tensor& target_local_max,
    int64_t target_max_stride, torch::Tensor& target_local_sumexp,
    int64_t target_sumexp_stride, torch::Tensor& draft_sampled,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& draft_local_max,
    int64_t draft_max_stride, torch::Tensor& draft_local_sumexp,
    int64_t draft_sumexp_stride, torch::Tensor& cu_num_logits,
    torch::Tensor& idx_mapping, torch::Tensor& temperatures,
    torch::Tensor& seeds, torch::Tensor& positions,
    const std::optional<torch::Tensor>& synthetic_rates,
    int64_t vocab_num_blocks, bool has_draft_logits, bool synthetic_mode,
    bool use_block_verification);

void triton_resample(
    torch::Tensor& local_argmax, int64_t argmax_stride,
    torch::Tensor& local_max, int64_t max_stride,
    torch::Tensor& target_logits, int64_t target_logits_stride,
    torch::Tensor& target_rejected_lse,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& draft_rejected_lse,
    torch::Tensor& rejected_step, torch::Tensor& cu_num_logits,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& draft_sampled,
    torch::Tensor& temperatures, torch::Tensor& seeds,
    torch::Tensor& positions,
    const std::optional<torch::Tensor>& cumulative_log_p, int64_t vocab_size,
    int64_t block_size, bool has_draft_logits, bool use_fp64,
    bool use_block_verification);

void triton_insert_resampled(
    torch::Tensor& sampled, int64_t sampled_stride,
    torch::Tensor& num_sampled, torch::Tensor& local_argmax,
    int64_t argmax_stride, torch::Tensor& local_max, int64_t max_stride,
    int64_t resample_num_blocks, torch::Tensor& cu_num_logits,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& temperatures,
    bool use_fp64);

std::tuple<torch::Tensor, torch::Tensor> greedy_rejection_sample(
    torch::Tensor& target_logits, torch::Tensor& draft_sampled,
    torch::Tensor& cu_num_logits, int64_t max_speculative_steps);

// apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor
// Plugin-internal op (community torch.ops._C has NO such op): backs the
// ApplyRotaryEmb OOT override with a real CUDA kernel (flash-attn style:
// separate cos/sin, out-of-place). Self-contained (no ixformer dependency).
torch::Tensor apply_rotary_emb(torch::Tensor& x, torch::Tensor& cos,
                               torch::Tensor& sin, bool interleaved);

// Meta/fake impl for apply_rotary_emb: output has the same shape as ``x`` (the
// rotary is applied out-of-place). Enables shape propagation under torch.compile
// / FakeTensor and torch.library.opcheck.
torch::Tensor apply_rotary_emb_meta(torch::Tensor& x, torch::Tensor& cos,
                                    torch::Tensor& sin, bool interleaved);

// Fuses the Qwen3.5 per-head Gemma RMSNorm, NeoX rotary embedding, and
// extraction of the gate half from q_gate. q_gate and k may have padded token
// strides, while the three returned tensors are contiguous.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_rope_gate(torch::Tensor& q_gate, torch::Tensor& k,
                          torch::Tensor& q_weight, torch::Tensor& k_weight,
                          torch::Tensor& cos_sin_cache,
                          torch::Tensor& positions, double epsilon,
                          int64_t num_q_heads, int64_t num_kv_heads,
                          int64_t head_dim, int64_t rotary_dim);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_rope_gate_meta(
    torch::Tensor& q_gate, torch::Tensor& k, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
    torch::Tensor& positions, double epsilon, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim);

// Qwen3.5 gated path without RoPE: split q/gate and apply per-head Gemma
// RMSNorm to q/k. All three returned tensors are contiguous.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_gate(torch::Tensor& q_gate, torch::Tensor& k,
                     torch::Tensor& q_weight, torch::Tensor& k_weight,
                     double epsilon, int64_t num_q_heads,
                     int64_t num_kv_heads, int64_t head_dim);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_gate_meta(torch::Tensor& q_gate, torch::Tensor& k,
                          torch::Tensor& q_weight, torch::Tensor& k_weight,
                          double epsilon, int64_t num_q_heads,
                          int64_t num_kv_heads, int64_t head_dim);

// Write int8 DSA KV + per-token nope/rope scales into paged cache by
// slot_mapping. All cache tensors are mutated in-place.
void concat_and_cache_dsa_int8(torch::Tensor& kv_int8,
    torch::Tensor& nope_scale,
    torch::Tensor& rope_scale,
    torch::Tensor& kv_cache,
    torch::Tensor& nope_scale_cache,
    torch::Tensor& rope_scale_cache,
    torch::Tensor& slot_mapping);

// Write int8 K/V and per-token K scales into paged caches.
void reshape_and_cache_with_scale(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_scale,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& key_scale_cache, torch::Tensor& slot_mapping);

// Mixed cache writer: int8 K, fp16/bf16 V, and fp32 per-token K scales.
void reshape_and_cache_mix(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_scale,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& key_scale_cache, torch::Tensor& slot_mapping);

// ---------------------------------------------------------------------------
// MiniMax QK-Norm two-stage ops (TP>1 path). Portable CUDA, no corex30-only
// intrinsics. All-reduce stays in Python between the two stages.
//
// Shared constraints:
//   dtype   : qkv / q_weight / k_weight are fp16 or bf16 and share dtype;
//             qk_var is fp32.
//   shape   : qkv is 2D [T, >= q_size + 2*kv_size];
//             q_weight.numel() == q_size, k_weight.numel() == kv_size,
//             both 1D contiguous; qk_var is contiguous [T, 2].
//   stride  : qkv.stride(1) == 1 (outer stride arbitrary).
//   packing : PACKED=2 when q_size, kv_size, row_stride are even and pointers
//             are 4-byte aligned; otherwise scalar PACKED=1.
//   values  : q_size > 0, kv_size > 0, tp_world >= 1, epsilon > 0,
//             T <= INT_MAX.
// ---------------------------------------------------------------------------

// minimax_qk_var(Tensor qkv, int q_size, int kv_size) -> Tensor
// Returns contiguous fp32 [T, 2]: [:,0]=q mean-of-squares, [:,1]=k.
torch::Tensor minimax_qk_var(torch::Tensor& qkv, int64_t q_size,
                             int64_t kv_size);

torch::Tensor minimax_qk_var_meta(torch::Tensor& qkv, int64_t q_size,
                                  int64_t kv_size);

// minimax_qk_rms_apply(Tensor qkv, Tensor qk_var, Tensor q_weight,
//                      Tensor k_weight, int q_size, int kv_size,
//                      int tp_world, float epsilon) -> (Tensor, Tensor)
// Returns contiguous q_out [T, q_size], k_out [T, kv_size], dtype == qkv.
// qk_var is the all-reduced sum of per-shard means; /tp_world is folded into
// rsqrt.
std::tuple<torch::Tensor, torch::Tensor> minimax_qk_rms_apply(
    torch::Tensor& qkv, torch::Tensor& qk_var, torch::Tensor& q_weight,
    torch::Tensor& k_weight, int64_t q_size, int64_t kv_size,
    int64_t tp_world, double epsilon);

std::tuple<torch::Tensor, torch::Tensor> minimax_qk_rms_apply_meta(
    torch::Tensor& qkv, torch::Tensor& qk_var, torch::Tensor& q_weight,
    torch::Tensor& k_weight, int64_t q_size, int64_t kv_size,
    int64_t tp_world, double epsilon);

// ---------------------------------------------------------------------------
// DeepSeek-V4.1 Engram. Row-sharded table stays on the host; these ops replace
// the per-shift index_select lookback and the D2H/index_select/H2D gather.
// Gate and wkv stay on the existing ixformer kernels.
// ---------------------------------------------------------------------------

// Write this chunk's compressed token ids into the slot cache. Out-of-range
// slots land on the sink at index ``nslots`` (cache must hold nslots + 1).
void engram_write_hash_cache(torch::Tensor& input_ids, torch::Tensor& token_map,
                             torch::Tensor& slot_mapping, torch::Tensor& cache,
                             int64_t nslots);

// Paged lookback + n-gram hash. Output int64 [T, (max_ngram-1) * n_heads].
// ``paged`` selects the slot-cache path; otherwise ``cache`` is indexed by
// position. Semantics match EngramTokenHistory.lookback_paged + hash_ids:
// block id <= 0 is dead, and a dead source sticks for later shifts.
torch::Tensor engram_ngram_hash(
    torch::Tensor& input_ids, torch::Tensor& positions, torch::Tensor& token_map,
    torch::Tensor& cache, torch::Tensor& block_table,
    torch::Tensor& token_to_req, torch::Tensor& query_start_loc,
    torch::Tensor& multipliers, torch::Tensor& primes, torch::Tensor& offsets,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    bool paged, const std::optional<torch::Tensor>& lookback_token_ids,
    bool lookback_is_prompt_only);

torch::Tensor engram_ngram_hash_meta(
    torch::Tensor& input_ids, torch::Tensor& positions, torch::Tensor& token_map,
    torch::Tensor& cache, torch::Tensor& block_table,
    torch::Tensor& token_to_req, torch::Tensor& query_start_loc,
    torch::Tensor& multipliers, torch::Tensor& primes, torch::Tensor& offsets,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    bool paged, const std::optional<torch::Tensor>& lookback_token_ids,
    bool lookback_is_prompt_only);

// Gather fp8 e4m3 rows, multiply by ue8m0 block scales, write bf16.
// Ids outside [vocab_start, vocab_end) store zeros (row-shard all-reduce).
// ``weight`` / ``scale`` are uint8 CUDA views, including UVA aliases of host
// memory. Output is [T, n_hash_cols, dim].
torch::Tensor engram_lookup_fp8(torch::Tensor& indices, torch::Tensor& weight,
                                torch::Tensor& scale, int64_t vocab_start,
                                int64_t vocab_end);

torch::Tensor engram_lookup_fp8_meta(torch::Tensor& indices,
                                     torch::Tensor& weight, torch::Tensor& scale,
                                     int64_t vocab_start, int64_t vocab_end);

// cudaHostRegister a page-aligned span of a CPU tensor and return a CUDA
// alias. Does not copy. Used for the read-only engram mmap.
torch::Tensor engram_register_host(torch::Tensor& cpu_tensor);

// ---------------------------------------------------------------------------
// DeepSeek-V4.1 compressor / indexer ops.
//
// Shared contract:
//   slots   : compressed-domain global row indices, as
//             ``get_compressed_slot_mapping`` emits them (negative = skip). The
//             page and the row inside it come out of the slot alone, so none of
//             these take a block table.
//   ratio   : 1 or 2. A row is published only by the last token of its group,
//             ``(position + 1) % compress_ratio == 0``, and is rotated against
//             the group's first raw position.
//   cos_sin : fp32 [max_pos, >= rope_dim], cos half then sin half (GPT-J).
//   quant   : true quantization, never quantize-dequantize. The FP8 paths need
//             an ivcore30 build and raise otherwise.
// ---------------------------------------------------------------------------

// Ratio-1 compressor latent: per-token RMSNorm of a 512-wide fp32 row into a
// bf16 ``latent_out`` row. Rows whose slot is negative are left untouched.
void v41_compress_norm_cr1(torch::Tensor& kv_score,
                           torch::Tensor& slot_mapping,
                           torch::Tensor& rms_norm_weight, double rms_norm_eps,
                           torch::Tensor& latent_out);

// Ratio-2 compressor latent: pool each closed group of two into one row, then
// the same RMSNorm. ``state_cache`` is the per-request ring of the open group's
// [kv | score] rows, addressed by ``slot_mapping``'s
// ``block * rows_per_page + position % rows_per_page``. A group that a chunk's
// first token closes takes its predecessor from the ring; every other group is
// pooled from ``kv_score`` directly, so one ring block per request is enough.
// The chunk's last ``rows_per_page`` rows are saved for the next chunk.
void v41_compress_pool_norm(torch::Tensor& kv_score, torch::Tensor& positions,
                            torch::Tensor& state_cache,
                            torch::Tensor& slot_mapping,
                            torch::Tensor& query_start_loc,
                            torch::Tensor& token_to_req_indices,
                            torch::Tensor& rms_norm_weight,
                            double rms_norm_eps, torch::Tensor& latent_out);

// ``row_out``, on both store ops below, additionally mirrors the record in
// token order into a dense ``[num_tokens, record_width]`` buffer. That buffer
// is what crosses a pipeline boundary when a KV group is split across ranks,
// so the receiving rank replays bytes rather than recomputing them.

// GPT-J RoPE on a compressor latent, then publish it. ``kv_cache``'s dtype
// picks the record: uint8 [., ., 584] is the packed fp8_ds_mla row, bfloat16
// [., ., 512] the plain one. The reference's per-tensor-fp8 row has no reader
// on this platform and is not offered.
void v41_rope_insert(torch::Tensor& latent, torch::Tensor& positions,
                     torch::Tensor& cos_sin_cache, torch::Tensor& kv_cache,
                     torch::Tensor& slot_mapping, int64_t compress_ratio,
                     const std::optional<torch::Tensor>& row_out);

// GPT-J RoPE on the query and on the latent KV row, then publish that row into
// the plain bf16 sliding-window cache. ``slot_mapping`` is the framework's own
// per-token window slot, not a compressed-domain one, so this op is the
// exception to the ratio contract above: every token publishes. An empty head
// axis on ``q`` rotates nothing and only publishes, which is how the DSpark
// proposer writes context KV.
void v41_qkv_rope_insert(torch::Tensor& q, torch::Tensor& kv_cache,
                         torch::Tensor& kv, torch::Tensor& slot_mapping,
                         torch::Tensor& positions,
                         torch::Tensor& cos_sin_cache);

// k_norm -> RoPE -> paged store for indexer keys. ``record`` picks the layout:
// "bf16" is [., ., 128]; "fp8" and "int8" share a uint8 [., ., 132] page
// holding 128 value bytes per row followed by one fp32 scale per row, and
// differ only in the quantizer (E4M3 on a power-of-two scale vs amax/127).
// Only "fp8" needs FP8 hardware.
void v41_indexer_k_norm_rope_store(torch::Tensor& k_pre,
                                   torch::Tensor& positions,
                                   torch::Tensor& cos_sin_cache,
                                   torch::Tensor& rms_norm_weight,
                                   double rms_norm_eps, torch::Tensor& k_cache,
                                   torch::Tensor& kv_slot_mapping,
                                   int64_t compress_ratio,
                                   const std::string& record,
                                   const std::optional<torch::Tensor>& row_out);

// Place dense ``[num_tokens, record_width]`` rows received from an earlier
// pipeline rank into this rank's paged cache, under this rank's slot mapping.
// The same "valid slot and group boundary" test the sender applied is reapplied
// here, so rows the sender skipped are never read.
void v41_move_rows_to_paged_cache(torch::Tensor& rows, torch::Tensor& kv_cache,
                                  torch::Tensor& slot_mapping,
                                  torch::Tensor& positions,
                                  int64_t compress_ratio);

// ---------------------------------------------------------------------------
// DeepSeek-V4.1 two-level indexer. Columns are packed between ``row_ks`` and
// ``row_ke``, block ``b`` spans ``[row_ks + b * block_size, ...)``, and rows
// share bounds in groups of ``row_repeat``.
// ---------------------------------------------------------------------------

// Per-block max of the index logits, ``[rows, ceil(width / block_size)]`` fp32.
// The block holding a row's newest column scores ``+inf`` so the shortlist
// always keeps the most recent context.
torch::Tensor v41_candidate_block_scores(
    torch::Tensor& logits, const std::optional<torch::Tensor>& row_ks,
    torch::Tensor& row_ke, int64_t block_size, int64_t row_repeat);

// Fold ``torch.topk``'s (values, indices) into the shortlist ``out`` row: the
// first ``k`` columns take an index (``-1`` where the slot scored ``-inf``),
// the tail is ``-1``. Replaces the select / cast / copy / tail-fill the caller
// otherwise spells as separate torch ops.
void v41_write_candidate_topk(torch::Tensor& out, torch::Tensor& values,
                              torch::Tensor& indices, int64_t k);

// Set every logit outside the causal bounds or outside a shortlisted block to
// ``-inf``, in place. A candidate whose first column lands past the logits
// width keeps the last column instead.
void v41_apply_candidate_mask(torch::Tensor& logits,
                              const std::optional<torch::Tensor>& row_ks,
                              torch::Tensor& row_ke,
                              torch::Tensor& candidate_blocks,
                              int64_t block_size, int64_t row_repeat);

// Private block8 decode path. The scorer must have initialized the exterior
// to -inf; only GPU-visible columns are masked, through the ML store path.
void v41_apply_initialized_decode_mask(torch::Tensor& logits,
                                       torch::Tensor& row_ke,
                                       torch::Tensor& candidate_blocks);

// RoPE the index query in place and rescale the per-head weights.
// ``weights_out`` may alias ``weights_in``.
void v41_indexer_q_rope(torch::Tensor& q, torch::Tensor& positions,
                        torch::Tensor& cos_sin_cache,
                        torch::Tensor& weights_in, torch::Tensor& weights_out,
                        double weights_scale);

// RoPE the index query and quantize it to E4M3 against one power-of-two scale
// per (token, head). That scale is folded into the returned fp32 weights, since
// the FP8 scoring kernel takes no companion scale tensor.
std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_fp8(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale);

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_fp8_meta(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale);

// RoPE the index query and quantize it to INT8 against one ``amax / 127`` scale
// per (token, head), folded into the returned fp32 weights for the same reason
// as the FP8 op above. Needs no FP8 hardware.
std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_int8(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale);

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_int8_meta(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale);

// The identity selection, for when a layer's whole compressed context fits in
// ``topk`` and scoring cannot change the outcome: ``topk_idxs[t, j]`` is the
// group index ``j`` while the token can see it and ``-1`` past that, and
// ``topk_lens[t]`` is how many that is. A token sees
// ``(positions[t] + 1) / compress_ratio`` groups; a negative position is
// graph-capture padding and yields an empty row.
void v41_fill_short_context_topk(torch::Tensor& topk_idxs,
                                 torch::Tensor& topk_lens,
                                 torch::Tensor& positions,
                                 int64_t compress_ratio);

// Paged key slots of each token's sliding window, matching
// ``_compute_swa_indices_and_lens_kernel``. Output row ``pid`` is absolute
// token ``pid + token_offset``. ``swa_lens[pid]`` is the window span
// (``end_pos - start_pos``), including null pages. A padding token gets
// length 0 and its index row filled with -1.
//
// ``replay_start`` ([num_reqs]) is the position from which each request holds
// window KV. ``has_image`` loads ``left_visible`` / ``right_visible`` at the
// absolute token and widens the window; otherwise those tensors are unused.
void v41_swa_indices_and_lens(torch::Tensor& swa_indices,
                              torch::Tensor& swa_lens, int64_t window_size,
                              int64_t index_width, torch::Tensor& left_visible,
                              torch::Tensor& right_visible,
                              torch::Tensor& query_start_loc,
                              torch::Tensor& seq_lens,
                              torch::Tensor& token_to_req_indices,
                              torch::Tensor& is_valid_token,
                              torch::Tensor& block_table, int64_t block_size,
                              torch::Tensor& replay_start, int64_t num_tokens,
                              int64_t token_offset, bool has_image);

// Local compressed top-k slots resolved against the block table into paged
// global slots, the plugin's copy of ixformer's
// ``ds4_compute_global_topk_indices_and_lens``. Kept slots compact to the
// front of each row, the tail is -1, and ``topk_lens`` counts the kept ones.
// Page 0 is the null block and is dropped. An invalid token gets length 0
// and its ``token_to_req_indices`` entry is never read.
//
// ``positions`` (optional) caps the scan at ``(pos + 1) / compress_ratio``
// columns. ``slot_limit`` drops slots at or above it (0 disables).
// ``pad_walk_tail`` fills the tail with the last kept slot instead of -1.
std::tuple<torch::Tensor, torch::Tensor> v41_global_topk_indices_and_lens(
    torch::Tensor& topk_indices, torch::Tensor& token_to_req_indices,
    torch::Tensor& block_table, int64_t block_size,
    torch::Tensor& is_valid_token,
    const std::optional<torch::Tensor>& positions, int64_t compress_ratio,
    int64_t slot_limit, bool pad_walk_tail);

// Context rows a prefill request gathers for its window:
// ``query_len + min(max(prefix_len - replay_start, 0), window_size - 1)``,
// written to ``prefill_gather_lens[:num_prefills]``. The batch is decode-first,
// so the prefill requests start at ``num_decodes``.
void v41_swa_prefill_gather_lens(torch::Tensor& prefill_gather_lens,
                                 torch::Tensor& seq_lens,
                                 torch::Tensor& query_start_loc,
                                 torch::Tensor& replay_start,
                                 int64_t num_prefills, int64_t num_decodes,
                                 int64_t window_size);

// DSpark's noncausal draft window uses actual valid query count and includes
// all block keys. Writes every output/PAD row without transferring page tables.
void v41_dspark_noncausal_swa_indices_and_lens(
    torch::Tensor& indices, torch::Tensor& lens, torch::Tensor& seq_lens,
    torch::Tensor& query_start_loc, torch::Tensor& block_table,
    torch::Tensor& is_valid_token, int64_t block_size, int64_t window_size,
    int64_t index_width, int64_t num_tokens);

}  // namespace vllm_iluvatar
