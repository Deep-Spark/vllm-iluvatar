// SPDX-License-Identifier: Apache-2.0
// Declarations for plugin-internal torch.ops._C operators that have NO upstream
// vLLM counterpart. These ops serve the plugin's OOT overrides exclusively.
#pragma once

#include <torch/all.h>

#include <optional>
#include <tuple>

namespace vllm_iluvatar {

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
    bool paged);

torch::Tensor engram_ngram_hash_meta(
    torch::Tensor& input_ids, torch::Tensor& positions, torch::Tensor& token_map,
    torch::Tensor& cache, torch::Tensor& block_table,
    torch::Tensor& token_to_req, torch::Tensor& query_start_loc,
    torch::Tensor& multipliers, torch::Tensor& primes, torch::Tensor& offsets,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    bool paged);

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

// k_norm -> RoPE -> paged store for indexer keys. ``store_fp8`` picks the
// record: bf16 [., ., 128], or uint8 [., ., 132] holding a page of 128 E4M3
// bytes per row followed by a page of one fp32 scale per row.
void v41_indexer_k_norm_rope_store(torch::Tensor& k_pre,
                                   torch::Tensor& positions,
                                   torch::Tensor& cos_sin_cache,
                                   torch::Tensor& rms_norm_weight,
                                   double rms_norm_eps, torch::Tensor& k_cache,
                                   torch::Tensor& kv_slot_mapping,
                                   int64_t compress_ratio, bool store_fp8,
                                   const std::optional<torch::Tensor>& row_out);

// Place dense ``[num_tokens, record_width]`` rows received from an earlier
// pipeline rank into this rank's paged cache, under this rank's slot mapping.
// The same "valid slot and group boundary" test the sender applied is reapplied
// here, so rows the sender skipped are never read.
void v41_move_rows_to_paged_cache(torch::Tensor& rows, torch::Tensor& kv_cache,
                                  torch::Tensor& slot_mapping,
                                  torch::Tensor& positions,
                                  int64_t compress_ratio);

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

}  // namespace vllm_iluvatar
