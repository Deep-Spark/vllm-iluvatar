// SPDX-License-Identifier: Apache-2.0
// Declarations for plugin-internal torch.ops._C operators that have NO upstream
// vLLM counterpart. These ops serve the plugin's OOT overrides exclusively.
#pragma once

#include <torch/all.h>

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

}  // namespace vllm_iluvatar
