// SPDX-License-Identifier: Apache-2.0
// Declarations for the vLLM-compatible torch.ops._C operators implemented by the
// Iluvatar plugin. Signatures mirror vLLM upstream csrc/ops.h so the community
// vllm/kernels/vllm_c.py and vllm/utils/torch_utils.py can call them unchanged.
#pragma once

#include <torch/all.h>

#include <optional>
#include <string>
#include <vector>

namespace vllm_iluvatar {

// rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()
// Backed by ixformer's RMSNorm kernels.
void rms_norm(torch::Tensor& result, torch::Tensor& input,
              torch::Tensor& weight, double epsilon);

// weak_ref_tensor(Tensor input) -> Tensor
// Returns a new tensor aliasing the same storage (no refcount on the source).
torch::Tensor weak_ref_tensor(torch::Tensor& tensor);

// get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor
// Maps a (pinned) CPU tensor into the CUDA address space and returns a view.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

// rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
//                  int head_size, Tensor cos_sin_cache, bool is_neox,
//                  int rope_dim_offset=0, bool inverse=False) -> ()
// Self-contained CUDA kernel (no ixformer dependency); rotates query/key in
// place. Semantics mirror vLLM upstream pos_encoding_kernels.cu.
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

// static_scaled_fp8_quant(
//     Tensor! result, Tensor input, Tensor scale, int[]? group_shape=None) -> ()
// Per-tensor static FP8 E4M3 quantization used by the FP8 attention query path.
// ivcore30 only; raises on every other arch.
void static_scaled_fp8_quant(
    torch::Tensor& result, torch::Tensor& input, torch::Tensor& scale,
    const std::optional<std::vector<int64_t>>& group_shape);

// reshape_and_cache_fp8(
//     Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache,
//     Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale)
//     -> ()
// Quantizes FP16/BF16 K/V and stores FP8 E4M3 values in an HND paged cache.
// ivcore30 only; raises on every other arch.
void reshape_and_cache_fp8(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale);

}  // namespace vllm_iluvatar
