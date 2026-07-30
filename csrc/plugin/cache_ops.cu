// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.concat_and_cache_dsa_int8 -- plugin-internal op
//   Write int8 DSA KV + per-token nope/rope scales into paged cache by
//   slot_mapping. Self-contained (no ixformer dependency).

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "ops.h"

namespace vllm_iluvatar {
namespace {

constexpr int kWarpSize = 64;
constexpr int kWarpsPerBlock = 16;

__global__ void concat_and_cache_dsa_int8_kernel(
    const int8_t* __restrict__ kv_int8,        // [num_tokens, kv_dim]
    const float* __restrict__ nope_scale,      // [num_tokens, 1]
    const float* __restrict__ rope_scale,      // [num_tokens, 1]
    int8_t* __restrict__ kv_cache,             // [num_blocks, block_size, kv_dim]
    float* __restrict__ nope_scale_cache,      // [num_blocks, block_size, 1]
    float* __restrict__ rope_scale_cache,      // [num_blocks, block_size, 1]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_size, const int num_blocks, const int kv_dim,
    const int kv_token_stride, const int nope_scale_token_stride,
    const int rope_scale_token_stride, const int cache_block_stride,
    const int nope_scale_cache_block_stride,
    const int rope_scale_cache_block_stride, const int cache_entry_stride,
    const int nope_scale_cache_entry_stride,
    const int rope_scale_cache_entry_stride, const int num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int token_idx = blockIdx.x * kWarpsPerBlock + warp_id;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t total_slots =
      static_cast<int64_t>(num_blocks) * block_size;
  if (slot_idx >= total_slots) {
    return;
  }

  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int32_t* src_ptr = reinterpret_cast<const int32_t*>(
      kv_int8 + token_idx * kv_token_stride);
  int32_t* dst_ptr = reinterpret_cast<int32_t*>(
      kv_cache + block_idx * cache_block_stride +
      block_offset * cache_entry_stride);
  const int num_int32 = kv_dim / static_cast<int>(sizeof(int32_t));

  // One warp handles one token. Each lane copies contiguous 4-byte chunks.
  for (int i = lane_id; i < num_int32; i += kWarpSize) {
    dst_ptr[i] = src_ptr[i];
  }

  // Only lane 0 writes scales.
  if (lane_id == 0) {
    nope_scale_cache[block_idx * nope_scale_cache_block_stride +
                     block_offset * nope_scale_cache_entry_stride] =
        nope_scale[token_idx * nope_scale_token_stride];
    rope_scale_cache[block_idx * rope_scale_cache_block_stride +
                     block_offset * rope_scale_cache_entry_stride] =
        rope_scale[token_idx * rope_scale_token_stride];
  }
}

void launch_concat_and_cache_dsa_int8(
    const int8_t* kv_int8, const float* nope_scale, const float* rope_scale,
    int8_t* kv_cache, float* nope_scale_cache, float* rope_scale_cache,
    const int64_t* slot_mapping, int block_size, int num_blocks, int kv_dim,
    int kv_token_stride, int nope_scale_token_stride,
    int rope_scale_token_stride, int cache_block_stride,
    int nope_scale_cache_block_stride, int rope_scale_cache_block_stride,
    int cache_entry_stride, int nope_scale_cache_entry_stride,
    int rope_scale_cache_entry_stride, int num_tokens, cudaStream_t stream) {
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid((num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock);
  dim3 block(kWarpsPerBlock * kWarpSize);
  concat_and_cache_dsa_int8_kernel<<<grid, block, 0, stream>>>(
      kv_int8, nope_scale, rope_scale, kv_cache, nope_scale_cache,
      rope_scale_cache, slot_mapping, block_size, num_blocks, kv_dim,
      kv_token_stride, nope_scale_token_stride, rope_scale_token_stride,
      cache_block_stride, nope_scale_cache_block_stride,
      rope_scale_cache_block_stride, cache_entry_stride,
      nope_scale_cache_entry_stride, rope_scale_cache_entry_stride,
      num_tokens);
}

}  // namespace

void concat_and_cache_dsa_int8(torch::Tensor& kv_int8,
                               torch::Tensor& nope_scale,
                               torch::Tensor& rope_scale,
                               torch::Tensor& kv_cache,
                               torch::Tensor& nope_scale_cache,
                               torch::Tensor& rope_scale_cache,
                               torch::Tensor& slot_mapping) {
  TORCH_CHECK(kv_int8.scalar_type() == at::ScalarType::Char &&
              kv_int8.is_cuda() && kv_int8.dim() == 2);
  TORCH_CHECK(nope_scale.scalar_type() == at::ScalarType::Float &&
              nope_scale.is_cuda() && nope_scale.dim() == 2);
  TORCH_CHECK(rope_scale.scalar_type() == at::ScalarType::Float &&
              rope_scale.is_cuda() && rope_scale.dim() == 2);
  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Char &&
              kv_cache.is_cuda() && kv_cache.dim() == 3);
  TORCH_CHECK(nope_scale_cache.scalar_type() == at::ScalarType::Float &&
              nope_scale_cache.is_cuda() && nope_scale_cache.dim() == 3);
  TORCH_CHECK(rope_scale_cache.scalar_type() == at::ScalarType::Float &&
              rope_scale_cache.is_cuda() && rope_scale_cache.dim() == 3);
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long &&
              slot_mapping.is_cuda() && slot_mapping.is_contiguous());

  TORCH_CHECK(kv_int8.size(0) == nope_scale.size(0) &&
              kv_int8.size(0) == rope_scale.size(0) &&
              kv_int8.size(0) == slot_mapping.size(0));
  TORCH_CHECK(nope_scale.size(1) == 1 && rope_scale.size(1) == 1);
  TORCH_CHECK(kv_int8.size(1) == kv_cache.size(2),
              "kv_int8/kv_cache kv_dim mismatch");
  TORCH_CHECK(kv_cache.size(0) == nope_scale_cache.size(0) &&
              kv_cache.size(1) == nope_scale_cache.size(1));
  TORCH_CHECK(kv_cache.size(0) == rope_scale_cache.size(0) &&
              kv_cache.size(1) == rope_scale_cache.size(1));
  TORCH_CHECK(nope_scale_cache.size(2) == 1 &&
              rope_scale_cache.size(2) == 1);
  TORCH_CHECK(kv_int8.size(1) % static_cast<int64_t>(sizeof(int32_t)) == 0,
              "kv_dim must be a multiple of 4");

  const int num_tokens = static_cast<int>(kv_int8.size(0));
  const int num_blocks = static_cast<int>(kv_cache.size(0));
  const int block_size = static_cast<int>(kv_cache.size(1));
  const int kv_dim = static_cast<int>(kv_int8.size(1));

  const int kv_token_stride = static_cast<int>(kv_int8.stride(0));
  const int nope_scale_token_stride =
      static_cast<int>(nope_scale.stride(0));
  const int rope_scale_token_stride =
      static_cast<int>(rope_scale.stride(0));
  const int cache_block_stride = static_cast<int>(kv_cache.stride(0));
  const int nope_scale_cache_block_stride =
      static_cast<int>(nope_scale_cache.stride(0));
  const int rope_scale_cache_block_stride =
      static_cast<int>(rope_scale_cache.stride(0));
  const int cache_entry_stride = static_cast<int>(kv_cache.stride(1));
  const int nope_scale_cache_entry_stride =
      static_cast<int>(nope_scale_cache.stride(1));
  const int rope_scale_cache_entry_stride =
      static_cast<int>(rope_scale_cache.stride(1));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_int8));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_concat_and_cache_dsa_int8(
      reinterpret_cast<const int8_t*>(kv_int8.data_ptr()),
      reinterpret_cast<const float*>(nope_scale.data_ptr()),
      reinterpret_cast<const float*>(rope_scale.data_ptr()),
      reinterpret_cast<int8_t*>(kv_cache.data_ptr()),
      reinterpret_cast<float*>(nope_scale_cache.data_ptr()),
      reinterpret_cast<float*>(rope_scale_cache.data_ptr()),
      reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
      block_size, num_blocks, kv_dim, kv_token_stride,
      nope_scale_token_stride, rope_scale_token_stride, cache_block_stride,
      nope_scale_cache_block_stride, rope_scale_cache_block_stride,
      cache_entry_stride, nope_scale_cache_entry_stride,
      rope_scale_cache_entry_stride, num_tokens, stream);
}

}  // namespace vllm_iluvatar
