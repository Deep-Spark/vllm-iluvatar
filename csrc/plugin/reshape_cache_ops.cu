// SPDX-License-Identifier: Apache-2.0
// Native paged-cache writers migrated from ixformer.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

#include "ops.h"
#include "utils.cuh"

namespace vllm_iluvatar {
namespace {

template <typename scalar_t, int pack_num>
__global__ void reshape_and_cache_with_scale_kernel(
    const scalar_t *__restrict__ key, const scalar_t *__restrict__ value,
    const float *__restrict__ key_scale, scalar_t *__restrict__ key_cache,
    scalar_t *__restrict__ value_cache, float *__restrict__ key_scale_cache,
    const int64_t *__restrict__ slot_mapping, const int key_token_stride,
    const int value_token_stride, const int num_heads, const int head_size,
    const int block_size, 
    const int64_t key_cache_block_stride,
    const int64_t key_cache_head_stride, 
    const int64_t key_cache_token_stride,
    const int64_t value_cache_block_stride,
    const int64_t value_cache_head_stride,
    const int64_t value_cache_token_stride,
    const int64_t key_scale_cache_block_stride,
    const int64_t key_scale_cache_head_stride,
    const int64_t key_scale_cache_token_stride) {
  const int token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  using packed_t = typename TypeTraits<scalar_t, pack_num>::type;
  const int numel = num_heads * head_size;

  for (int i = threadIdx.x * pack_num; i < numel; i += blockDim.x * pack_num) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int src_key_idx = token_idx * key_token_stride + i;
    const int src_value_idx = token_idx * value_token_stride + i;
    const int64_t dst_key_idx =
        block_idx * key_cache_block_stride +
        static_cast<int64_t>(head_idx) * key_cache_head_stride +
        block_offset * key_cache_token_stride + head_offset;
    const int64_t dst_value_idx =
        block_idx * value_cache_block_stride +
        static_cast<int64_t>(head_idx) * value_cache_head_stride +
        block_offset * value_cache_token_stride + head_offset;
    *reinterpret_cast<packed_t *>(key_cache + dst_key_idx) =
        *reinterpret_cast<const packed_t *>(key + src_key_idx);
    *reinterpret_cast<packed_t *>(value_cache + dst_value_idx) =
        *reinterpret_cast<const packed_t *>(value + src_value_idx);
  }

  for (int head_idx = threadIdx.x; head_idx < num_heads;
       head_idx += blockDim.x) {
    const int64_t dst_scale_idx =
        block_idx * key_scale_cache_block_stride +
        static_cast<int64_t>(head_idx) * key_scale_cache_head_stride +
        block_offset * key_scale_cache_token_stride;
    key_scale_cache[dst_scale_idx] =
        key_scale[token_idx * num_heads + head_idx];
  }
}

template <typename scalar_t, int value_pack_num>
__global__ void reshape_and_cache_mix_kernel(
    const int8_t *__restrict__ key, const scalar_t *__restrict__ value,
    const float *__restrict__ key_scale, int8_t *__restrict__ key_cache,
    scalar_t *__restrict__ value_cache, float *__restrict__ key_scale_cache,
    const int64_t *__restrict__ slot_mapping, const int key_token_stride,
    const int value_token_stride, const int num_heads, const int head_size,
    const int block_size, 
    const int64_t key_cache_block_stride,
    const int64_t key_cache_head_stride, 
    const int64_t key_cache_token_stride,
    const int64_t value_cache_block_stride,
    const int64_t value_cache_head_stride,
    const int64_t value_cache_token_stride,
    const int64_t key_scale_cache_block_stride,
    const int64_t key_scale_cache_head_stride,
    const int64_t key_scale_cache_token_stride) {
  const int token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  constexpr int key_pack_num = sizeof(int32_t) / sizeof(int8_t);
  const int numel = num_heads * head_size;

  for (int i = threadIdx.x * key_pack_num; i < numel;
       i += blockDim.x * key_pack_num) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int src_key_idx = token_idx * key_token_stride + i;
    const int64_t dst_key_idx =
        block_idx * key_cache_block_stride +
        static_cast<int64_t>(head_idx) * key_cache_head_stride +
        block_offset * key_cache_token_stride + head_offset;
    *reinterpret_cast<int32_t *>(key_cache + dst_key_idx) =
        *reinterpret_cast<const int32_t *>(key + src_key_idx);
  }

  using packed_value_t =
      typename TypeTraits<scalar_t, value_pack_num>::type;
  for (int i = threadIdx.x * value_pack_num; i < numel;
       i += blockDim.x * value_pack_num) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int src_value_idx = token_idx * value_token_stride + i;
    const int64_t dst_value_idx =
        block_idx * value_cache_block_stride +
        static_cast<int64_t>(head_idx) * value_cache_head_stride +
        block_offset * value_cache_token_stride + head_offset;
    *reinterpret_cast<packed_value_t *>(value_cache + dst_value_idx) =
        *reinterpret_cast<const packed_value_t *>(value + src_value_idx);
  }

  for (int head_idx = threadIdx.x; head_idx < num_heads;
       head_idx += blockDim.x) {
    const int64_t dst_scale_idx =
        block_idx * key_scale_cache_block_stride +
        static_cast<int64_t>(head_idx) * key_scale_cache_head_stride +
        block_offset * key_scale_cache_token_stride;
    key_scale_cache[dst_scale_idx] =
        key_scale[token_idx * num_heads + head_idx];
  }
}

void check_reshape_and_cache_args(const torch::Tensor &key,
                                  const torch::Tensor &value,
                                  const torch::Tensor &key_cache,
                                  const torch::Tensor &value_cache,
                                  const torch::Tensor &slot_mapping,
                                  const torch::Tensor &key_scale,
                                  const torch::Tensor &key_scale_cache) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
              value_cache.is_cuda() && slot_mapping.is_cuda() &&
              key_scale.is_cuda() && key_scale_cache.is_cuda());
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3);
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4);
  TORCH_CHECK(key_scale.dim() == 2 && key_scale_cache.dim() == 3);
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long &&
              slot_mapping.dim() == 1 && slot_mapping.is_contiguous());
  TORCH_CHECK(key_scale.scalar_type() == at::ScalarType::Float &&
              key_scale.is_contiguous());
  TORCH_CHECK(key_scale_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1);
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
              key_scale_cache.stride(2) == 1);

  const int64_t num_tokens = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t head_size = key.size(2);
  TORCH_CHECK(value.sizes() == key.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(key.stride(1) == head_size && value.stride(1) == head_size,
              "key and value must be contiguous in their last two dimensions");
  TORCH_CHECK(slot_mapping.size(0) == num_tokens);
  TORCH_CHECK(key_scale.size(0) == num_tokens &&
              key_scale.size(1) == num_heads);
  TORCH_CHECK(key_cache.size(1) == num_heads && key_cache.size(3) == head_size);
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes());
  TORCH_CHECK(key_scale_cache.size(0) == key_cache.size(0) &&
              key_scale_cache.size(1) == num_heads &&
              key_scale_cache.size(2) == key_cache.size(2));
}

template <typename scalar_t>
void launch_reshape_and_cache_mix(
    const torch::Tensor &key, const torch::Tensor &value,
    torch::Tensor &key_cache, torch::Tensor &value_cache,
    const torch::Tensor &slot_mapping, const torch::Tensor &key_scale,
    torch::Tensor &key_scale_cache, cudaStream_t stream) {
  const int num_tokens = static_cast<int>(key.size(0));
  if (num_tokens == 0) {
    return;
  }
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(2));
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

#define LAUNCH_MIX(PACK_NUM)                                                   \
  reshape_and_cache_mix_kernel<scalar_t, PACK_NUM>                             \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const int8_t *>(key.data_ptr()),                    \
          reinterpret_cast<const scalar_t *>(value.data_ptr()),                \
          reinterpret_cast<const float *>(key_scale.data_ptr()),               \
          reinterpret_cast<int8_t *>(key_cache.data_ptr()),                    \
          reinterpret_cast<scalar_t *>(value_cache.data_ptr()),                \
          reinterpret_cast<float *>(key_scale_cache.data_ptr()),               \
          reinterpret_cast<const int64_t *>(slot_mapping.data_ptr()),          \
          static_cast<int>(key.stride(0)), static_cast<int>(value.stride(0)),  \
          num_heads, head_size, block_size, key_cache.stride(0),               \
          key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),     \
          value_cache.stride(1), value_cache.stride(2),                        \
          key_scale_cache.stride(0), key_scale_cache.stride(1),                \
          key_scale_cache.stride(2))

  if (head_size % 2 == 0) {
    LAUNCH_MIX(2);
  } else {
    LAUNCH_MIX(1);
  }
#undef LAUNCH_MIX
}

} // namespace

void reshape_and_cache_with_scale(torch::Tensor &key, torch::Tensor &value,
                                  torch::Tensor &key_scale,
                                  torch::Tensor &key_cache,
                                  torch::Tensor &value_cache,
                                  torch::Tensor &key_scale_cache,
                                  torch::Tensor &slot_mapping) {
  check_reshape_and_cache_args(key, value, key_cache, value_cache, slot_mapping,
                               key_scale, key_scale_cache);
  TORCH_CHECK(key.scalar_type() == at::ScalarType::Char &&
              value.scalar_type() == at::ScalarType::Char &&
              key_cache.scalar_type() == at::ScalarType::Char &&
              value_cache.scalar_type() == at::ScalarType::Char);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_tokens = static_cast<int>(key.size(0));
  if (num_tokens == 0) {
    return;
  }
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(2));
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

#define LAUNCH_WITH_SCALE(PACK_NUM)                                            \
  reshape_and_cache_with_scale_kernel<int8_t, PACK_NUM>                        \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const int8_t *>(key.data_ptr()),                    \
          reinterpret_cast<const int8_t *>(value.data_ptr()),                  \
          reinterpret_cast<const float *>(key_scale.data_ptr()),               \
          reinterpret_cast<int8_t *>(key_cache.data_ptr()),                    \
          reinterpret_cast<int8_t *>(value_cache.data_ptr()),                  \
          reinterpret_cast<float *>(key_scale_cache.data_ptr()),               \
          reinterpret_cast<const int64_t *>(slot_mapping.data_ptr()),          \
          static_cast<int>(key.stride(0)), static_cast<int>(value.stride(0)),  \
          num_heads, head_size, block_size, key_cache.stride(0),               \
          key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),     \
          value_cache.stride(1), value_cache.stride(2),                        \
          key_scale_cache.stride(0), key_scale_cache.stride(1),                \
          key_scale_cache.stride(2))

  if (head_size % 4 == 0) {
    LAUNCH_WITH_SCALE(4);
  } else if (head_size % 2 == 0) {
    LAUNCH_WITH_SCALE(2);
  } else {
    LAUNCH_WITH_SCALE(1);
  }
#undef LAUNCH_WITH_SCALE
}

void reshape_and_cache_mix(torch::Tensor &key, 
                           torch::Tensor &value,
                           torch::Tensor &key_scale, 
                           torch::Tensor &key_cache,
                           torch::Tensor &value_cache,
                           torch::Tensor &key_scale_cache,
                           torch::Tensor &slot_mapping) {
  check_reshape_and_cache_args(key, value, key_cache, value_cache, slot_mapping,
                               key_scale, key_scale_cache);
  TORCH_CHECK(key.scalar_type() == at::ScalarType::Char &&
              key_cache.scalar_type() == at::ScalarType::Char);
  TORCH_CHECK(value.scalar_type() == value_cache.scalar_type());
  TORCH_CHECK(key.size(2) % 4 == 0, "head_size must be a multiple of 4");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (value.scalar_type() == at::ScalarType::Half) {
    launch_reshape_and_cache_mix<half>(key, value, key_cache, value_cache,
                                       slot_mapping, key_scale, key_scale_cache,
                                       stream);
  } else if (value.scalar_type() == at::ScalarType::BFloat16) {
    launch_reshape_and_cache_mix<__nv_bfloat16>(key, value, key_cache,
                                                value_cache, slot_mapping,
                                                key_scale, key_scale_cache,
                                                stream);
  } else {
    TORCH_CHECK(false, "value must be float16 or bfloat16");
  }
}

} // namespace vllm_iluvatar
