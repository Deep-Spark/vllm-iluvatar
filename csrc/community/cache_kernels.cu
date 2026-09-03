// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.reshape_and_cache_fp8 -- vLLM community signature.
//
// K/V arrive as FP16/BF16 and are statically quantized while being scattered
// into the HND paged cache. Query quantization uses the same packed conversion
// in a standalone static_scaled_fp8_quant kernel. The cache indexing follows
// ixformer's vllm_reshape_and_cache implementation while retaining vLLM V1's
// rule that slot_mapping.size(0), rather than key.size(0), is the number of real
// tokens.
//
// FP8 E4M3 is an ivcore30 (corex30) feature, so the kernels below are compiled
// only under ILUVATAR_HAS_FP8_FP4. Both operators are still declared and
// bound on every arch -- on the other platforms they raise instead of silently
// disappearing from torch.ops._C. torch.ops._C.compiled_cuda_arch() tells
// callers which build they are on.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "ops.h"

#ifdef ILUVATAR_HAS_FP8_FP4
#include <cuda_fp8.h>

#include "utils.cuh"
#endif

namespace vllm_iluvatar {

#ifdef ILUVATAR_HAS_FP8_FP4
namespace {

constexpr int kElemsPerVec = 8;

template <typename scalar_t>
__device__ __forceinline__ uint32_t quantize_fp8x4(
    typename TypeTraits<scalar_t, 2>::type a,
    typename TypeTraits<scalar_t, 2>::type b, float scale) {
  float4 values = make_float4(
      to_float(reinterpret_cast<scalar_t*>(&a)[0]) / scale,
      to_float(reinterpret_cast<scalar_t*>(&a)[1]) / scale,
      to_float(reinterpret_cast<scalar_t*>(&b)[0]) / scale,
      to_float(reinterpret_cast<scalar_t*>(&b)[1]) / scale);
  return __nv_fp8x4_e4m3(values).__x;
}

template <typename scalar_t>
__device__ __forceinline__ uint2 quantize_fp8x8(const scalar_t* input,
                                                float scale) {
  const uint4 raw = *reinterpret_cast<const uint4*>(input);
  using scalar2_t = typename TypeTraits<scalar_t, 2>::type;
  const scalar2_t* values = reinterpret_cast<const scalar2_t*>(&raw);
  uint2 packed;
  packed.x = quantize_fp8x4<scalar_t>(values[0], values[1], scale);
  packed.y = quantize_fp8x4<scalar_t>(values[2], values[3], scale);
  return packed;
}

template <typename scalar_t>
__device__ __forceinline__ __nv_fp8_e4m3 quantize_fp8(
    scalar_t input, float scale) {
  return __nv_fp8_e4m3(to_float(input) / scale);
}

template <typename scalar_t, bool vectorized>
__global__ void static_scaled_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output,
    const scalar_t* __restrict__ input, const float* __restrict__ scale,
    int num_heads, int hidden_size, int64_t input_token_stride,
    int64_t input_head_stride) {
  const int token_idx = blockIdx.x;
  const float quant_scale = scale[0];

  for (int head_idx = threadIdx.x; head_idx < num_heads;
       head_idx += blockDim.x) {
    const scalar_t* input_head =
        input + static_cast<int64_t>(token_idx) * input_token_stride +
        static_cast<int64_t>(head_idx) * input_head_stride;
    __nv_fp8_e4m3* output_head =
        output +
        (static_cast<int64_t>(token_idx) * num_heads + head_idx) * hidden_size;

    if constexpr (vectorized) {
      const int num_vec = hidden_size / kElemsPerVec;
      for (int vec_idx = threadIdx.y; vec_idx < num_vec;
           vec_idx += blockDim.y) {
        reinterpret_cast<uint2*>(output_head)[vec_idx] =
            quantize_fp8x8(input_head + vec_idx * kElemsPerVec, quant_scale);
      }
    } else {
      for (int i = threadIdx.y; i < hidden_size; i += blockDim.y) {
        output_head[i] = quantize_fp8(input_head[i], quant_scale);
      }
    }
  }
}

template <typename scalar_t, bool vectorized>
__global__ void reshape_and_cache_fp8_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ key_scale,
    const float* __restrict__ value_scale,
    const int64_t* __restrict__ slot_mapping, const int64_t key_token_stride,
    const int64_t key_head_stride, const int64_t value_token_stride,
    const int64_t value_head_stride, const int num_heads,
    const int key_head_size, const int value_head_size, const int block_size,
    const int scale_stride,
    const int64_t key_cache_block_stride,
    const int64_t key_cache_head_stride,
    const int64_t key_cache_token_stride,
    const int64_t value_cache_block_stride,
    const int64_t value_cache_head_stride,
    const int64_t value_cache_token_stride) {
  const int token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  if constexpr (vectorized) {
    const int key_num_vec = num_heads * key_head_size / kElemsPerVec;
    for (int vec_idx = threadIdx.x; vec_idx < key_num_vec;
         vec_idx += blockDim.x) {
      const int i = vec_idx * kElemsPerVec;
      const int head_idx = i / key_head_size;
      const int head_offset = i % key_head_size;
      const int64_t src_idx =
          static_cast<int64_t>(token_idx) * key_token_stride +
          static_cast<int64_t>(head_idx) * key_head_stride + head_offset;
      const int64_t dst_idx =
          block_idx * key_cache_block_stride +
          static_cast<int64_t>(head_idx) * key_cache_head_stride +
          block_offset * key_cache_token_stride + head_offset;
      *reinterpret_cast<uint2*>(key_cache + dst_idx) =
          quantize_fp8x8(key + src_idx, key_scale[head_idx * scale_stride]);
    }
  } else {
    const int key_numel = num_heads * key_head_size;
    for (int i = threadIdx.x; i < key_numel; i += blockDim.x) {
      const int head_idx = i / key_head_size;
      const int head_offset = i % key_head_size;
      const int64_t src_idx =
          static_cast<int64_t>(token_idx) * key_token_stride +
          static_cast<int64_t>(head_idx) * key_head_stride + head_offset;
      const int64_t dst_idx =
          block_idx * key_cache_block_stride +
          static_cast<int64_t>(head_idx) * key_cache_head_stride +
          block_offset * key_cache_token_stride + head_offset;
      key_cache[dst_idx] =
          quantize_fp8(key[src_idx], key_scale[head_idx * scale_stride]);
    }
  }

  if constexpr (vectorized) {
    const int value_num_vec = num_heads * value_head_size / kElemsPerVec;
    for (int vec_idx = threadIdx.x; vec_idx < value_num_vec;
         vec_idx += blockDim.x) {
      const int i = vec_idx * kElemsPerVec;
      const int head_idx = i / value_head_size;
      const int head_offset = i % value_head_size;
      const int64_t src_idx =
          static_cast<int64_t>(token_idx) * value_token_stride +
          static_cast<int64_t>(head_idx) * value_head_stride + head_offset;
      const int64_t dst_idx =
          block_idx * value_cache_block_stride +
          static_cast<int64_t>(head_idx) * value_cache_head_stride +
          block_offset * value_cache_token_stride + head_offset;
      *reinterpret_cast<uint2*>(value_cache + dst_idx) = quantize_fp8x8(
          value + src_idx, value_scale[head_idx * scale_stride]);
    }
  } else {
    const int value_numel = num_heads * value_head_size;
    for (int i = threadIdx.x; i < value_numel; i += blockDim.x) {
      const int head_idx = i / value_head_size;
      const int head_offset = i % value_head_size;
      const int64_t src_idx =
          static_cast<int64_t>(token_idx) * value_token_stride +
          static_cast<int64_t>(head_idx) * value_head_stride + head_offset;
      const int64_t dst_idx =
          block_idx * value_cache_block_stride +
          static_cast<int64_t>(head_idx) * value_cache_head_stride +
          block_offset * value_cache_token_stride + head_offset;
      value_cache[dst_idx] =
          quantize_fp8(value[src_idx], value_scale[head_idx * scale_stride]);
    }
  }
}

bool can_vectorize_cache_quant(
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache) {
  return reinterpret_cast<uintptr_t>(key.data_ptr()) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(key_cache.data_ptr()) % 8 == 0 &&
         reinterpret_cast<uintptr_t>(value_cache.data_ptr()) % 8 == 0 &&
         key.size(2) % kElemsPerVec == 0 &&
         value.size(2) % kElemsPerVec == 0 &&
         key.stride(0) % kElemsPerVec == 0 &&
         key.stride(1) % kElemsPerVec == 0 &&
         value.stride(0) % kElemsPerVec == 0 &&
         value.stride(1) % kElemsPerVec == 0 &&
         key_cache.stride(0) % kElemsPerVec == 0 &&
         key_cache.stride(1) % kElemsPerVec == 0 &&
         key_cache.stride(2) % kElemsPerVec == 0 &&
         value_cache.stride(0) % kElemsPerVec == 0 &&
         value_cache.stride(1) % kElemsPerVec == 0 &&
         value_cache.stride(2) % kElemsPerVec == 0;
}

void check_reshape_and_cache_fp8_args(
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping, const std::string& kv_cache_dtype,
    const torch::Tensor& k_scale, const torch::Tensor& v_scale) {
  TORCH_CHECK(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3",
              "reshape_and_cache_fp8 only supports fp8/fp8_e4m3 cache, got ",
              kv_cache_dtype);
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && slot_mapping.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda(),
              "reshape_and_cache_fp8 expects CUDA tensors");
  TORCH_CHECK(key.device() == value.device() &&
                  key.device() == key_cache.device() &&
                  key.device() == value_cache.device() &&
                  key.device() == slot_mapping.device() &&
                  key.device() == k_scale.device() &&
                  key.device() == v_scale.device(),
              "reshape_and_cache_fp8 expects all tensors on the same device");

  const auto input_type = key.scalar_type();
  constexpr auto fp8_type = at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK((input_type == at::ScalarType::Half ||
               input_type == at::ScalarType::BFloat16) &&
                  value.scalar_type() == input_type,
              "key and value must have the same float16 or bfloat16 dtype");
  TORCH_CHECK(key_cache.scalar_type() == fp8_type &&
                  value_cache.scalar_type() == fp8_type,
              "key_cache and value_cache must be float8_e4m3fn");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long &&
                  slot_mapping.dim() == 1 && slot_mapping.is_contiguous(),
              "slot_mapping must be a contiguous int64 vector");
  TORCH_CHECK(k_scale.scalar_type() == at::ScalarType::Float &&
                  v_scale.scalar_type() == at::ScalarType::Float &&
                  k_scale.is_contiguous() && v_scale.is_contiguous(),
              "k_scale and v_scale must be contiguous float32 tensors");

  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key and value must have shape [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "key_cache and value_cache must have shape "
              "[num_blocks, num_heads, block_size, head_size]");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "key and value last dimensions must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "key_cache and value_cache last dimensions must be contiguous");

  const int64_t num_heads = key.size(1);
  TORCH_CHECK(value.size(0) >= slot_mapping.size(0) &&
                  key.size(0) >= slot_mapping.size(0),
              "slot_mapping cannot contain more tokens than key/value");
  TORCH_CHECK(value.size(1) == num_heads,
              "key and value must have the same number of heads");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
                  key_cache.size(1) == num_heads &&
                  value_cache.size(1) == num_heads &&
                  key_cache.size(2) == value_cache.size(2),
              "key/value caches must have matching block/head dimensions");
  TORCH_CHECK(key_cache.size(3) >= key.size(2) &&
                  value_cache.size(3) >= value.size(2),
              "cache head dimensions must fit key/value head dimensions");
  TORCH_CHECK(k_scale.sizes() == v_scale.sizes(),
              "k_scale and v_scale must have the same shape");
  TORCH_CHECK(k_scale.numel() == 1 || k_scale.numel() == num_heads,
              "k_scale and v_scale must contain one value or one per head");
}

void check_static_scaled_fp8_quant_args(
    const torch::Tensor& result, const torch::Tensor& input,
    const torch::Tensor& scale,
    const std::optional<std::vector<int64_t>>& group_shape) {
  TORCH_CHECK(!group_shape.has_value(),
              "static_scaled_fp8_quant currently supports per-tensor scale "
              "only; group_shape must be None");
  TORCH_CHECK(result.is_cuda() && input.is_cuda() && scale.is_cuda(),
              "static_scaled_fp8_quant expects CUDA tensors");
  TORCH_CHECK(result.device() == input.device() &&
                  result.device() == scale.device(),
              "static_scaled_fp8_quant expects all tensors on the same device");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "static_scaled_fp8_quant input must be float16 or bfloat16");
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "static_scaled_fp8_quant result must be float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float &&
                  scale.is_contiguous() && scale.numel() == 1,
              "static_scaled_fp8_quant scale must be one contiguous float32 "
              "value");
  TORCH_CHECK(input.dim() == 3,
              "static_scaled_fp8_quant input must have shape "
              "[num_tokens, num_heads, head_dim]");
  TORCH_CHECK(input.size(-1) > 0 && input.stride(-1) == 1,
              "static_scaled_fp8_quant head_dim must be non-empty and "
              "contiguous");
  TORCH_CHECK(result.sizes() == input.sizes() && result.is_contiguous(),
              "static_scaled_fp8_quant result must be contiguous and have the "
              "same shape as input");
}

}  // namespace
#endif  // ILUVATAR_HAS_FP8_FP4

void static_scaled_fp8_quant(
    torch::Tensor& result, torch::Tensor& input, torch::Tensor& scale,
    const std::optional<std::vector<int64_t>>& group_shape) {
#ifndef ILUVATAR_HAS_FP8_FP4
  TORCH_CHECK(false,
              "static_scaled_fp8_quant requires the ivcore30 (corex30) FP8 "
              "kernels; this vllm_iluvatar._C was built for ",
              VLLM_ILUVATAR_CUDA_ARCH,
              ". Query torch.ops._C.compiled_cuda_arch() before taking the FP8 "
              "path.");
#else
  check_static_scaled_fp8_quant_args(result, input, scale, group_shape);

  const int num_tokens = static_cast<int>(input.size(0));
  const int num_heads = static_cast<int>(input.size(1));
  const int hidden_size = static_cast<int>(input.size(2));
  const int num_rows = num_tokens * num_heads;
  if (num_rows == 0) {
    return;
  }

  const int64_t input_head_stride = input.stride(1);
  const int64_t input_token_stride = input.stride(0);

  const bool vectorized =
      hidden_size % kElemsPerVec == 0 &&
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(result.data_ptr()) % 8 == 0 &&
      input_head_stride % kElemsPerVec == 0 &&
      input_token_stride % kElemsPerVec == 0;
  const dim3 grid(num_tokens);
  const dim3 block(std::min(8, num_heads), 64);
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_STATIC_SCALED_FP8_QUANT(SCALAR_T, VECTORIZED)                  \
  static_scaled_fp8_quant_kernel<SCALAR_T, VECTORIZED>                        \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<__nv_fp8_e4m3*>(result.data_ptr()),                \
          reinterpret_cast<const SCALAR_T*>(input.data_ptr()),                \
          reinterpret_cast<const float*>(scale.data_ptr()), num_heads,        \
          hidden_size, input_token_stride, input_head_stride)

  if (input.scalar_type() == at::ScalarType::Half) {
    if (vectorized) {
      LAUNCH_STATIC_SCALED_FP8_QUANT(half, true);
    } else {
      LAUNCH_STATIC_SCALED_FP8_QUANT(half, false);
    }
  } else {
    if (vectorized) {
      LAUNCH_STATIC_SCALED_FP8_QUANT(__nv_bfloat16, true);
    } else {
      LAUNCH_STATIC_SCALED_FP8_QUANT(__nv_bfloat16, false);
    }
  }
#undef LAUNCH_STATIC_SCALED_FP8_QUANT
#endif  // ILUVATAR_HAS_FP8_FP4
}

void reshape_and_cache_fp8(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
#ifndef ILUVATAR_HAS_FP8_FP4
  TORCH_CHECK(false,
              "reshape_and_cache_fp8 requires the ivcore30 (corex30) FP8 "
              "kernels; this vllm_iluvatar._C was built for ",
              VLLM_ILUVATAR_CUDA_ARCH,
              ". Query torch.ops._C.compiled_cuda_arch() before taking the FP8 "
              "path.");
#else
  check_reshape_and_cache_fp8_args(key, value, key_cache, value_cache,
                                   slot_mapping, kv_cache_dtype, k_scale,
                                   v_scale);

  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  if (num_tokens == 0) {
    return;
  }

  const int num_heads = static_cast<int>(key.size(1));
  const int key_head_size = static_cast<int>(key.size(2));
  const int value_head_size = static_cast<int>(value.size(2));
  const int block_size = static_cast<int>(key_cache.size(2));
  dim3 grid(num_tokens);
  dim3 block(std::min(std::max(num_heads * key_head_size,
                               num_heads * value_head_size),
                      512));

  const at::cuda::OptionalCUDAGuard device_guard(key.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int scale_stride = k_scale.numel() > 1 ? 1 : 0;
  const bool vectorized =
      can_vectorize_cache_quant(key, value, key_cache, value_cache);

#define LAUNCH_RESHAPE_AND_CACHE_FP8(SCALAR_T, VECTORIZED)                    \
  reshape_and_cache_fp8_kernel<SCALAR_T, VECTORIZED>                          \
      <<<grid, block, 0, stream>>>(                                           \
      reinterpret_cast<const SCALAR_T*>(key.data_ptr()),                      \
      reinterpret_cast<const SCALAR_T*>(value.data_ptr()),                    \
      reinterpret_cast<__nv_fp8_e4m3*>(key_cache.data_ptr()),                 \
      reinterpret_cast<__nv_fp8_e4m3*>(value_cache.data_ptr()),               \
      reinterpret_cast<const float*>(k_scale.data_ptr()),                     \
      reinterpret_cast<const float*>(v_scale.data_ptr()),                     \
      reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),              \
      key.stride(0), key.stride(1), value.stride(0), value.stride(1),         \
      num_heads, key_head_size, value_head_size, block_size, scale_stride,    \
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),          \
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2))

  if (key.scalar_type() == at::ScalarType::Half) {
    if (vectorized) {
      LAUNCH_RESHAPE_AND_CACHE_FP8(half, true);
    } else {
      LAUNCH_RESHAPE_AND_CACHE_FP8(half, false);
    }
  } else {
    if (vectorized) {
      LAUNCH_RESHAPE_AND_CACHE_FP8(__nv_bfloat16, true);
    } else {
      LAUNCH_RESHAPE_AND_CACHE_FP8(__nv_bfloat16, false);
    }
  }
#undef LAUNCH_RESHAPE_AND_CACHE_FP8
#endif  // ILUVATAR_HAS_FP8_FP4
}

}  // namespace vllm_iluvatar
