// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace vllm_iluvatar {

template <typename scalar_t, int pack_num>
struct TypeTraits;

template <>
struct TypeTraits<float, 1> {
  using type = float;
};

template <>
struct TypeTraits<float, 2> {
  using type = float2;
};

template <>
struct TypeTraits<half, 1> {
  using type = half;
};

template <>
struct TypeTraits<half, 2> {
  using type = half2;
};

template <>
struct TypeTraits<__nv_bfloat16, 1> {
  using type = __nv_bfloat16;
};

template <>
struct TypeTraits<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};

template <>
struct TypeTraits<int8_t, 1> {
  using type = int8_t;
};

template <>
struct TypeTraits<int8_t, 2> {
  using type = int16_t;
};

template <>
struct TypeTraits<int8_t, 4> {
  using type = int32_t;
};

__device__ __forceinline__ float to_float(float value) {
  return value;
}

__device__ __forceinline__ float to_float(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

// Packed (2-element) converters aligned with to_float / from_float naming.
// Distinct from fused_qk_norm_rope.cu's local packed_to_float2 / float2_to_packed.
template <typename T>
__device__ __forceinline__ float2 to_float2(
    typename TypeTraits<T, 2>::type packed) {
  return make_float2(to_float(packed.x), to_float(packed.y));
}

template <typename T>
__device__ __forceinline__ typename TypeTraits<T, 2>::type from_float2(
    float2 value) {
  using packed_t = typename TypeTraits<T, 2>::type;
  packed_t packed;
  packed.x = from_float<T>(value.x);
  packed.y = from_float<T>(value.y);
  return packed;
}

// Warp-level fp32 sum reduction. Uses the compile-time constant warpSize
// (Iluvatar=64, NV=32); never hard-code 32 or 64.
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

// Max threads per block for block_reduce_sum smem sizing.
constexpr int kMaxThreadsPerBlock = 1024;

// Block-level fp32 sum reduction. ``smem`` must hold at least
// ``kMaxThreadsPerBlock / warpSize`` floats. Returns the block sum to every
// thread (broadcast via smem[0] + __syncthreads).
__device__ __forceinline__ float block_reduce_sum(float value, float* smem) {
  const int lane = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    smem[warpid] = value;
  }
  __syncthreads();

  value = (lane < num_warps) ? smem[lane] : 0.0f;
  value = warp_reduce_sum(value);
  if (threadIdx.x == 0) {
    smem[0] = value;
  }
  __syncthreads();
  return smem[0];
}

}  // namespace vllm_iluvatar
