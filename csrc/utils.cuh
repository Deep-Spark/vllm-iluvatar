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

}  // namespace vllm_iluvatar
