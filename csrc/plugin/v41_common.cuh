// SPDX-License-Identifier: Apache-2.0
//
// Device-side pieces the DeepSeek-V4.1 plugin kernels share: the GPT-J rotation,
// the packed fp8_ds_mla record geometry, and the width-agnostic index load for
// vLLM's mixed int32/int64 metadata.
//
// The arithmetic here is a contract, not a convenience. Several V4.1 ops have to
// reproduce the reference (``off/vllm/vllm/models/deepseek_v41/common/ops``) and
// the ixformer kernels that already write these records bit for bit, so a second
// copy of an expression is a second thing to keep in step.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#ifdef ILUVATAR_HAS_FP8_FP4
#include <cuda_fp8.h>
#endif

namespace vllm_iluvatar {
namespace v41 {

// A GPT-J pair is adjacent channels, so one thread owning both keeps the
// rotation inside a thread and makes every load and store 4 bytes wide.
constexpr int kRopePacked = 2;

// Packed fp8_ds_mla record (V4 584 B form), laid out segregated inside a page:
// every row's data first, then every row's scales.
constexpr int kFp8TokenDataBytes = 576;  // 448 e4m3 NoPE + 64 bf16 RoPE
constexpr int kFp8NopeBytes = 448;
constexpr int kFp8ScaleBytes = 8;  // 7 UE8M0 + 1 pad
constexpr int kFp8NumScales = 7;   // 448 / 64
constexpr int kFp8QuantGroup = 64;
constexpr int kFp8RecordBytes = kFp8TokenDataBytes + kFp8ScaleBytes;
constexpr float kE4m3Max = 448.0f;

// vLLM hands positions and slot mappings as int64 but block tables as int32,
// and the dummy/profile paths sometimes narrow. Take either.
__device__ __forceinline__ int64_t load_index(const void* p, bool is_int64,
                                              int64_t i) {
  return is_int64 ? static_cast<const int64_t*>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

// Forward GPT-J RoPE on one adjacent pair. ``sin_v`` is the **negated** sine,
// which is how ds4_rope spells its forward case and therefore how every V4.1
// kernel on this platform spells it.
//
// The expression matters, not just the algebra: the cos product stays inside the
// FMA and the sin product rounds on its own, which is what ixformer's
// v41_rope_pair does and what the compressed records were written against.
// Intrinsics rather than plain arithmetic so the intent survives the optimizer.
__device__ __forceinline__ void rope_pair(float& even, float& odd, float cos_v,
                                          float sin_v) {
  const float e_sin = __fmul_rn(even, sin_v);
  const float o_sin = __fmul_rn(odd, sin_v);
  const float new_even = __fmaf_rn(even, cos_v, o_sin);
  const float new_odd = __fmaf_rn(odd, cos_v, -e_sin);
  even = new_even;
  odd = new_odd;
}

// Rotate a pair against ``cos_sin_cache[pos]``, whose row is [cos | sin] of
// ``rope_dim / 2`` each. ``pair`` indexes the pair inside the RoPE band.
__device__ __forceinline__ void rope_pair_at(float& even, float& odd,
                                             const float* __restrict__ cs_row,
                                             int pair, int rope_dim) {
  rope_pair(even, odd, cs_row[pair], -cs_row[rope_dim / 2 + pair]);
}

// Round-trip through bf16. The reference stores intermediate rows as bf16, so
// consumers see the rounded number; anything that has to match it has to round
// at the same points rather than carry fp32 all the way through.
__device__ __forceinline__ float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// Warp-level fp32 max over ``span`` lanes. ``span`` must be a power of two no
// wider than the warp, and the lanes of one span must not straddle a warp.
__device__ __forceinline__ float warp_reduce_max(float value, int span) {
#pragma unroll
  for (int offset = span / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor(value, offset));
  }
  return value;
}

// Warp-level fp32 sum over ``span`` lanes, same constraints.
__device__ __forceinline__ float warp_reduce_sum_span(float value, int span) {
#pragma unroll
  for (int offset = span / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

#ifdef ILUVATAR_HAS_FP8_FP4
// E4M3 bit pattern of a value already divided by its scale. The cast saturates
// at 448 rather than overflowing, which is the clamp the reference applies.
__device__ __forceinline__ uint8_t e4m3_bits(float x) {
  const __nv_fp8_e4m3 q = __nv_fp8_e4m3(x);
  return *reinterpret_cast<const uint8_t*>(&q);
}

// UE8M0 byte for an already-reduced amax: the exponent of the smallest power of
// two bringing amax inside E4M3's range, biased by 127.
__device__ __forceinline__ uint8_t ue8m0_bits(float amax) {
  if (amax == 0.0f) return 127u;  // 2^0
  const int exp = static_cast<int>(ceilf(log2f(amax / kE4m3Max)));
  return static_cast<uint8_t>(max(0, min(254, exp + 127)));
}

__device__ __forceinline__ float ue8m0_scale(uint8_t bits) {
  return exp2f(static_cast<float>(static_cast<int>(bits) - 127));
}

// The power-of-two scale the reference picks for a whole row: it divides amax by
// the E4M3 max and then rounds the quotient up to a power of two.
__device__ __forceinline__ float row_pow2_scale(float amax) {
  return exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / kE4m3Max)));
}
#endif  // ILUVATAR_HAS_FP8_FP4

}  // namespace v41
}  // namespace vllm_iluvatar
