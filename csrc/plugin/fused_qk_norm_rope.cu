// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 fused Q/K Gemma-RMSNorm + partial NeoX RoPE + gate extraction.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "ops.h"
#include "utils.cuh"

namespace vllm_iluvatar {
namespace {

constexpr int kWarpSize = 64;
constexpr int kMaxWarpsPerBlock = 16;

template <typename T>
__device__ __forceinline__ float2 packed_to_float2(
    typename TypeTraits<T, 2>::type packed) {
  return make_float2(to_float(packed.x), to_float(packed.y));
}

template <typename T>
__device__ __forceinline__ typename TypeTraits<T, 2>::type float2_to_packed(
    float2 value) {
  using packed_t = typename TypeTraits<T, 2>::type;
  packed_t packed;
  packed.x = from_float<T>(value.x);
  packed.y = from_float<T>(value.y);
  return packed;
}

template <typename scalar_t, typename cache_t, typename position_t,
          int HEAD_DIM>
__global__ void fused_qk_rmsnorm_rope_gate_kernel(
    const scalar_t* __restrict__ q_gate, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const cache_t* __restrict__ cos_sin_cache,
    const position_t* __restrict__ positions, scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out, scalar_t* __restrict__ gate_out,
    int64_t q_gate_stride, int64_t k_stride, int64_t cache_stride,
    int num_q_heads, int num_kv_heads, int rotary_dim, float epsilon) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  const int total_heads = num_q_heads + num_kv_heads;
  const int half_rotary = rotary_dim / 2;

  for (int head = warp; head < total_heads; head += num_warps) {
    const bool is_q = head < num_q_heads;
    const int local_head = is_q ? head : head - num_q_heads;
    const scalar_t* row =
        is_q ? q_gate + static_cast<int64_t>(token) * q_gate_stride +
                         static_cast<int64_t>(local_head) * 2 * HEAD_DIM
             : k + static_cast<int64_t>(token) * k_stride +
                       static_cast<int64_t>(local_head) * HEAD_DIM;
    const scalar_t* weight = is_q ? q_weight : k_weight;
    scalar_t* out =
        is_q ? q_out + (static_cast<int64_t>(token) * num_q_heads +
                        local_head) *
                           HEAD_DIM
             : k_out + (static_cast<int64_t>(token) * num_kv_heads +
                        local_head) *
                           HEAD_DIM;
    using scalar_pack_t = typename TypeTraits<scalar_t, 2>::type;
    using cache_pack_t = typename TypeTraits<cache_t, 2>::type;
    const scalar_pack_t* packed_row =
        reinterpret_cast<const scalar_pack_t*>(row);
    const scalar_pack_t* packed_weight =
        reinterpret_cast<const scalar_pack_t*>(weight);
    scalar_pack_t* packed_out = reinterpret_cast<scalar_pack_t*>(out);

    // Keep the complete input head in registers. Each lane performs one
    // 32-bit packed load per unroll round, so each float2 contains two
    // contiguous fp16/bf16 values.
    constexpr int UNROLL = HEAD_DIM / 128;
    float2 values[UNROLL];
    float square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const int pair = i * kWarpSize + lane;
      values[i] = packed_to_float2<scalar_t>(packed_row[pair]);
      square_sum +=
          values[i].x * values[i].x + values[i].y * values[i].y;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      square_sum += __shfl_xor(square_sum, offset);
    }
    const float inv_rms =
        rsqrtf(square_sum / static_cast<float>(HEAD_DIM) + epsilon);

    // Normalize entirely in fp32 and retain the results in registers. Conversion
    // back to fp16/bf16 happens only at the final output stores.
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const int pair = i * kWarpSize + lane;
      const float2 weights =
          packed_to_float2<scalar_t>(packed_weight[pair]);
      values[i].x *= inv_rms * (weights.x + 1.0f);
      values[i].y *= inv_rms * (weights.y + 1.0f);
      // Match the Triton reference's RMSNorm -> input-dtype storage -> RoPE
      // boundary while retaining the rounded values in fp32 registers.
      values[i].x = to_float(from_float<scalar_t>(values[i].x));
      values[i].y = to_float(from_float<scalar_t>(values[i].y));
    }

    const int64_t position = static_cast<int64_t>(positions[token]);
    const cache_t* cache =
        cos_sin_cache + position * static_cast<int64_t>(cache_stride);
    const cache_t* cos = cache;
    const cache_t* sin = cache + half_rotary;
    const cache_pack_t* packed_cos =
        reinterpret_cast<const cache_pack_t*>(cos);
    const cache_pack_t* packed_sin =
        reinterpret_cast<const cache_pack_t*>(sin);

    // One active lane handles two adjacent x values and their two NeoX y
    // partners. For rotary_dim=64, only lanes [0, 16) execute this section.
    const int rotary_packs_per_half = half_rotary / 2;
    float2 y_from_register;
    if constexpr (UNROLL == 2) {
      if (rotary_dim == HEAD_DIM) {
        y_from_register = values[1];
      } else {
        y_from_register =
            make_float2(
                __shfl_xor(values[0].x, rotary_packs_per_half),
                __shfl_xor(values[0].y, rotary_packs_per_half));
      }
    } else {
      y_from_register =
          make_float2(__shfl_xor(values[0].x, rotary_packs_per_half),
                      __shfl_xor(values[0].y, rotary_packs_per_half));
    }

    if (lane < rotary_packs_per_half) {
      const unsigned x_pair = lane;
      const unsigned y_pair = rotary_packs_per_half + lane;
      const float2 x = values[0];
      const float2 y = y_from_register;

      const float2 c = packed_to_float2<cache_t>(packed_cos[lane]);
      const float2 s = packed_to_float2<cache_t>(packed_sin[lane]);
      const float2 x_result =
          make_float2(x.x * c.x - y.x * s.x, x.y * c.y - y.y * s.y);
      const float2 y_result =
          make_float2(y.x * c.x + x.x * s.x, y.y * c.y + x.y * s.y);
      packed_out[x_pair] = float2_to_packed<scalar_t>(x_result);
      packed_out[y_pair] = float2_to_packed<scalar_t>(y_result);
    }

    // Each lane only writes its own register packs when they are outside the
    // rotary prefix. No element in [0, rotary_dim) is revisited.
    const int first_non_rotary_pack = rotary_dim / 2;
    if (lane >= first_non_rotary_pack) {
      packed_out[lane] = float2_to_packed<scalar_t>(values[0]);
    }
    if constexpr (UNROLL == 2) {
      const int second_pair = kWarpSize + lane;
      if (second_pair >= first_non_rotary_pack) {
        packed_out[second_pair] = float2_to_packed<scalar_t>(values[1]);
      }
    }

    if (is_q) {
      scalar_t* gate =
          gate_out + (static_cast<int64_t>(token) * num_q_heads +
                      local_head) *
                         HEAD_DIM;
      const scalar_pack_t* packed_gate_in =
          reinterpret_cast<const scalar_pack_t*>(row + HEAD_DIM);
      scalar_pack_t* packed_gate_out =
          reinterpret_cast<scalar_pack_t*>(gate);
#pragma unroll
      for (int i = 0; i < UNROLL; ++i) {
        const int pair = i * kWarpSize + lane;
        packed_gate_out[pair] = packed_gate_in[pair];
      }
    }
  }
}

template <typename scalar_t, typename cache_t, typename position_t>
void launch_fused_qk_rmsnorm_rope_gate(
    const scalar_t* q_gate, const scalar_t* k, const scalar_t* q_weight,
    const scalar_t* k_weight, const cache_t* cos_sin_cache,
    const position_t* positions, scalar_t* q_out, scalar_t* k_out,
    scalar_t* gate_out, int64_t q_gate_stride, int64_t k_stride,
    int64_t cache_stride, int num_tokens, int num_q_heads, int num_kv_heads,
    int head_dim, int rotary_dim, float epsilon, cudaStream_t stream) {
  const int total_heads = num_q_heads + num_kv_heads;
  const int warps = std::min(total_heads, kMaxWarpsPerBlock);
  const dim3 grid(num_tokens);
  const dim3 block(warps * kWarpSize);

#define VLLM_ILU_LAUNCH_FUSED_QK(HEAD_DIM)                                  \
  fused_qk_rmsnorm_rope_gate_kernel<scalar_t, cache_t, position_t, HEAD_DIM> \
      <<<grid, block, 0, stream>>>(                                          \
          q_gate, k, q_weight, k_weight, cos_sin_cache, positions, q_out,    \
          k_out, gate_out, q_gate_stride, k_stride, cache_stride,            \
          num_q_heads, num_kv_heads, rotary_dim, epsilon)

  if (head_dim == 128) {
    VLLM_ILU_LAUNCH_FUSED_QK(128);
  } else {
    VLLM_ILU_LAUNCH_FUSED_QK(256);
  }
#undef VLLM_ILU_LAUNCH_FUSED_QK
}

template <typename scalar_t, int HEAD_DIM>
__global__ void fused_qk_rmsnorm_gate_kernel(
    const scalar_t* __restrict__ q_gate, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight, scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out, scalar_t* __restrict__ gate_out,
    int64_t q_gate_stride, int64_t k_stride, int num_q_heads,
    int num_kv_heads, float epsilon) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  const int total_heads = num_q_heads + num_kv_heads;
  constexpr int UNROLL = HEAD_DIM / 128;
  using scalar_pack_t = typename TypeTraits<scalar_t, 2>::type;

  for (int head = warp; head < total_heads; head += num_warps) {
    const bool is_q = head < num_q_heads;
    const int local_head = is_q ? head : head - num_q_heads;
    const scalar_t* row =
        is_q ? q_gate + static_cast<int64_t>(token) * q_gate_stride +
                         static_cast<int64_t>(local_head) * 2 * HEAD_DIM
             : k + static_cast<int64_t>(token) * k_stride +
                       static_cast<int64_t>(local_head) * HEAD_DIM;
    const scalar_t* weight = is_q ? q_weight : k_weight;
    scalar_t* out =
        is_q ? q_out + (static_cast<int64_t>(token) * num_q_heads +
                        local_head) *
                           HEAD_DIM
             : k_out + (static_cast<int64_t>(token) * num_kv_heads +
                        local_head) *
                           HEAD_DIM;
    const scalar_pack_t* packed_row =
        reinterpret_cast<const scalar_pack_t*>(row);
    const scalar_pack_t* packed_weight =
        reinterpret_cast<const scalar_pack_t*>(weight);
    scalar_pack_t* packed_out = reinterpret_cast<scalar_pack_t*>(out);

    float2 values[UNROLL];
    float square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const int pair = i * kWarpSize + lane;
      values[i] = packed_to_float2<scalar_t>(packed_row[pair]);
      square_sum +=
          values[i].x * values[i].x + values[i].y * values[i].y;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      square_sum += __shfl_xor(square_sum, offset);
    }
    const float inv_rms =
        rsqrtf(square_sum / static_cast<float>(HEAD_DIM) + epsilon);

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      const int pair = i * kWarpSize + lane;
      const float2 weights =
          packed_to_float2<scalar_t>(packed_weight[pair]);
      values[i].x *= inv_rms * (weights.x + 1.0f);
      values[i].y *= inv_rms * (weights.y + 1.0f);
      packed_out[pair] = float2_to_packed<scalar_t>(values[i]);
    }

    if (is_q) {
      const scalar_pack_t* packed_gate_in =
          reinterpret_cast<const scalar_pack_t*>(row + HEAD_DIM);
      scalar_pack_t* packed_gate_out = reinterpret_cast<scalar_pack_t*>(
          gate_out + (static_cast<int64_t>(token) * num_q_heads +
                      local_head) *
                         HEAD_DIM);
#pragma unroll
      for (int i = 0; i < UNROLL; ++i) {
        const int pair = i * kWarpSize + lane;
        packed_gate_out[pair] = packed_gate_in[pair];
      }
    }
  }
}

template <typename scalar_t>
void launch_fused_qk_rmsnorm_gate(
    const scalar_t* q_gate, const scalar_t* k, const scalar_t* q_weight,
    const scalar_t* k_weight, scalar_t* q_out, scalar_t* k_out,
    scalar_t* gate_out, int64_t q_gate_stride, int64_t k_stride,
    int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
    float epsilon, cudaStream_t stream) {
  const int total_heads = num_q_heads + num_kv_heads;
  const int warps = std::min(total_heads, kMaxWarpsPerBlock);
  const dim3 grid(num_tokens);
  const dim3 block(warps * kWarpSize);

#define VLLM_ILU_LAUNCH_USED_QK(HEAD_DIM)                            \
  fused_qk_rmsnorm_gate_kernel<scalar_t, HEAD_DIM>                    \
      <<<grid, block, 0, stream>>>(                                  \
          q_gate, k, q_weight, k_weight, q_out, k_out, gate_out,      \
          q_gate_stride, k_stride, num_q_heads, num_kv_heads, epsilon)

  if (head_dim == 128) {
    VLLM_ILU_LAUNCH_USED_QK(128);
  } else {
    VLLM_ILU_LAUNCH_USED_QK(256);
  }
#undef VLLM_ILU_LAUNCH_USED_QK
}

void check_cuda(const torch::Tensor& tensor, const char* name,
                const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name,
              " must be on the same device as q_gate");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_rope_gate(
    torch::Tensor& q_gate, torch::Tensor& k, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
    torch::Tensor& positions, double epsilon, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim) {
  TORCH_CHECK(q_gate.is_cuda(), "q_gate must be a CUDA tensor");
  const auto device = q_gate.device();
  check_cuda(k, "k", device);
  check_cuda(q_weight, "q_weight", device);
  check_cuda(k_weight, "k_weight", device);
  check_cuda(cos_sin_cache, "cos_sin_cache", device);
  check_cuda(positions, "positions", device);

  TORCH_CHECK(q_gate.dim() == 2, "q_gate must be a 2D tensor");
  TORCH_CHECK(k.dim() == 2, "k must be a 2D tensor");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1,
              "q_weight and k_weight must be 1D tensors");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "cos_sin_cache must be a 2D tensor");
  TORCH_CHECK(positions.dim() == 1, "positions must be a 1D tensor");
  TORCH_CHECK(head_dim == 128 || head_dim == 256,
              "head_dim must be 128 or 256, got ", head_dim);
  TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0,
              "num_q_heads and num_kv_heads must be positive");
  TORCH_CHECK(rotary_dim >= 4 && rotary_dim <= head_dim &&
                  (rotary_dim & (rotary_dim - 1)) == 0,
              "rotary_dim must be a power of 2 in [4, head_dim]");

  const int64_t num_tokens = q_gate.size(0);
  TORCH_CHECK(k.size(0) == num_tokens && positions.numel() == num_tokens,
              "q_gate, k, and positions must have the same token count");
  TORCH_CHECK(q_gate.size(1) == num_q_heads * 2 * head_dim,
              "q_gate last dimension does not match num_q_heads and head_dim");
  TORCH_CHECK(k.size(1) == num_kv_heads * head_dim,
              "k last dimension does not match num_kv_heads and head_dim");
  TORCH_CHECK(q_weight.numel() == head_dim &&
                  k_weight.numel() == head_dim,
              "q_weight and k_weight must contain head_dim elements");
  TORCH_CHECK(cos_sin_cache.size(1) == rotary_dim,
              "cos_sin_cache last dimension must equal rotary_dim");

  TORCH_CHECK(q_gate.stride(1) == 1 && k.stride(1) == 1,
              "q_gate and k must be contiguous in the last dimension");
  TORCH_CHECK(cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be contiguous in the last dimension");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(),
              "q_weight and k_weight must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(q_gate.stride(0) % 2 == 0 && k.stride(0) % 2 == 0,
              "q_gate and k token strides must be even for packed loads");
  TORCH_CHECK(cos_sin_cache.stride(0) % 2 == 0,
              "cos_sin_cache token stride must be even for packed loads");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q_gate.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(q_weight.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k_weight.data_ptr()) % 4 == 0,
              "q_gate, k, q_weight, and k_weight must be 4-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr()) % 8 == 0,
              "cos_sin_cache must be 8-byte aligned");

  const auto input_type = q_gate.scalar_type();
  TORCH_CHECK(input_type == at::kHalf || input_type == at::kBFloat16,
              "q_gate only supports float16 and bfloat16");
  TORCH_CHECK(k.scalar_type() == input_type &&
                  q_weight.scalar_type() == input_type &&
                  k_weight.scalar_type() == input_type,
              "q_gate, k, q_weight, and k_weight must share a dtype");
  const auto cache_type = cos_sin_cache.scalar_type();
  TORCH_CHECK(cache_type == at::kFloat || cache_type == at::kHalf ||
                  cache_type == at::kBFloat16,
              "cos_sin_cache only supports float32, float16, and bfloat16");
  const auto position_type = positions.scalar_type();
  TORCH_CHECK(position_type == at::kInt || position_type == at::kLong,
              "positions only supports int32 and int64");
  TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(),
              "token count exceeds the CUDA grid limit");

  auto q_out = torch::empty(
      {num_tokens, num_q_heads * head_dim},
      q_gate.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, num_kv_heads * head_dim},
      k.options().memory_format(torch::MemoryFormat::Contiguous));
  auto gate_out = torch::empty_like(
      q_out, q_out.options(), torch::MemoryFormat::Contiguous);
  if (num_tokens == 0) return {q_out, k_out, gate_out};

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define VLLM_ILU_DISPATCH_POSITION(ST, CT)                                  \
  do {                                                                      \
    if (position_type == at::kInt) {                                        \
      launch_fused_qk_rmsnorm_rope_gate<ST, CT, int32_t>(                   \
          reinterpret_cast<const ST*>(q_gate.data_ptr()),                   \
          reinterpret_cast<const ST*>(k.data_ptr()),                        \
          reinterpret_cast<const ST*>(q_weight.data_ptr()),                 \
          reinterpret_cast<const ST*>(k_weight.data_ptr()),                 \
          reinterpret_cast<const CT*>(cos_sin_cache.data_ptr()),            \
          positions.data_ptr<int32_t>(),                                    \
          reinterpret_cast<ST*>(q_out.data_ptr()),                          \
          reinterpret_cast<ST*>(k_out.data_ptr()),                          \
          reinterpret_cast<ST*>(gate_out.data_ptr()), q_gate.stride(0),     \
          k.stride(0), cos_sin_cache.stride(0),                             \
          static_cast<int>(num_tokens), static_cast<int>(num_q_heads),      \
          static_cast<int>(num_kv_heads), static_cast<int>(head_dim),       \
          static_cast<int>(rotary_dim), static_cast<float>(epsilon),        \
          stream);                                                          \
    } else {                                                                \
      launch_fused_qk_rmsnorm_rope_gate<ST, CT, int64_t>(                   \
          reinterpret_cast<const ST*>(q_gate.data_ptr()),                   \
          reinterpret_cast<const ST*>(k.data_ptr()),                        \
          reinterpret_cast<const ST*>(q_weight.data_ptr()),                 \
          reinterpret_cast<const ST*>(k_weight.data_ptr()),                 \
          reinterpret_cast<const CT*>(cos_sin_cache.data_ptr()),            \
          positions.data_ptr<int64_t>(),                                    \
          reinterpret_cast<ST*>(q_out.data_ptr()),                          \
          reinterpret_cast<ST*>(k_out.data_ptr()),                          \
          reinterpret_cast<ST*>(gate_out.data_ptr()), q_gate.stride(0),     \
          k.stride(0), cos_sin_cache.stride(0),                             \
          static_cast<int>(num_tokens), static_cast<int>(num_q_heads),      \
          static_cast<int>(num_kv_heads), static_cast<int>(head_dim),       \
          static_cast<int>(rotary_dim), static_cast<float>(epsilon),        \
          stream);                                                          \
    }                                                                       \
  } while (0)

#define VLLM_ILU_DISPATCH_CACHE(ST)                         \
  do {                                                      \
    if (cache_type == at::kFloat) {                         \
      VLLM_ILU_DISPATCH_POSITION(ST, float);                \
    } else if (cache_type == at::kHalf) {                   \
      VLLM_ILU_DISPATCH_POSITION(ST, half);                 \
    } else {                                                \
      VLLM_ILU_DISPATCH_POSITION(ST, __nv_bfloat16);        \
    }                                                       \
  } while (0)

  if (input_type == at::kHalf) {
    VLLM_ILU_DISPATCH_CACHE(half);
  } else {
    VLLM_ILU_DISPATCH_CACHE(__nv_bfloat16);
  }

#undef VLLM_ILU_DISPATCH_CACHE
#undef VLLM_ILU_DISPATCH_POSITION

  return {q_out, k_out, gate_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_rope_gate_meta(
    torch::Tensor& q_gate, torch::Tensor& k, torch::Tensor& /*q_weight*/,
    torch::Tensor& /*k_weight*/, torch::Tensor& /*cos_sin_cache*/,
    torch::Tensor& /*positions*/, double /*epsilon*/, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t /*rotary_dim*/) {
  const int64_t num_tokens = q_gate.size(0);
  auto q_out = torch::empty(
      {num_tokens, num_q_heads * head_dim},
      q_gate.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, num_kv_heads * head_dim},
      k.options().memory_format(torch::MemoryFormat::Contiguous));
  auto gate_out = torch::empty_like(
      q_out, q_out.options(), torch::MemoryFormat::Contiguous);
  return {q_out, k_out, gate_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_gate(
    torch::Tensor& q_gate, torch::Tensor& k, torch::Tensor& q_weight,
    torch::Tensor& k_weight, double epsilon, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim) {
  TORCH_CHECK(q_gate.is_cuda(), "q_gate must be a CUDA tensor");
  const auto device = q_gate.device();
  check_cuda(k, "k", device);
  check_cuda(q_weight, "q_weight", device);
  check_cuda(k_weight, "k_weight", device);

  TORCH_CHECK(q_gate.dim() == 2 && k.dim() == 2,
              "q_gate and k must be 2D tensors");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1,
              "q_weight and k_weight must be 1D tensors");
  TORCH_CHECK(head_dim == 128 || head_dim == 256,
              "head_dim must be 128 or 256, got ", head_dim);
  TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0,
              "num_q_heads and num_kv_heads must be positive");

  const int64_t num_tokens = q_gate.size(0);
  TORCH_CHECK(k.size(0) == num_tokens,
              "q_gate and k must have the same token count");
  TORCH_CHECK(q_gate.size(1) == num_q_heads * 2 * head_dim,
              "q_gate last dimension does not match num_q_heads and head_dim");
  TORCH_CHECK(k.size(1) == num_kv_heads * head_dim,
              "k last dimension must match num_kv_heads and head_dim");
  TORCH_CHECK(q_weight.numel() == head_dim &&
                  k_weight.numel() == head_dim,
              "q_weight and k_weight must contain head_dim elements");
  TORCH_CHECK(q_gate.stride(1) == 1 && k.stride(1) == 1,
              "q_gate and k must be contiguous in the last dimension");
  TORCH_CHECK(q_gate.stride(0) % 2 == 0 && k.stride(0) % 2 == 0,
              "q_gate and k token strides must be even");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(),
              "q_weight and k_weight must be contiguous");

  const auto input_type = q_gate.scalar_type();
  TORCH_CHECK(input_type == at::kHalf || input_type == at::kBFloat16,
              "q_gate only supports float16 and bfloat16");
  TORCH_CHECK(k.scalar_type() == input_type &&
                  q_weight.scalar_type() == input_type &&
                  k_weight.scalar_type() == input_type,
              "q_gate, k, q_weight, and k_weight must share a dtype");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q_gate.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(q_weight.data_ptr()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k_weight.data_ptr()) % 4 == 0,
              "q_gate, k, q_weight, and k_weight must be 4-byte aligned");
  TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(),
              "token count exceeds the CUDA grid limit");

  auto q_out = torch::empty(
      {num_tokens, num_q_heads * head_dim},
      q_gate.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, num_kv_heads * head_dim},
      k.options().memory_format(torch::MemoryFormat::Contiguous));
  auto gate_out = torch::empty_like(
      q_out, q_out.options(), torch::MemoryFormat::Contiguous);
  if (num_tokens == 0) return {q_out, k_out, gate_out};

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define VLLM_ILU_LAUNCH_USED_QK_HOST(ST)                                  \
  launch_fused_qk_rmsnorm_gate<ST>(                                       \
      reinterpret_cast<const ST*>(q_gate.data_ptr()),                    \
      reinterpret_cast<const ST*>(k.data_ptr()),                         \
      reinterpret_cast<const ST*>(q_weight.data_ptr()),                  \
      reinterpret_cast<const ST*>(k_weight.data_ptr()),                  \
      reinterpret_cast<ST*>(q_out.data_ptr()),                           \
      reinterpret_cast<ST*>(k_out.data_ptr()),                           \
      reinterpret_cast<ST*>(gate_out.data_ptr()), q_gate.stride(0),      \
      k.stride(0), static_cast<int>(num_tokens),                         \
      static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),     \
      static_cast<int>(head_dim), static_cast<float>(epsilon), stream)

  if (input_type == at::kHalf) {
    VLLM_ILU_LAUNCH_USED_QK_HOST(half);
  } else {
    VLLM_ILU_LAUNCH_USED_QK_HOST(__nv_bfloat16);
  }
#undef VLLM_ILU_LAUNCH_USED_QK_HOST

  return {q_out, k_out, gate_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qk_rmsnorm_gate_meta(
    torch::Tensor& q_gate, torch::Tensor& k, torch::Tensor& /*q_weight*/,
    torch::Tensor& /*k_weight*/, double /*epsilon*/, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim) {
  const int64_t num_tokens = q_gate.size(0);
  auto q_out = torch::empty(
      {num_tokens, num_q_heads * head_dim},
      q_gate.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, num_kv_heads * head_dim},
      k.options().memory_format(torch::MemoryFormat::Contiguous));
  auto gate_out = torch::empty_like(
      q_out, q_out.options(), torch::MemoryFormat::Contiguous);
  return {q_out, k_out, gate_out};
}

}  // namespace vllm_iluvatar
