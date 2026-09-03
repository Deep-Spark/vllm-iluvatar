// SPDX-License-Identifier: Apache-2.0
//
// Two-stage MiniMax QK-Norm kernels for TP>1:
//   minimax_qk_var        -- per-token mean-of-squares for q and k segments
//   minimax_qk_rms_apply  -- normalize with all-reduced variance + per-channel weight
//
// Portable CUDA only (no corex30-only intrinsics). Layout inspired by
// ixformer fused_rms_norm_qk (one block per row), with grid.y == 2 for the
// q / k segments of a packed qkv row.

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

// One block processes one (token, segment) pair.
//   blockIdx.x = token id
//   blockIdx.y = 0 -> q segment, 1 -> k segment
template <typename scalar_t, int PACKED>
__global__ void minimax_qk_var_kernel(
    const scalar_t* __restrict__ qkv, float* __restrict__ qk_var,
    int64_t row_stride, int q_size, int kv_size) {
  using pack_t = typename TypeTraits<scalar_t, PACKED>::type;
  constexpr int VEC = sizeof(pack_t) / sizeof(scalar_t);
  static_assert(PACKED == VEC && (PACKED == 1 || PACKED == 2));

  const int token = blockIdx.x;
  const bool is_q = blockIdx.y == 0;
  const int seg_size = is_q ? q_size : kv_size;
  const int seg_offset = is_q ? 0 : q_size;

  const scalar_t* row =
      qkv + static_cast<int64_t>(token) * row_stride + seg_offset;
  const pack_t* packed = reinterpret_cast<const pack_t*>(row);
  const int pack_n = seg_size / PACKED;

  float x2_sum = 0.0f;
  for (int index = threadIdx.x; index < pack_n; index += blockDim.x) {
    pack_t value = packed[index];
    if constexpr (PACKED == 2) {
      float2 v = to_float2<scalar_t>(value);
      x2_sum += v.x * v.x + v.y * v.y;
    } else {
      float v = to_float(value);
      x2_sum += v * v;
    }
  }

  __shared__ float smem[kMaxThreadsPerBlock / warpSize];
  x2_sum = block_reduce_sum(x2_sum, smem);
  if (threadIdx.x == 0) {
    qk_var[token * 2 + (is_q ? 0 : 1)] =
        x2_sum / static_cast<float>(seg_size);
  }
}

template <typename scalar_t, int PACKED>
__global__ void minimax_qk_rms_apply_kernel(
    const scalar_t* __restrict__ qkv, const float* __restrict__ qk_var,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight, scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out, int64_t row_stride, int q_size, int kv_size,
    int tp_world, float epsilon) {
  using pack_t = typename TypeTraits<scalar_t, PACKED>::type;
  constexpr int VEC = sizeof(pack_t) / sizeof(scalar_t);
  static_assert(PACKED == VEC && (PACKED == 1 || PACKED == 2));

  const int token = blockIdx.x;
  const bool is_q = blockIdx.y == 0;
  const int seg_size = is_q ? q_size : kv_size;
  const int seg_offset = is_q ? 0 : q_size;

  const scalar_t* row =
      qkv + static_cast<int64_t>(token) * row_stride + seg_offset;
  const scalar_t* weight = is_q ? q_weight : k_weight;
  scalar_t* out =
      is_q ? q_out + static_cast<int64_t>(token) * q_size
           : k_out + static_cast<int64_t>(token) * kv_size;

  const pack_t* packed_in = reinterpret_cast<const pack_t*>(row);
  const pack_t* packed_w = reinterpret_cast<const pack_t*>(weight);
  pack_t* packed_out = reinterpret_cast<pack_t*>(out);
  const int pack_n = seg_size / PACKED;

  // All-reduced qk_var is the sum of per-shard means; fold /tp_world into rsqrt.
  const float inv_rms = rsqrtf(qk_var[token * 2 + (is_q ? 0 : 1)] /
                                   static_cast<float>(tp_world) +
                               epsilon);

  for (int index = threadIdx.x; index < pack_n; index += blockDim.x) {
    if constexpr (PACKED == 2) {
      float2 x = to_float2<scalar_t>(packed_in[index]);
      float2 w = to_float2<scalar_t>(packed_w[index]);
      packed_out[index] = from_float2<scalar_t>(
          make_float2(x.x * inv_rms * w.x, x.y * inv_rms * w.y));
    } else {
      float x = to_float(packed_in[index]);
      float w = to_float(packed_w[index]);
      packed_out[index] = from_float<scalar_t>(x * inv_rms * w);
    }
  }
}

unsigned choose_threads(unsigned hidden_pack) {
  // warpSize is a compile-time constant on Iluvatar (64); never hard-code
  // 32/64. Round hidden_pack up to a multiple of warpSize and clamp.
  unsigned threads =
      ((hidden_pack + static_cast<unsigned>(warpSize) - 1u) /
       static_cast<unsigned>(warpSize)) *
      static_cast<unsigned>(warpSize);
  if (threads < static_cast<unsigned>(warpSize)) {
    threads = static_cast<unsigned>(warpSize);
  }
  if (threads > static_cast<unsigned>(kMaxThreadsPerBlock)) {
    threads = static_cast<unsigned>(kMaxThreadsPerBlock);
  }
  return threads;
}

bool can_pack2(int64_t q_size, int64_t kv_size, int64_t row_stride,
               const void* qkv_ptr) {
  return ((q_size | kv_size | row_stride) & 1) == 0 &&
         (reinterpret_cast<uintptr_t>(qkv_ptr) % 4 == 0);
}

void check_qkv_common(const torch::Tensor& qkv, int64_t q_size,
                      int64_t kv_size) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be a 2D tensor [T, hidden]");
  TORCH_CHECK(qkv.stride(1) == 1, "qkv must be contiguous in the last dim");
  TORCH_CHECK(q_size > 0 && kv_size > 0,
              "q_size and kv_size must be positive");
  TORCH_CHECK(qkv.size(1) >= q_size + 2 * kv_size,
              "qkv.size(1) must be >= q_size + 2*kv_size, got ", qkv.size(1),
              " vs q_size=", q_size, " kv_size=", kv_size);
  const auto dtype = qkv.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "qkv only supports float16 and bfloat16");
  TORCH_CHECK(qkv.size(0) <= std::numeric_limits<int>::max(),
              "token count exceeds the CUDA grid limit");
}

}  // namespace

torch::Tensor minimax_qk_var(torch::Tensor& qkv, int64_t q_size,
                             int64_t kv_size) {
  check_qkv_common(qkv, q_size, kv_size);

  const int64_t num_tokens = qkv.size(0);
  auto qk_var = torch::empty(
      {num_tokens, 2},
      qkv.options().dtype(torch::kFloat32).memory_format(
          torch::MemoryFormat::Contiguous));
  if (num_tokens == 0) {
    return qk_var;
  }

  const at::cuda::OptionalCUDAGuard device_guard(qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t row_stride = qkv.stride(0);
  const unsigned hidden_max =
      static_cast<unsigned>(std::max(q_size, kv_size));
  const bool pack2 =
      can_pack2(q_size, kv_size, row_stride, qkv.data_ptr());
  const unsigned packed = pack2 ? 2u : 1u;
  const unsigned threads = choose_threads(hidden_max / packed);
  const dim3 grid(static_cast<unsigned>(num_tokens), 2);
  const dim3 block(threads);

#define VLLM_ILU_LAUNCH_VAR(ST, PACK)                                       \
  minimax_qk_var_kernel<ST, PACK><<<grid, block, 0, stream>>>(               \
      reinterpret_cast<const ST*>(qkv.data_ptr()),                          \
      qk_var.data_ptr<float>(), row_stride, static_cast<int>(q_size),       \
      static_cast<int>(kv_size))

  if (qkv.scalar_type() == at::kHalf) {
    if (pack2) {
      VLLM_ILU_LAUNCH_VAR(half, 2);
    } else {
      VLLM_ILU_LAUNCH_VAR(half, 1);
    }
  } else {
    if (pack2) {
      VLLM_ILU_LAUNCH_VAR(__nv_bfloat16, 2);
    } else {
      VLLM_ILU_LAUNCH_VAR(__nv_bfloat16, 1);
    }
  }
#undef VLLM_ILU_LAUNCH_VAR

  return qk_var;
}

torch::Tensor minimax_qk_var_meta(torch::Tensor& qkv, int64_t /*q_size*/,
                                  int64_t /*kv_size*/) {
  return torch::empty(
      {qkv.size(0), 2},
      qkv.options().dtype(torch::kFloat32).memory_format(
          torch::MemoryFormat::Contiguous));
}

std::tuple<torch::Tensor, torch::Tensor> minimax_qk_rms_apply(
    torch::Tensor& qkv, torch::Tensor& qk_var, torch::Tensor& q_weight,
    torch::Tensor& k_weight, int64_t q_size, int64_t kv_size,
    int64_t tp_world, double epsilon) {
  check_qkv_common(qkv, q_size, kv_size);

  TORCH_CHECK(qk_var.is_cuda() && qk_var.device() == qkv.device(),
              "qk_var must be a CUDA tensor on the same device as qkv");
  TORCH_CHECK(qk_var.scalar_type() == at::kFloat,
              "qk_var must be float32");
  TORCH_CHECK(qk_var.dim() == 2 && qk_var.size(0) == qkv.size(0) &&
                  qk_var.size(1) == 2,
              "qk_var must have shape [T, 2]");
  TORCH_CHECK(qk_var.is_contiguous(), "qk_var must be contiguous");

  TORCH_CHECK(q_weight.is_cuda() && q_weight.device() == qkv.device(),
              "q_weight must be a CUDA tensor on the same device as qkv");
  TORCH_CHECK(k_weight.is_cuda() && k_weight.device() == qkv.device(),
              "k_weight must be a CUDA tensor on the same device as qkv");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1,
              "q_weight and k_weight must be 1D");
  TORCH_CHECK(q_weight.numel() == q_size && k_weight.numel() == kv_size,
              "q_weight.numel() must equal q_size and k_weight.numel() "
              "must equal kv_size");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(),
              "q_weight and k_weight must be contiguous");
  TORCH_CHECK(q_weight.scalar_type() == qkv.scalar_type() &&
                  k_weight.scalar_type() == qkv.scalar_type(),
              "q_weight and k_weight must share qkv dtype");

  TORCH_CHECK(tp_world >= 1, "tp_world must be >= 1");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be > 0");

  const int64_t num_tokens = qkv.size(0);
  auto q_out = torch::empty(
      {num_tokens, q_size},
      qkv.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, kv_size},
      qkv.options().memory_format(torch::MemoryFormat::Contiguous));
  if (num_tokens == 0) {
    return {q_out, k_out};
  }

  // Packed weight loads require 4-byte alignment when PACKED=2.
  const bool weight_aligned =
      reinterpret_cast<uintptr_t>(q_weight.data_ptr()) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(k_weight.data_ptr()) % 4 == 0;

  const at::cuda::OptionalCUDAGuard device_guard(qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t row_stride = qkv.stride(0);
  const unsigned hidden_max =
      static_cast<unsigned>(std::max(q_size, kv_size));
  const bool pack2 = can_pack2(q_size, kv_size, row_stride, qkv.data_ptr()) &&
                     weight_aligned;
  const unsigned packed = pack2 ? 2u : 1u;
  const unsigned threads = choose_threads(hidden_max / packed);
  const dim3 grid(static_cast<unsigned>(num_tokens), 2);
  const dim3 block(threads);

#define VLLM_ILU_LAUNCH_APPLY(ST, PACK)                                     \
  minimax_qk_rms_apply_kernel<ST, PACK><<<grid, block, 0, stream>>>(        \
      reinterpret_cast<const ST*>(qkv.data_ptr()),                          \
      qk_var.data_ptr<float>(),                                             \
      reinterpret_cast<const ST*>(q_weight.data_ptr()),                     \
      reinterpret_cast<const ST*>(k_weight.data_ptr()),                     \
      reinterpret_cast<ST*>(q_out.data_ptr()),                              \
      reinterpret_cast<ST*>(k_out.data_ptr()), row_stride,                   \
      static_cast<int>(q_size), static_cast<int>(kv_size),                  \
      static_cast<int>(tp_world), static_cast<float>(epsilon))

  if (qkv.scalar_type() == at::kHalf) {
    if (pack2) {
      VLLM_ILU_LAUNCH_APPLY(half, 2);
    } else {
      VLLM_ILU_LAUNCH_APPLY(half, 1);
    }
  } else {
    if (pack2) {
      VLLM_ILU_LAUNCH_APPLY(__nv_bfloat16, 2);
    } else {
      VLLM_ILU_LAUNCH_APPLY(__nv_bfloat16, 1);
    }
  }
#undef VLLM_ILU_LAUNCH_APPLY

  return {q_out, k_out};
}

std::tuple<torch::Tensor, torch::Tensor> minimax_qk_rms_apply_meta(
    torch::Tensor& qkv, torch::Tensor& /*qk_var*/,
    torch::Tensor& /*q_weight*/, torch::Tensor& /*k_weight*/, int64_t q_size,
    int64_t kv_size, int64_t /*tp_world*/, double /*epsilon*/) {
  const int64_t num_tokens = qkv.size(0);
  auto q_out = torch::empty(
      {num_tokens, q_size},
      qkv.options().memory_format(torch::MemoryFormat::Contiguous));
  auto k_out = torch::empty(
      {num_tokens, kv_size},
      qkv.options().memory_format(torch::MemoryFormat::Contiguous));
  return {q_out, k_out};
}

}  // namespace vllm_iluvatar
