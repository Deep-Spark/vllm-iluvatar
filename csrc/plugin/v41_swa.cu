// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 sliding-window insert:
//   v41_qkv_rope_insert -- GPT-J RoPE on the query and on the latent KV row,
//                          then publish that row into the plain BF16 window
//                          cache.
//
// The BF16 counterpart of the packed-record insert. ixformer offers
// ``v41_qkv_rope_qdq_insert``, which does the same work but pushes the row
// through a quantize-dequantize grid on the way out; V4.1 on this platform
// stores the row as it comes off the rotation, so the grid is the one thing
// that has to go. V4's ``ds4_fused_qnorm_rope_kv_insert`` is not a substitute
// either: it normalises q per head, where V4.1 has already normalised the lora
// rank ``wq_b`` projects from.
//
// ``slots`` is the framework's own per-token ``slot_mapping`` for the window
// cache -- a global ``block_number * block_size + offset`` -- so the page and
// the row inside it come out of the slot alone. Tokens whose slot is negative
// or past the cache still get q and kv rotated; they just publish nothing.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "ops.h"
#include "utils.cuh"
#include "v41_common.cuh"

namespace vllm_iluvatar {
namespace {

using namespace v41;

constexpr int kRowDim = 512;
constexpr int kRopeDim = 64;
constexpr int kRopeStart = kRowDim - kRopeDim;
// One thread per GPT-J pair, so a whole row is four 64-lane warps and the
// rotation band alone is half of one.
constexpr int kRowThreads = kRowDim / kRopePacked;
constexpr int kRopeThreads = kRopeDim / kRopePacked;

// One block per (token, head). Only the trailing band moves, so the block
// covers the band rather than the row.
__global__ void v41_q_rope_kernel(__nv_bfloat16* __restrict__ q,
                                  int64_t token_stride, int64_t head_stride,
                                  const void* __restrict__ positions,
                                  bool positions_are_int64,
                                  const float* __restrict__ cos_sin_cache,
                                  int64_t cos_sin_stride) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int pair = threadIdx.x;

  const int64_t pos = load_index(positions, positions_are_int64, token);
  __nv_bfloat16* const row = q + token * token_stride + head * head_stride;
  __nv_bfloat162 v =
      *reinterpret_cast<__nv_bfloat162*>(row + kRopeStart + pair * kRopePacked);
  float even = __bfloat162float(v.x);
  float odd = __bfloat162float(v.y);
  rope_pair_at(even, odd, cos_sin_cache + pos * cos_sin_stride, pair, kRopeDim);
  v.x = __float2bfloat16_rn(even);
  v.y = __float2bfloat16_rn(odd);
  *reinterpret_cast<__nv_bfloat162*>(row + kRopeStart + pair * kRopePacked) = v;
}

// One block per token over the full 512-wide latent row: the trailing band is
// rotated in place and the whole row is published.
__global__ void v41_kv_rope_insert_kernel(
    __nv_bfloat16* __restrict__ kv, int64_t kv_stride,
    const void* __restrict__ positions, bool positions_are_int64,
    const void* __restrict__ slot_mapping, bool slots_are_int64,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    __nv_bfloat16* __restrict__ cache, int64_t page_stride, int64_t row_stride,
    int rows_per_page, int64_t num_pages) {
  const int64_t token = blockIdx.x;
  const int d0 = threadIdx.x * kRopePacked;

  __nv_bfloat16* const src = kv + token * kv_stride;
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(src + d0);

  if (d0 >= kRopeStart) {
    const int64_t pos = load_index(positions, positions_are_int64, token);
    float even = __bfloat162float(v.x);
    float odd = __bfloat162float(v.y);
    rope_pair_at(even, odd, cos_sin_cache + pos * cos_sin_stride,
                 (d0 - kRopeStart) / 2, kRopeDim);
    v.x = __float2bfloat16_rn(even);
    v.y = __float2bfloat16_rn(odd);
    *reinterpret_cast<__nv_bfloat162*>(src + d0) = v;
  }

  const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
  if (slot < 0) {
    return;
  }
  const int64_t page = slot / rows_per_page;
  if (page >= num_pages) {
    return;
  }
  const int64_t row = slot % rows_per_page;
  *reinterpret_cast<__nv_bfloat162*>(cache + page * page_stride +
                                     row * row_stride + d0) = v;
}

void check_index_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kLong || t.scalar_type() == at::kInt,
              name, " must be int32 or int64");
}

}  // namespace

void v41_qkv_rope_insert(torch::Tensor& q, torch::Tensor& kv_cache,
                         torch::Tensor& kv, torch::Tensor& slot_mapping,
                         torch::Tensor& positions,
                         torch::Tensor& cos_sin_cache) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 &&
                  q.scalar_type() == at::kBFloat16,
              "q must be a 3D bf16 CUDA tensor [T, heads, 512]");
  TORCH_CHECK(q.size(2) == kRowDim && q.stride(2) == 1,
              "q heads are ", kRowDim, " wide with a contiguous last dim");
  TORCH_CHECK(kv.is_cuda() && kv.dim() == 2 &&
                  kv.scalar_type() == at::kBFloat16,
              "kv must be a 2D bf16 CUDA tensor [T, 512]");
  TORCH_CHECK(kv.size(1) == kRowDim && kv.stride(1) == 1,
              "kv rows are ", kRowDim, " wide with a contiguous last dim");
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dim() == 3 &&
                  kv_cache.scalar_type() == at::kBFloat16,
              "the plain window cache is bf16 [pages, rows, 512]");
  TORCH_CHECK(kv_cache.size(2) == kRowDim && kv_cache.stride(2) == 1,
              "a plain window cache row is ", kRowDim,
              " wide and contiguous");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.scalar_type() == at::kFloat &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be a 2D fp32 CUDA tensor, last dim "
              "contiguous");
  TORCH_CHECK(cos_sin_cache.size(1) >= kRopeDim,
              "cos_sin_cache must be at least ", kRopeDim, " wide");
  check_index_tensor(positions, "positions");
  check_index_tensor(slot_mapping, "slot_mapping");

  const int64_t num_tokens = slot_mapping.numel();
  TORCH_CHECK(num_tokens <= kv.size(0) && num_tokens <= positions.numel(),
              "slot_mapping is longer than kv / positions");
  if (num_tokens == 0) {
    return;
  }

  const bool positions_are_int64 = positions.scalar_type() == at::kLong;
  const bool slots_are_int64 = slot_mapping.scalar_type() == at::kLong;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float* cos_sin_ptr = cos_sin_cache.data_ptr<float>();

  // The DSpark proposer publishes context KV with no query of its own, and
  // hands an empty head axis to say so.
  const int64_t num_heads = q.size(1);
  if (num_heads > 0) {
    TORCH_CHECK(num_tokens <= q.size(0),
                "slot_mapping is longer than q");
    v41_q_rope_kernel<<<dim3(static_cast<unsigned>(num_tokens),
                             static_cast<unsigned>(num_heads)),
                        kRopeThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()), q.stride(0),
        q.stride(1), positions.data_ptr(), positions_are_int64, cos_sin_ptr,
        cos_sin_cache.stride(0));
  }

  v41_kv_rope_insert_kernel<<<static_cast<unsigned>(num_tokens), kRowThreads, 0,
                              stream>>>(
      reinterpret_cast<__nv_bfloat16*>(kv.data_ptr()), kv.stride(0),
      positions.data_ptr(), positions_are_int64, slot_mapping.data_ptr(),
      slots_are_int64, cos_sin_ptr, cos_sin_cache.stride(0),
      reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr()),
      kv_cache.stride(0), kv_cache.stride(1),
      static_cast<int>(kv_cache.size(1)), kv_cache.size(0));
}

}  // namespace vllm_iluvatar
