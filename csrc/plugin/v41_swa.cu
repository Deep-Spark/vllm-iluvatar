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

// Both kernels move bf16 in 16-byte units -- the widest access this data type
// has -- so a thread owns four adjacent GPT-J pairs rather than one. That is
// a quarter of the memory instructions of a pair-per-thread kernel, and these
// kernels are small enough that their cost is instruction count and launch
// shape rather than bandwidth.
constexpr int kPairsPerThread = 4;
constexpr int kElemsPerThread = kPairsPerThread * kRopePacked;  // 8 bf16 = 16 B
// A whole 512-wide row is then one 64-lane warp.
constexpr int kRowThreads = kRowDim / kElemsPerThread;
// The 64-wide rotation band is eight lanes.
constexpr int kRopeLanes = kRopeDim / kElemsPerThread;
// Query rows are rotation band only, so a block batches whole heads to reach
// a full warp; 32 heads is 256 threads.
constexpr int kQRopeHeadsPerBlock = 32;
constexpr int kQRopeThreads = kQRopeHeadsPerBlock * kRopeLanes;

// Rotate the four pairs packed into one 16-byte unit. ``first_pair`` is the
// index of the unit's first pair inside the band.
__device__ __forceinline__ void rope_quad(uint4& packed,
                                          const float* __restrict__ cs_row,
                                          int first_pair) {
  __nv_bfloat162* const v = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
  for (int i = 0; i < kPairsPerThread; ++i) {
    float even = __bfloat162float(v[i].x);
    float odd = __bfloat162float(v[i].y);
    rope_pair_at(even, odd, cs_row, first_pair + i, kRopeDim);
    v[i].x = __float2bfloat16_rn(even);
    v[i].y = __float2bfloat16_rn(odd);
  }
}

// Rotate and publish the token's latent KV row. One warp covers all 512
// channels at 16 bytes a lane.
//
// ``page_shift`` is the log2 of ``rows_per_page``, or -1 when it is not a
// power of two. Splitting the slot with a shift rather than a 64-bit divide
// is what takes this off the integer pipeline: the divide is emulated in
// software and ran once per thread.
__device__ __forceinline__ void kv_rope_insert_row(
    __nv_bfloat16* __restrict__ kv, int64_t kv_stride, int64_t token,
    const float* __restrict__ cs_row, const void* __restrict__ slot_mapping,
    bool slots_are_int64, __nv_bfloat16* __restrict__ cache,
    int64_t page_stride, int64_t row_stride, int rows_per_page, int page_shift,
    int64_t num_pages, int lane) {
  const int d0 = lane * kElemsPerThread;
  __nv_bfloat16* const src = kv + token * kv_stride + d0;
  uint4 packed = *reinterpret_cast<uint4*>(src);

  if (d0 >= kRopeStart) {
    rope_quad(packed, cs_row, (d0 - kRopeStart) / kRopePacked);
    *reinterpret_cast<uint4*>(src) = packed;
  }

  const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
  if (slot < 0) {
    return;
  }
  int64_t page, row;
  split_slot(slot, rows_per_page, page_shift, page, row);
  if (page >= num_pages) {
    return;
  }
  *reinterpret_cast<uint4*>(cache + page * page_stride + row * row_stride +
                            d0) = packed;
}

// The query and the latent row share a token, a position and a cos/sin row,
// and nothing either writes is read by the other, so they are one launch.
// That matters more than what either does: at these shapes a launch costs
// about as much as the work, and this op used to pay for two.
//
// ``blockIdx.y == 0`` is the latent row -- one warp of the block, the rest
// exit -- and the remaining y cover ``kQRopeHeadsPerBlock`` query bands each.
// Only the trailing band of a query head moves, so a block covers bands
// rather than rows: one band per block came to a 32-thread block, which
// leaves half of every 64-lane warp idle.
__global__ void v41_qkv_rope_insert_kernel(
    __nv_bfloat16* __restrict__ q, int64_t q_token_stride,
    int64_t q_head_stride, int num_heads, __nv_bfloat16* __restrict__ kv,
    int64_t kv_stride, const void* __restrict__ positions,
    bool positions_are_int64, const void* __restrict__ slot_mapping,
    bool slots_are_int64, const float* __restrict__ cos_sin_cache,
    int64_t cos_sin_stride, __nv_bfloat16* __restrict__ cache,
    int64_t page_stride, int64_t row_stride, int rows_per_page, int page_shift,
    int64_t num_pages) {
  const int64_t token = blockIdx.x;
  const int64_t pos = load_index(positions, positions_are_int64, token);
  const float* const cs_row = cos_sin_cache + pos * cos_sin_stride;

  if (blockIdx.y == 0) {
    if (threadIdx.x < kRowThreads) {
      kv_rope_insert_row(kv, kv_stride, token, cs_row, slot_mapping,
                         slots_are_int64, cache, page_stride, row_stride,
                         rows_per_page, page_shift, num_pages,
                         static_cast<int>(threadIdx.x));
    }
    return;
  }

  const int64_t head =
      (blockIdx.y - 1) * kQRopeHeadsPerBlock + threadIdx.x / kRopeLanes;
  if (head >= num_heads) {
    return;
  }
  const int lane = threadIdx.x % kRopeLanes;
  __nv_bfloat16* const unit = q + token * q_token_stride +
                              head * q_head_stride + kRopeStart +
                              lane * kElemsPerThread;
  uint4 packed = *reinterpret_cast<uint4*>(unit);
  rope_quad(packed, cs_row, lane * kPairsPerThread);
  *reinterpret_cast<uint4*>(unit) = packed;
}

void check_index_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kLong || t.scalar_type() == at::kInt,
              name, " must be int32 or int64");
}

void check_slot_tensor(const torch::Tensor& t, const char* name) {
  check_index_tensor(t, name);
  TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64");
}

// The 16-byte accesses need every row this kernel addresses to start on a
// 16-byte boundary. A torch allocation is aligned well past that, so this is
// really a check on the strides.
void check_wide_stride(int64_t stride, const char* name) {
  TORCH_CHECK(stride % kElemsPerThread == 0, name,
              " must be a multiple of ", kElemsPerThread,
              " elements for 16-byte access, got ", stride);
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
  check_slot_tensor(slot_mapping, "slot_mapping");

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
    check_wide_stride(q.stride(0), "q.stride(0)");
    check_wide_stride(q.stride(1), "q.stride(1)");
  }
  check_wide_stride(kv.stride(0), "kv.stride(0)");
  check_wide_stride(kv_cache.stride(0), "kv_cache.stride(0)");
  check_wide_stride(kv_cache.stride(1), "kv_cache.stride(1)");

  // y == 0 is the latent row; the rest are query head bands.
  const unsigned head_blocks = static_cast<unsigned>(
      (num_heads + kQRopeHeadsPerBlock - 1) / kQRopeHeadsPerBlock);
  v41_qkv_rope_insert_kernel<<<
      dim3(static_cast<unsigned>(num_tokens), head_blocks + 1), kQRopeThreads,
      0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()), q.stride(0), q.stride(1),
      static_cast<int>(num_heads),
      reinterpret_cast<__nv_bfloat16*>(kv.data_ptr()), kv.stride(0),
      positions.data_ptr(), positions_are_int64, slot_mapping.data_ptr(),
      slots_are_int64, cos_sin_ptr, cos_sin_cache.stride(0),
      reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr()),
      kv_cache.stride(0), kv_cache.stride(1),
      static_cast<int>(kv_cache.size(1)), exact_log2(kv_cache.size(1)),
      kv_cache.size(0));
}

}  // namespace vllm_iluvatar
