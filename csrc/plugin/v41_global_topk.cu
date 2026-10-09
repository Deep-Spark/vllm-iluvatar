// SPDX-License-Identifier: Apache-2.0
//
// v41_global_topk_indices_and_lens -- local compressed slots to paged global
// slots, the plugin's copy of ixformer's
// ``ds4_compute_global_topk_indices_and_lens``.
//
// Semantics follow ``ref_ds4_compute_global_topk_indices_and_lens`` in
// ``ixformer/inference/functions/deepseek/decode_attn.py``:
//   * ``block_idx = local // block_size``, ``off = local % block_size``
//   * page 0 is the null block; a slot resolving to it is dropped, not emitted
//   * kept slots compact to the front of the row, ``topk_lens`` counts them
//   * an invalid token gets length 0 and a row of -1 (0 under pad_walk_tail),
//     and its ``token_to_req_indices`` entry is never read
//   * ``positions`` caps the scan at ``(pos + 1) / compress_ratio`` columns
//
// The slot arithmetic is done in 64-bit even though the emitted slot is int32:
// the block table is a physical id into a pool that is now large enough for
// ``phys * block_size`` to be worth computing carefully.
//
// The compaction is a block-wide prefix sum rather than one thread walking
// the row. At ``index_topk = 512`` the serial walk cost 69 us against
// ixformer's 11 us; the scan brings it to 10 us.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <tuple>

#include "ops.h"
#include "v41_common.cuh"

namespace vllm_iluvatar {
namespace {

using namespace v41;

// One thread per column, so the block covers a whole row in one pass at the
// widths this model runs (``index_topk`` is 512). A wider row loops.
constexpr int kGlobalTopkThreads = 512;

// Resolve one column to a global slot, or -1 if it must not be emitted.
// ``*_shift`` is the log2 of the matching divisor, or -1 when it is not a
// power of two; the shift is what keeps a per-column integer division out of
// the inner loop.
__device__ __forceinline__ int32_t resolve_slot(
    const int32_t* __restrict__ in_row, const int32_t* __restrict__ bt_row,
    int k, int eff_topk, int block_size, int block_shift, int block_table_width,
    int64_t slot_limit) {
  if (k >= eff_topk) {
    return -1;
  }
  const int32_t local = in_row[k];
  if (local < 0) {
    return -1;
  }
  const int block_idx =
      block_shift >= 0 ? (local >> block_shift) : (local / block_size);
  const int block_off =
      block_shift >= 0 ? (local & (block_size - 1)) : (local % block_size);
  if (block_idx >= block_table_width) {
    return -1;
  }
  const int64_t page = bt_row[block_idx];
  // Page 0 is the null block: never emitted, never counted.
  if (page <= 0) {
    return -1;
  }
  const int64_t slot = page * static_cast<int64_t>(block_size) + block_off;
  if (slot_limit > 0 && slot >= slot_limit) {
    return -1;
  }
  return static_cast<int32_t>(slot);
}

template <int BLOCK>
__global__ void v41_global_topk_indices_and_lens_kernel(
    const int32_t* __restrict__ topk_indices, int64_t topk_stride,
    int32_t* __restrict__ global_indices, int64_t global_stride,
    int32_t* __restrict__ topk_lens,
    const void* __restrict__ token_to_req, bool reqs_are_int64,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int block_table_width, const bool* __restrict__ is_valid_token,
    const void* __restrict__ positions, bool positions_are_int64,
    bool has_positions, int block_size, int block_shift, int compress_ratio,
    int compress_shift, int topk, int64_t slot_limit, bool pad_walk_tail) {
  constexpr int kWarps = (BLOCK + warpSize - 1) / warpSize;
  __shared__ int warp_kept[kWarps];

  const int64_t token = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  int32_t* const out_row = global_indices + token * global_stride;

  if (!is_valid_token[token]) {
    // A padding row must not read ``token_to_req_indices`` (it may be stale)
    // and must not leave stale slots behind for a tile-wise walk.
    const int32_t fill = pad_walk_tail ? 0 : -1;
    for (int k = tid; k < topk; k += BLOCK) {
      out_row[k] = fill;
    }
    if (tid == 0) {
      topk_lens[token] = 0;
    }
    return;
  }

  const int64_t req = load_index(token_to_req, reqs_are_int64, token);
  int eff_topk = topk;
  if (has_positions) {
    const int64_t pos1 = load_index(positions, positions_are_int64, token) + 1;
    int64_t limit = compress_shift >= 0 ? (pos1 >> compress_shift)
                                        : (pos1 / compress_ratio);
    if (limit < 0) {
      limit = 0;
    }
    if (limit > topk) {
      limit = topk;
    }
    eff_topk = static_cast<int>(limit);
  }

  const int32_t* const bt_row = block_table + req * block_table_stride;
  const int32_t* const in_row = topk_indices + token * topk_stride;

  const int lane = tid % warpSize;
  const int warp_id = tid / warpSize;

  // Compaction is a prefix sum over "did this column survive", so the block
  // can place every kept slot at once instead of one thread walking the row.
  // ``written`` carries the count across tiles of a row wider than the block.
  int written = 0;
  for (int base = 0; base < topk; base += BLOCK) {
    const int k = base + tid;
    const int32_t slot =
        k < topk ? resolve_slot(in_row, bt_row, k, eff_topk, block_size,
                                block_shift, block_table_width, slot_limit)
                 : -1;
    const bool keep = slot >= 0;

    // The flag is 0/1, so the warp's prefix sum is a popcount of its ballot:
    // two instructions instead of a log-depth shuffle chain, which on this
    // platform lowers to shared memory and shows up as LDS bank conflicts.
    const uint64_t ballot = __ballot_sync(~0ULL, keep);
    const int rank_in_warp = __popcll(ballot & ((uint64_t{1} << lane) - 1));
    if (lane == 0) {
      warp_kept[warp_id] = __popcll(ballot);
    }
    __syncthreads();

    // kWarps is 8 at most, so every thread summing every subtotal is cheaper
    // than a second scan, and the loads all broadcast.
    int warp_base = 0;
    int tile_kept = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int subtotal = warp_kept[w];
      warp_base += w < warp_id ? subtotal : 0;
      tile_kept += subtotal;
    }

    if (keep) {
      out_row[written + warp_base + rank_in_warp] = slot;
    }
    written += tile_kept;
    // The next tile overwrites warp_kept, and the tail fill below reads the
    // last slot this tile just placed.
    __syncthreads();
  }

  // A row with nothing kept still gets walked by a tile-aligned reader, so it
  // needs an in-range slot rather than -1; row 0 belongs to the null page,
  // which no valid slot names.
  const int32_t filler =
      pad_walk_tail ? (written > 0 ? out_row[written - 1] : 0) : -1;
  for (int k = written + tid; k < topk; k += BLOCK) {
    out_row[k] = filler;
  }
  if (tid == 0) {
    topk_lens[token] = written;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> v41_global_topk_indices_and_lens(
    torch::Tensor& topk_indices, torch::Tensor& token_to_req_indices,
    torch::Tensor& block_table, int64_t block_size,
    torch::Tensor& is_valid_token,
    const std::optional<torch::Tensor>& positions, int64_t compress_ratio,
    int64_t slot_limit, bool pad_walk_tail) {
  TORCH_CHECK(topk_indices.is_cuda() && topk_indices.dim() == 2 &&
                  topk_indices.scalar_type() == at::kInt &&
                  topk_indices.stride(1) == 1,
              "topk_indices must be a 2D int32 CUDA tensor [T, topk] with a "
              "contiguous last dim");
  TORCH_CHECK(token_to_req_indices.is_cuda() &&
                  token_to_req_indices.is_contiguous() &&
                  (token_to_req_indices.scalar_type() == at::kInt ||
                   token_to_req_indices.scalar_type() == at::kLong),
              "token_to_req_indices must be a contiguous int32/int64 CUDA "
              "tensor");
  TORCH_CHECK(block_table.is_cuda() && block_table.dim() == 2 &&
                  block_table.scalar_type() == at::kInt &&
                  block_table.stride(1) == 1,
              "block_table must be a 2D int32 CUDA tensor with contiguous rows");
  TORCH_CHECK(is_valid_token.is_cuda() && is_valid_token.dim() == 1 &&
                  is_valid_token.is_contiguous() &&
                  is_valid_token.scalar_type() == at::kBool,
              "is_valid_token must be a contiguous 1D bool CUDA tensor");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t num_tokens = topk_indices.size(0);
  const int64_t topk = topk_indices.size(1);
  auto global_indices = torch::empty_like(topk_indices);
  auto topk_lens = torch::empty(
      {num_tokens}, topk_indices.options().dtype(torch::kInt32));
  if (num_tokens == 0 || topk == 0) {
    return {global_indices, topk_lens};
  }
  TORCH_CHECK(is_valid_token.numel() >= num_tokens,
              "is_valid_token is shorter than topk_indices' token count");

  const bool has_positions = positions.has_value();
  if (has_positions) {
    const torch::Tensor& pos = positions.value();
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() &&
                    (pos.scalar_type() == at::kInt ||
                     pos.scalar_type() == at::kLong),
                "positions must be a contiguous int32/int64 CUDA tensor");
    TORCH_CHECK(pos.numel() >= num_tokens,
                "positions is shorter than topk_indices' token count");
    TORCH_CHECK(compress_ratio > 0,
                "compress_ratio must be positive when positions is given");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_GLOBAL_TOPK(BLOCK)                                             \
  v41_global_topk_indices_and_lens_kernel<BLOCK>                              \
      <<<static_cast<unsigned>(num_tokens), (BLOCK), 0, stream>>>(            \
          topk_indices.data_ptr<int32_t>(), topk_indices.stride(0),           \
          global_indices.data_ptr<int32_t>(), global_indices.stride(0),       \
          topk_lens.data_ptr<int32_t>(), token_to_req_indices.data_ptr(),     \
          token_to_req_indices.scalar_type() == at::kLong,                    \
          block_table.data_ptr<int32_t>(), block_table.stride(0),             \
          static_cast<int>(block_table.size(1)),                              \
          is_valid_token.data_ptr<bool>(),                                    \
          has_positions ? positions.value().data_ptr() : nullptr,             \
          has_positions && positions.value().scalar_type() == at::kLong,      \
          has_positions, static_cast<int>(block_size),                        \
          exact_log2(block_size), static_cast<int>(compress_ratio),           \
          exact_log2(compress_ratio), static_cast<int>(topk), slot_limit,     \
          pad_walk_tail)

  // One thread per column where the row fits; past that the kernel loops, so
  // the widest block is also the fallback.
  if (topk <= 128) {
    LAUNCH_GLOBAL_TOPK(128);
  } else if (topk <= 256) {
    LAUNCH_GLOBAL_TOPK(256);
  } else {
    LAUNCH_GLOBAL_TOPK(kGlobalTopkThreads);
  }
#undef LAUNCH_GLOBAL_TOPK
  return {global_indices, topk_lens};
}

}  // namespace vllm_iluvatar
