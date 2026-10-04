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

constexpr int kGlobalTopkThreads = 128;

__global__ void v41_global_topk_indices_and_lens_kernel(
    const int32_t* __restrict__ topk_indices, int64_t topk_stride,
    int32_t* __restrict__ global_indices, int64_t global_stride,
    int32_t* __restrict__ topk_lens,
    const void* __restrict__ token_to_req, bool reqs_are_int64,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int block_table_width, const bool* __restrict__ is_valid_token,
    const void* __restrict__ positions, bool positions_are_int64,
    bool has_positions, int block_size, int compress_ratio, int topk,
    int64_t slot_limit, bool pad_walk_tail) {
  extern __shared__ int32_t staged[];  // topk entries
  const int64_t token = blockIdx.x;
  int32_t* const out_row = global_indices + token * global_stride;

  if (!is_valid_token[token]) {
    // A padding row must not read ``token_to_req_indices`` (it may be stale)
    // and must not leave stale slots behind for a tile-wise walk.
    const int32_t fill = pad_walk_tail ? 0 : -1;
    for (int k = threadIdx.x; k < topk; k += blockDim.x) {
      out_row[k] = fill;
    }
    if (threadIdx.x == 0) {
      topk_lens[token] = 0;
    }
    return;
  }

  const int64_t req = load_index(token_to_req, reqs_are_int64, token);
  int eff_topk = topk;
  if (has_positions) {
    const int64_t pos = load_index(positions, positions_are_int64, token);
    int64_t limit = (pos + 1) / compress_ratio;
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

  for (int k = threadIdx.x; k < topk; k += blockDim.x) {
    int32_t resolved = -1;
    if (k < eff_topk) {
      const int32_t local = in_row[k];
      if (local >= 0) {
        const int block_idx = local / block_size;
        const int block_off = local % block_size;
        if (block_idx < block_table_width) {
          const int64_t page = bt_row[block_idx];
          // Page 0 is the null block: never emitted, never counted.
          if (page > 0) {
            const int64_t slot =
                page * static_cast<int64_t>(block_size) + block_off;
            if (slot_limit <= 0 || slot < slot_limit) {
              resolved = static_cast<int32_t>(slot);
            }
          }
        }
      }
    }
    staged[k] = resolved;
  }
  __syncthreads();

  // Compaction is order-preserving, so the write index never overtakes the
  // read index and the pass can run in place in shared memory.
  //
  // It used to run entirely on thread 0: a serial walk over all ``topk``
  // entries while the other 127 threads idled, which at index_topk=512 was
  // essentially the whole kernel (68 us per call, 2.4 ms/step). A block-wide
  // stable scan does the same compaction in ceil(topk / blockDim.x) tiles.
  __shared__ int32_t scan_buf[kGlobalTopkThreads];
  __shared__ int32_t kept_total;
  if (threadIdx.x == 0) {
    kept_total = 0;
  }
  __syncthreads();

  for (int base = 0; base < topk; base += kGlobalTopkThreads) {
    const int k = base + static_cast<int>(threadIdx.x);
    // Load before any store: the compacted position never exceeds the read
    // position, but a lower-indexed thread must not land on a slot this
    // thread has yet to read.
    const int32_t val = (k < topk) ? staged[k] : -1;
    const int32_t pred = (val >= 0) ? 1 : 0;
    __syncthreads();

    scan_buf[threadIdx.x] = pred;
    __syncthreads();
    for (int off = 1; off < kGlobalTopkThreads; off <<= 1) {
      const int32_t add =
          (threadIdx.x >= static_cast<unsigned>(off)) ? scan_buf[threadIdx.x - off] : 0;
      __syncthreads();
      scan_buf[threadIdx.x] += add;
      __syncthreads();
    }
    const int32_t incl = scan_buf[threadIdx.x];
    if (pred) {
      staged[kept_total + incl - 1] = val;
    }
    __syncthreads();
    if (threadIdx.x == kGlobalTopkThreads - 1) {
      kept_total += incl;
    }
    __syncthreads();
  }

  const int32_t kept = kept_total;
  const int32_t fill = pad_walk_tail ? (kept > 0 ? staged[kept - 1] : 0) : -1;
  __syncthreads();
  if (threadIdx.x == 0) {
    topk_lens[token] = kept;
  }
  for (int k = kept + static_cast<int>(threadIdx.x); k < topk;
       k += kGlobalTopkThreads) {
    staged[k] = fill;
  }
  __syncthreads();

  for (int k = threadIdx.x; k < topk; k += blockDim.x) {
    out_row[k] = staged[k];
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
  const size_t smem = static_cast<size_t>(topk) * sizeof(int32_t);
  v41_global_topk_indices_and_lens_kernel<<<static_cast<unsigned>(num_tokens),
                                            kGlobalTopkThreads, smem, stream>>>(
      topk_indices.data_ptr<int32_t>(), topk_indices.stride(0),
      global_indices.data_ptr<int32_t>(), global_indices.stride(0),
      topk_lens.data_ptr<int32_t>(), token_to_req_indices.data_ptr(),
      token_to_req_indices.scalar_type() == at::kLong,
      block_table.data_ptr<int32_t>(), block_table.stride(0),
      static_cast<int>(block_table.size(1)), is_valid_token.data_ptr<bool>(),
      has_positions ? positions.value().data_ptr() : nullptr,
      has_positions && positions.value().scalar_type() == at::kLong,
      has_positions, static_cast<int>(block_size),
      static_cast<int>(compress_ratio), static_cast<int>(topk), slot_limit,
      pad_walk_tail);
  return {global_indices, topk_lens};
}

}  // namespace vllm_iluvatar
