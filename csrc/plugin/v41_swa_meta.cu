// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 sliding-window metadata ops:
//   v41_swa_indices_and_lens      -- per-token paged key slots for the window
//   v41_swa_prefill_gather_lens   -- per-request context rows a prefill gathers
//
// These replace ``ixfops.ds4_compute_swa_decode_indices_and_lens`` and
// ``ixfops.ds4_compute_swa_prefill_gather_lens``. The ixformer ops predate SWA
// bounded replay (``CacheConfig.swa_bounded_replay``) and take no per-request
// lower bound, which the replay needs: after a prefix hit the scheduler rewinds
// the request to ``replay_start`` and only the window from there on has been
// written, so anything below it must stay out of both the index rows and the
// gather length. See ``_compute_swa_indices_and_lens_kernel`` and
// ``ComputePrefillMetadataKernel`` in
// ``off/vllm/vllm/v1/attention/backends/mla/sparse_swa.py``.
//
// Apart from that lower bound the index op keeps ixformer's contract rather
// than the Triton one: slots whose page is the null block are dropped and the
// kept ones compacted to the front, so ``swa_lens`` counts what the row really
// holds. With the bound in place the two agree -- a retired page can no longer
// fall inside the window -- but the drop stays as the cheap backstop it was.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "ops.h"
#include "utils.cuh"

namespace vllm_iluvatar {
namespace {

constexpr int kSwaIndexThreads = 128;
constexpr int kGatherLenThreads = 128;

__global__ void v41_swa_indices_and_lens_kernel(
    int32_t* __restrict__ swa_indices, int64_t indices_stride,
    int32_t* __restrict__ swa_lens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ token_to_req_indices,
    const bool* __restrict__ is_valid_token,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    const int32_t* __restrict__ replay_start, int window_size,
    int block_size) {
  extern __shared__ int32_t slots[];  // window_size entries
  const int token = blockIdx.x;

  if (!is_valid_token[token]) {
    // Graph-capture or batch padding. Report an empty window and leave the
    // row as the caller staged it.
    if (threadIdx.x == 0) {
      swa_lens[token] = 0;
    }
    return;
  }

  const int req = token_to_req_indices[token];
  const int query_start = query_start_loc[req];
  const int query_len = query_start_loc[req + 1] - query_start;
  const int pos = seq_lens[req] - query_len + token - query_start;

  int start_pos = pos - window_size + 1;
  if (start_pos < 0) {
    start_pos = 0;
  }
  // SWA bounded replay: the request holds no window KV below this position.
  const int lower_bound = replay_start[req];
  if (start_pos < lower_bound) {
    start_pos = lower_bound;
  }
  const int span = pos + 1 - start_pos;

  for (int off = threadIdx.x; off < window_size; off += blockDim.x) {
    int32_t slot = -1;
    if (off < span) {
      const int p = start_pos + off;
      const int page = block_table[req * block_table_stride + p / block_size];
      // Page 0 is the null block: a position whose page was retired below the
      // window contributes nothing.
      if (page > 0) {
        slot = page * block_size + p % block_size;
      }
    }
    slots[off] = slot;
  }
  __syncthreads();

  // Compact in place -- the write index never overtakes the read index -- so
  // the store back to the row below stays coalesced.
  if (threadIdx.x == 0) {
    int kept = 0;
    for (int off = 0; off < window_size; ++off) {
      const int32_t slot = slots[off];
      if (slot >= 0) {
        slots[kept++] = slot;
      }
    }
    for (int off = kept; off < window_size; ++off) {
      slots[off] = -1;
    }
    swa_lens[token] = kept;
  }
  __syncthreads();

  int32_t* row = swa_indices + token * indices_stride;
  for (int off = threadIdx.x; off < window_size; off += blockDim.x) {
    row[off] = slots[off];
  }
}

__global__ void v41_swa_prefill_gather_lens_kernel(
    int32_t* __restrict__ prefill_gather_lens,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ replay_start, int num_prefills, int num_decodes,
    int window_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_prefills) {
    return;
  }
  const int req = num_decodes + i;
  const int query_len = query_start_loc[req + 1] - query_start_loc[req];
  const int prefix_len = seq_lens[req] - query_len;
  // SWA bounded replay: context below replay_start has no window KV.
  int visible_prefix = prefix_len - replay_start[req];
  if (visible_prefix < 0) {
    visible_prefix = 0;
  }
  const int cap = window_size - 1;
  prefill_gather_lens[i] =
      query_len + (visible_prefix < cap ? visible_prefix : cap);
}

void check_int32_1d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.is_contiguous() &&
                  t.scalar_type() == at::kInt,
              name, " must be a contiguous 1D int32 CUDA tensor");
}

}  // namespace

void v41_swa_indices_and_lens(torch::Tensor& swa_indices,
                              torch::Tensor& swa_lens,
                              torch::Tensor& query_start_loc,
                              torch::Tensor& seq_lens,
                              torch::Tensor& token_to_req_indices,
                              torch::Tensor& is_valid_token,
                              torch::Tensor& block_table,
                              torch::Tensor& replay_start, int64_t window_size,
                              int64_t block_size, int64_t num_tokens) {
  if (num_tokens == 0) {
    return;
  }
  TORCH_CHECK(swa_indices.is_cuda() && swa_indices.is_contiguous() &&
                  swa_indices.scalar_type() == at::kInt &&
                  swa_indices.dim() >= 2,
              "swa_indices must be a contiguous int32 CUDA tensor of rank >= 2");
  check_int32_1d(swa_lens, "swa_lens");
  check_int32_1d(query_start_loc, "query_start_loc");
  check_int32_1d(seq_lens, "seq_lens");
  check_int32_1d(token_to_req_indices, "token_to_req_indices");
  check_int32_1d(replay_start, "replay_start");
  TORCH_CHECK(is_valid_token.is_cuda() && is_valid_token.dim() == 1 &&
                  is_valid_token.is_contiguous() &&
                  is_valid_token.scalar_type() == at::kBool,
              "is_valid_token must be a contiguous 1D bool CUDA tensor");
  TORCH_CHECK(block_table.is_cuda() && block_table.dim() == 2 &&
                  block_table.scalar_type() == at::kInt &&
                  block_table.stride(1) == 1,
              "block_table must be a 2D int32 CUDA tensor with contiguous rows");
  TORCH_CHECK(window_size > 0 && block_size > 0,
              "window_size and block_size must be positive");

  const int64_t index_width = swa_indices.numel() / swa_indices.size(0);
  TORCH_CHECK(index_width >= window_size, "swa_indices rows are ", index_width,
              " wide, narrower than the ", window_size, " token window");
  TORCH_CHECK(swa_indices.size(0) >= num_tokens &&
                  swa_lens.numel() >= num_tokens &&
                  is_valid_token.numel() >= num_tokens,
              "swa_indices / swa_lens / is_valid_token are shorter than "
              "num_tokens");
  // ``seq_lens`` is the batch's request count; the other request-indexed
  // tensors must cover it. Which request a token belongs to only the device
  // knows, so this is as far as the bounds can be checked here.
  const int64_t num_reqs = seq_lens.numel();
  TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1 &&
                  replay_start.numel() >= num_reqs &&
                  block_table.size(0) >= num_reqs,
              "query_start_loc / replay_start / block_table are shorter than "
              "seq_lens' request count");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(swa_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const size_t smem = static_cast<size_t>(window_size) * sizeof(int32_t);
  v41_swa_indices_and_lens_kernel<<<static_cast<unsigned>(num_tokens),
                                    kSwaIndexThreads, smem, stream>>>(
      swa_indices.data_ptr<int32_t>(), index_width,
      swa_lens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), token_to_req_indices.data_ptr<int32_t>(),
      is_valid_token.data_ptr<bool>(), block_table.data_ptr<int32_t>(),
      block_table.stride(0), replay_start.data_ptr<int32_t>(),
      static_cast<int>(window_size), static_cast<int>(block_size));
}

void v41_swa_prefill_gather_lens(torch::Tensor& prefill_gather_lens,
                                 torch::Tensor& seq_lens,
                                 torch::Tensor& query_start_loc,
                                 torch::Tensor& replay_start,
                                 int64_t num_prefills, int64_t num_decodes,
                                 int64_t window_size) {
  if (num_prefills == 0) {
    return;
  }
  check_int32_1d(prefill_gather_lens, "prefill_gather_lens");
  check_int32_1d(seq_lens, "seq_lens");
  check_int32_1d(query_start_loc, "query_start_loc");
  check_int32_1d(replay_start, "replay_start");
  TORCH_CHECK(window_size > 0, "window_size must be positive");

  const int64_t num_reqs = num_decodes + num_prefills;
  TORCH_CHECK(prefill_gather_lens.numel() >= num_prefills,
              "prefill_gather_lens is shorter than num_prefills");
  TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1 &&
                  seq_lens.numel() >= num_reqs &&
                  replay_start.numel() >= num_reqs,
              "query_start_loc / seq_lens / replay_start are shorter than the "
              "batch's request count");

  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(prefill_gather_lens));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned blocks = static_cast<unsigned>(
      (num_prefills + kGatherLenThreads - 1) / kGatherLenThreads);
  v41_swa_prefill_gather_lens_kernel<<<blocks, kGatherLenThreads, 0, stream>>>(
      prefill_gather_lens.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(), replay_start.data_ptr<int32_t>(),
      static_cast<int>(num_prefills), static_cast<int>(num_decodes),
      static_cast<int>(window_size));
}

}  // namespace vllm_iluvatar
