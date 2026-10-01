// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 sliding-window metadata ops:
//   v41_swa_indices_and_lens      -- per-token paged key slots for the window
//   v41_swa_prefill_gather_lens   -- per-request context rows a prefill gathers
//
// These replace ``ixfops.ds4_compute_swa_decode_indices_and_lens`` and
// ``ixfops.ds4_compute_swa_prefill_gather_lens``, whose contract predates SWA
// bounded replay (``CacheConfig.swa_bounded_replay``): they take no per-request
// lower bound, and after a prefix hit the scheduler rewinds a request to
// ``replay_start`` so only the window from there on has been written.
//
// Both ops follow the community Triton kernels in
// ``vllm/vllm/v1/attention/backends/mla/sparse_swa.py``:
// ``_compute_swa_indices_and_lens_kernel`` and
// ``ComputePrefillMetadataKernel``. In particular the index op keeps the
// Triton contract rather than ixformer's: output row ``pid`` describes
// absolute token ``pid + token_offset``, ``swa_lens`` is the window span
// (``end_pos - start_pos``) rather than a compacted count, a null page stays
// in the row as ``page * block_size + offset``, an invalid token clears its
// whole index row to -1, and ``has_image`` widens the window with the
// per-token left/right in-image visibility.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "ops.h"

namespace vllm_iluvatar {
namespace {

constexpr int kSwaIndexThreads = 128;
constexpr int kGatherLenThreads = 128;

__global__ void v41_swa_indices_and_lens_kernel(
    int32_t* __restrict__ swa_indices, int64_t indices_stride,
    int32_t* __restrict__ swa_lens, int window_size, int index_width,
    const int32_t* __restrict__ left_visible,
    const int32_t* __restrict__ right_visible,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ token_to_req_indices,
    const bool* __restrict__ is_valid_token,
    const int32_t* __restrict__ block_table, int64_t block_table_stride,
    int block_size, const int32_t* __restrict__ replay_start, int token_offset,
    int has_image) {
  const int pid = blockIdx.x;
  const int token_idx = pid + token_offset;
  int32_t* row = swa_indices + static_cast<int64_t>(pid) * indices_stride;

  if (!is_valid_token[token_idx]) {
    // Graph-capture or batch padding. Clear the row so a padded token cannot
    // gather through stale indices.
    if (threadIdx.x == 0) {
      swa_lens[pid] = 0;
    }
    for (int off = threadIdx.x; off < index_width; off += blockDim.x) {
      row[off] = -1;
    }
    return;
  }

  const int req = token_to_req_indices[token_idx];
  const int query_start = query_start_loc[req];
  const int query_len = query_start_loc[req + 1] - query_start;
  const int seq_len = seq_lens[req];
  const int pos = seq_len - query_len + token_idx - query_start;

  // In-image bidirectional visibility widens the window: it starts up to
  // max(left - (window_size - 1), 0) positions earlier and extends ``right``
  // positions past the query token.
  int left = 0;
  int right = 0;
  if (has_image) {
    left = left_visible[token_idx];
    right = right_visible[token_idx];
  }
  int left_add = left - (window_size - 1);
  if (left_add < 0) {
    left_add = 0;
  }
  int start_pos = pos - (window_size - 1) - left_add;
  if (start_pos < 0) {
    start_pos = 0;
  }
  // SWA bounded replay: no window KV exists below the request's replay start.
  const int lower_bound = replay_start[req];
  if (start_pos < lower_bound) {
    start_pos = lower_bound;
  }
  const int end_pos = pos + right + 1;
  const int swa_len = end_pos - start_pos;
  if (threadIdx.x == 0) {
    swa_lens[pid] = swa_len;
  }

  const int32_t* pages =
      block_table + static_cast<int64_t>(req) * block_table_stride;
  for (int off = threadIdx.x; off < index_width; off += blockDim.x) {
    int32_t slot = -1;
    if (off < swa_len) {
      const int p = start_pos + off;
      const int page = pages[p / block_size];
      slot = page * block_size + p % block_size;
    }
    row[off] = slot;
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
                              torch::Tensor& swa_lens, int64_t window_size,
                              int64_t index_width,
                              torch::Tensor& left_visible,
                              torch::Tensor& right_visible,
                              torch::Tensor& query_start_loc,
                              torch::Tensor& seq_lens,
                              torch::Tensor& token_to_req_indices,
                              torch::Tensor& is_valid_token,
                              torch::Tensor& block_table, int64_t block_size,
                              torch::Tensor& replay_start, int64_t num_tokens,
                              int64_t token_offset, bool has_image) {
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
  TORCH_CHECK(window_size > 0 && index_width > 0 && block_size > 0,
              "window_size, index_width and block_size must be positive");
  TORCH_CHECK(token_offset >= 0, "token_offset must be non-negative");
  TORCH_CHECK(swa_indices.size(-1) >= index_width, "swa_indices last dim is ",
              swa_indices.size(-1), ", narrower than index_width ",
              index_width);
  const int64_t abs_end = num_tokens + token_offset;
  TORCH_CHECK(swa_indices.size(0) >= num_tokens &&
                  swa_lens.numel() >= num_tokens &&
                  is_valid_token.numel() >= abs_end &&
                  token_to_req_indices.numel() >= abs_end,
              "index rows or token maps are shorter than the token range");
  if (has_image) {
    check_int32_1d(left_visible, "left_visible");
    check_int32_1d(right_visible, "right_visible");
    TORCH_CHECK(left_visible.numel() >= abs_end &&
                    right_visible.numel() >= abs_end,
                "left_visible / right_visible are shorter than the token range");
  }
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
  v41_swa_indices_and_lens_kernel<<<static_cast<unsigned>(num_tokens),
                                    kSwaIndexThreads, 0, stream>>>(
      swa_indices.data_ptr<int32_t>(), swa_indices.stride(0),
      swa_lens.data_ptr<int32_t>(), static_cast<int>(window_size),
      static_cast<int>(index_width),
      has_image ? left_visible.data_ptr<int32_t>() : nullptr,
      has_image ? right_visible.data_ptr<int32_t>() : nullptr,
      query_start_loc.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      token_to_req_indices.data_ptr<int32_t>(),
      is_valid_token.data_ptr<bool>(), block_table.data_ptr<int32_t>(),
      block_table.stride(0), static_cast<int>(block_size),
      replay_start.data_ptr<int32_t>(), static_cast<int>(token_offset),
      has_image ? 1 : 0);
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
