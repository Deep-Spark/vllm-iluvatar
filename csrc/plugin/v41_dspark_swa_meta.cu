// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include "ops.h"

namespace vllm_iluvatar {
namespace {
// One CTA per output token; padding is written explicitly, including reused
// buffers. Only the window's actual page IDs are read, independent of capacity.
__global__ void v41_dspark_noncausal_swa_kernel(
    const int32_t* seq, const int32_t* qsl, const int32_t* table,
    const bool* valid, int32_t* indices, int32_t* lens,
    int nreq, int ntok, int cols, int block_size, int window, int width,
    int64_t table_row_stride, int64_t table_col_stride,
    int64_t index_row_stride, int64_t index_col_stride, int64_t lens_stride) {
  const int token = blockIdx.x;
  __shared__ int owner;
  __shared__ int first;
  __shared__ int count;
  if (threadIdx.x == 0) {
    owner = -1;
    first = count = 0;
    if (valid[token]) {
      for (int req = 0; req < nreq; ++req) {
        const int begin = qsl[req];
        const int end = qsl[req + 1] < ntok ? qsl[req + 1] : ntok;
        if (token < begin || token >= end) continue;
        owner = req;
        int query_len = 0;
        for (int i = begin; i < end; ++i) query_len += valid[i] ? 1 : 0;
        const int capacity = cols * block_size;
        const int length = seq[req] < capacity ? seq[req] : capacity;
        const int start = length - query_len - window;
        first = start > 0 ? start : 0;
        const int remaining = length - first;
        count = remaining > 0 ? (remaining < width ? remaining : width) : 0;
        break;
      }
    }
    lens[token * lens_stride] = count;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    int32_t slot = -1;
    if (j < count) {
      const int pos = first + j;
      const int32_t page = table[owner * table_row_stride +
                                  (pos / block_size) * table_col_stride];
      slot = page * block_size + pos % block_size;
    }
    indices[token * index_row_stride + j * index_col_stride] = slot;
  }
}

}  // namespace

void v41_dspark_noncausal_swa_indices_and_lens(
    torch::Tensor& indices, torch::Tensor& lens,
    torch::Tensor& seq_lens, torch::Tensor& query_start_loc,
    torch::Tensor& block_table, torch::Tensor& is_valid_token,
    int64_t block_size, int64_t window_size, int64_t index_width,
    int64_t num_tokens) {
  const auto device = indices.device();
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == at::kInt &&
                  indices.dim() == 3 && indices.size(1) == 1 &&
                  indices.size(0) >= num_tokens && indices.size(2) == index_width,
              "indices must be CUDA int32 [tokens,1,index_width]");
  for (const auto* tensor : {&lens, &seq_lens, &query_start_loc, &block_table}) {
    TORCH_CHECK(tensor->device() == device && tensor->scalar_type() == at::kInt,
                "integer metadata tensors must be int32 on the output device");
  }
  TORCH_CHECK(lens.dim() == 1 && lens.numel() >= num_tokens &&
                  seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  query_start_loc.dim() == 1 && query_start_loc.is_contiguous() &&
                  query_start_loc.numel() >= 1 && block_table.dim() == 2 &&
                  is_valid_token.device() == device &&
                  is_valid_token.scalar_type() == at::kBool &&
                  is_valid_token.dim() == 1 && is_valid_token.is_contiguous() &&
                  is_valid_token.numel() >= num_tokens,
              "invalid DSpark metadata shapes or strides");
  const int64_t num_reqs = query_start_loc.numel() - 1;
  TORCH_CHECK(seq_lens.numel() >= num_reqs && block_table.size(0) >= num_reqs &&
                  block_size > 0 && window_size >= 0 && index_width >= 0 &&
                  num_tokens >= 0 && num_tokens <= std::numeric_limits<int>::max() &&
                  index_width <= std::numeric_limits<int>::max() &&
                  window_size <= std::numeric_limits<int>::max() &&
                  block_table.size(1) <= std::numeric_limits<int>::max() / block_size,
              "invalid request count or integer metadata bounds");
  if (num_tokens == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  v41_dspark_noncausal_swa_kernel<<<static_cast<unsigned>(num_tokens), 256, 0, stream>>>(
      seq_lens.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
      block_table.data_ptr<int32_t>(), is_valid_token.data_ptr<bool>(),
      indices.data_ptr<int32_t>(), lens.data_ptr<int32_t>(),
      static_cast<int>(num_reqs), static_cast<int>(num_tokens),
      static_cast<int>(block_table.size(1)), static_cast<int>(block_size),
      static_cast<int>(window_size), static_cast<int>(index_width),
      block_table.stride(0), block_table.stride(1), indices.stride(0),
      indices.stride(2), lens.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace vllm_iluvatar
