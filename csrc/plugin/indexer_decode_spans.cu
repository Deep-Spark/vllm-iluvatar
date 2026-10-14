// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#include "ops.h"

namespace vllm_iluvatar {
namespace {
__device__ __forceinline__ int64_t floor_compressed(int64_t value, int ratio) {
  if (ratio == 1) return value;
  // CUDA integer division truncates negative values; metadata uses floor.
  return value >= 0 ? value / 2 : -((-value + 1) / 2);
}

__global__ void indexer_decode_spans_kernel(
    const int32_t* seq_lens, const int32_t* cu_query, int32_t* ks, int32_t* ke,
    int requests, int real_tokens, int padded_tokens, int ratio) {
  const int token = threadIdx.x;
  if (token >= padded_tokens) return;
  int64_t begin = 0;
  int64_t end = 0;
  if (token < real_tokens) {
    int64_t prefix = 0;
    for (int request = 0; request < requests; ++request) {
      const int query_begin = cu_query[request];
      const int query_end = cu_query[request + 1];
      const int64_t length = seq_lens[request];
      if (token >= query_begin && token < query_end) {
        begin = prefix;
        end = prefix + floor_compressed(
            length - (query_end - query_begin) + token - query_begin + 1,
            ratio);
        break;
      }
      prefix += floor_compressed(length, ratio);
    }
  }
  // Rewrite every padded row on each invocation, including empty requests.
  ks[token] = static_cast<int32_t>(begin);
  ke[token] = static_cast<int32_t>(end);
}

void check_int32_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 1 && tensor.is_contiguous() &&
                  tensor.scalar_type() == at::kInt,
              name, " must be a contiguous int32 CUDA vector");
}

bool disjoint(const torch::Tensor& left, int64_t left_elements,
              const torch::Tensor& right, int64_t right_elements) {
  const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr<int32_t>());
  const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr<int32_t>());
  return left_begin + left_elements * sizeof(int32_t) <= right_begin ||
         right_begin + right_elements * sizeof(int32_t) <= left_begin;
}
}  // namespace

void v41_indexer_decode_spans(
    torch::Tensor& seq_lens, torch::Tensor& cu_query, torch::Tensor& ks,
    torch::Tensor& ke, int64_t num_requests, int64_t num_real_tokens,
    int64_t num_padded_tokens, int64_t compress_ratio) {
  check_int32_vector(seq_lens, "seq_lens");
  check_int32_vector(cu_query, "cu_query");
  check_int32_vector(ks, "ks");
  check_int32_vector(ke, "ke");
  TORCH_CHECK(seq_lens.device() == ks.device() && cu_query.device() == ks.device() &&
                  ke.device() == ks.device(), "decode span tensors must share a device");
  TORCH_CHECK(num_requests >= 1 && num_requests <= 4 &&
                  num_padded_tokens >= 1 && num_padded_tokens <= 24 &&
                  num_real_tokens >= 0 && num_real_tokens <= num_padded_tokens &&
                  (compress_ratio == 1 || compress_ratio == 2),
              "unsupported small decode span dimensions");
  TORCH_CHECK(seq_lens.numel() >= num_requests && cu_query.numel() >= num_requests + 1 &&
                  ks.numel() >= num_padded_tokens && ke.numel() >= num_padded_tokens,
              "decode span tensors are shorter than the active batch");
  TORCH_CHECK(disjoint(ks, num_padded_tokens, ke, num_padded_tokens) &&
                  disjoint(ks, num_padded_tokens, seq_lens, num_requests) &&
                  disjoint(ke, num_padded_tokens, seq_lens, num_requests) &&
                  disjoint(ks, num_padded_tokens, cu_query, num_requests + 1) &&
                  disjoint(ke, num_padded_tokens, cu_query, num_requests + 1),
              "decode span outputs must not overlap inputs or each other");
  const at::cuda::OptionalCUDAGuard guard(device_of(ks));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  indexer_decode_spans_kernel<<<1, 64, 0, stream>>>(
      seq_lens.data_ptr<int32_t>(), cu_query.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(), ke.data_ptr<int32_t>(), static_cast<int>(num_requests),
      static_cast<int>(num_real_tokens), static_cast<int>(num_padded_tokens),
      static_cast<int>(compress_ratio));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace vllm_iluvatar
