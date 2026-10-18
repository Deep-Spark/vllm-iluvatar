// SPDX-License-Identifier: Apache-2.0
// Generic GPU/CPU block copies required by vLLM's native KV offloader.

#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace vllm_iluvatar {

void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       bool is_src_access_order_any) {
  TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
  TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
  TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
  TORCH_CHECK(src_ptrs.scalar_type() == at::kLong, "src_ptrs must be int64");
  TORCH_CHECK(dst_ptrs.scalar_type() == at::kLong, "dst_ptrs must be int64");
  TORCH_CHECK(sizes.scalar_type() == at::kLong, "sizes must be int64");
  TORCH_CHECK(src_ptrs.dim() == 1 && src_ptrs.is_contiguous(),
              "src_ptrs must be contiguous 1D");
  TORCH_CHECK(dst_ptrs.dim() == 1 && dst_ptrs.is_contiguous(),
              "dst_ptrs must be contiguous 1D");
  TORCH_CHECK(sizes.dim() == 1 && sizes.is_contiguous(),
              "sizes must be contiguous 1D");

  const int64_t count = src_ptrs.numel();
  TORCH_CHECK(dst_ptrs.numel() == count,
              "dst_ptrs length must match src_ptrs");
  TORCH_CHECK(sizes.numel() == count, "sizes length must match src_ptrs");
  if (count == 0) {
    return;
  }

  const int64_t* src_data = src_ptrs.data_ptr<int64_t>();
  const int64_t* dst_data = dst_ptrs.data_ptr<int64_t>();
  const int64_t* size_data = sizes.data_ptr<int64_t>();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // CoreX does not expose CUDA 12.8's cuMemcpyBatchAsync API. Preserve the
  // upstream operation's asynchronous stream semantics with per-copy DMA.
  // cudaMemcpyDefault infers D2D/H2D/D2H from UVA pointer attributes.
  (void)is_src_access_order_any;
  for (int64_t i = 0; i < count; ++i) {
    TORCH_CHECK(size_data[i] >= 0, "sizes must be non-negative");
    if (size_data[i] == 0) {
      continue;
    }
    AT_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(dst_data[i]),
        reinterpret_cast<const void*>(src_data[i]),
        static_cast<size_t>(size_data[i]), cudaMemcpyDefault, stream));
  }
}

}  // namespace vllm_iluvatar
