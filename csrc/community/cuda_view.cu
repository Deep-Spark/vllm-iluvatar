// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.weak_ref_tensor / get_cuda_view_from_cpu_tensor — vLLM community
// signatures. These two ops are pure torch + CUDA-runtime utilities (no ixformer
// kernel), so they are implemented self-contained here. Behaviour is ported from
// ixformer's src/ixformer/infer/vllm.cpp (which itself mirrors vLLM upstream
// csrc/cuda_view.cu / torch_bindings.cpp weak_ref_tensor).

#include <torch/all.h>

#include <cuda_runtime_api.h>

#include <vector>

#include "ops.h"

namespace vllm_iluvatar {

// Return a tensor that aliases the same storage as ``tensor`` without holding a
// reference on it (a "weak" view), matching vLLM's torch.ops._C.weak_ref_tensor.
torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");

  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();
  auto options = tensor.options();

  return torch::from_blob(data_ptr, sizes, strides, options);
}

// Map a (pinned) CPU tensor into the CUDA address space and return a CUDA view
// of the same data, matching vLLM's torch.ops._C.get_cuda_view_from_cpu_tensor.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  if (cpu_tensor.numel() == 0) {
    return torch::empty(cpu_tensor.sizes(),
                        cpu_tensor.options().device(torch::kCUDA));
  }

  if (cpu_tensor.is_pinned()) {
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {},
        cpu_tensor.options().device(torch::kCUDA));
  }

  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  size_t nbytes = contiguous_cpu.nbytes();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostAlloc failed: ", cudaGetErrorString(err));

  err = cudaMemcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    TORCH_CHECK(false,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::from_blob(device_ptr, contiguous_cpu.sizes(),
                          contiguous_cpu.strides(), deleter,
                          contiguous_cpu.options().device(torch::kCUDA));
}

}  // namespace vllm_iluvatar
