// SPDX-License-Identifier: Apache-2.0
// Declarations for the vLLM-compatible torch.ops._C operators implemented by the
// Iluvatar plugin. Signatures mirror vLLM upstream csrc/ops.h so the community
// vllm/kernels/vllm_c.py and vllm/utils/torch_utils.py can call them unchanged.
#pragma once

#include <torch/all.h>

namespace vllm_iluvatar {

// rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()
// Backed by ixformer's RMSNorm kernels.
void rms_norm(torch::Tensor& result, torch::Tensor& input,
              torch::Tensor& weight, double epsilon);

// weak_ref_tensor(Tensor input) -> Tensor
// Returns a new tensor aliasing the same storage (no refcount on the source).
torch::Tensor weak_ref_tensor(torch::Tensor& tensor);

// get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor
// Maps a (pinned) CPU tensor into the CUDA address space and returns a view.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

}  // namespace vllm_iluvatar
