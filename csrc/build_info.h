// SPDX-License-Identifier: Apache-2.0
// Build metadata for the compiled extension.
//
// Arch-gated kernels (e.g. the FP8 cache path, which only exists on ivcore30)
// keep their torch.ops._C schemas on every architecture and raise when called,
// so callers and tests need a way to ask which build they are talking to.
#pragma once

#include <string>

namespace vllm_iluvatar {

// compiled_cuda_arch() -> str
// The Iluvatar architecture this extension was compiled for -- "ivcore11" or
// "ivcore30" -- i.e. CMAKE_CUDA_ARCHITECTURES as seen at configure time.
std::string compiled_cuda_arch();

}  // namespace vllm_iluvatar
