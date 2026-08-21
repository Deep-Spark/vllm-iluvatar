// SPDX-License-Identifier: Apache-2.0
#include "build_info.h"

#ifndef VLLM_ILUVATAR_CUDA_ARCH
#error "VLLM_ILUVATAR_CUDA_ARCH must be defined by the build (see CMakeLists.txt)"
#endif

namespace vllm_iluvatar {

std::string compiled_cuda_arch() { return VLLM_ILUVATAR_CUDA_ARCH; }

}  // namespace vllm_iluvatar
