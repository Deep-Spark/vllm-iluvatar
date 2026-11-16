// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>
#include <string_view>

namespace vllm_iluvatar {

inline void check_sampling_fp64(bool needs_fp64) {
  // MR device double conversions/math can silently return incorrect values.
  TORCH_CHECK(!needs_fp64 ||
                  std::string_view(VLLM_ILUVATAR_CUDA_ARCH) != "ivcore11",
              "FP64 sampling is not supported on CoreX MR/ivcore11; "
              "use FP32 logits and use_fp64=False");
}

}  // namespace vllm_iluvatar
