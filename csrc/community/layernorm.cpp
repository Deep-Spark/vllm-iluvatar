// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.rms_norm — vLLM community signature
//   rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()
//
// Thin adapter over ixformer's high-level ``ixformer::infer::rms_norm`` (linked
// from libixformer.so; declared in ixformer_ops.h). ixformer's wrapper already
// does dtype dispatch, the strided-3D fast path and all shape/contiguity checks,
// so we only remap the argument order:
//   vLLM    (result, input, weight, epsilon)
//   ixformer(input, weight, output=result, fused_bias=None, eps)

#include <torch/all.h>

#include "ixformer_ops.h"
#include "ops.h"

namespace vllm_iluvatar {

void rms_norm(torch::Tensor& result, torch::Tensor& input, torch::Tensor& weight,
              double epsilon) {
  c10::optional<at::Tensor> no_fused_bias;
  ixformer::infer::rms_norm(input, weight, result, no_fused_bias, epsilon);
}

}  // namespace vllm_iluvatar
