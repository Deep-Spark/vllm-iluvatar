// SPDX-License-Identifier: Apache-2.0
//
// Forward declarations for ixformer's high-level inference ops that this
// extension links against (defined in libixformer.so, default visibility).
//
// ixformer only ships its kernels/ and comm/ headers — not infer/ — so we
// declare the exact symbols we call here. Signatures must match ixformer's
// definitions byte-for-byte (same torch headers + toolchain on the build host
// guarantee identical name mangling), copied from:
//   src/ixformer/infer/rms_norm.cpp
#pragma once

#include <torch/all.h>

namespace ixformer::infer {

// void rms_norm(Tensor& input, Tensor& weight, Tensor& output,
//               const optional<Tensor>& fused_bias, double eps)
// Handles dtype dispatch (fp32/fp16/bf16), the strided-3D fast path, and the
// contiguity/shape checks internally.
void rms_norm(at::Tensor& input, at::Tensor& weight, at::Tensor& output,
              const c10::optional<at::Tensor>& fused_bias, double eps);

}  // namespace ixformer::infer
