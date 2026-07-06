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

// rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
//                  int head_size, Tensor cos_sin_cache, bool is_neox,
//                  int rope_dim_offset=0, bool inverse=False) -> ()
// Self-contained CUDA kernel (no ixformer dependency); rotates query/key in
// place. Semantics mirror vLLM upstream pos_encoding_kernels.cu.
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

// apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor
// Plugin-internal op (community torch.ops._C has NO such op): backs the
// ApplyRotaryEmb OOT override with a real CUDA kernel (flash-attn style:
// separate cos/sin, out-of-place). Self-contained (no ixformer dependency).
torch::Tensor apply_rotary_emb(torch::Tensor& x, torch::Tensor& cos,
                               torch::Tensor& sin, bool interleaved);

// Meta/fake impl for apply_rotary_emb: output has the same shape as ``x`` (the
// rotary is applied out-of-place). Enables shape propagation under torch.compile
// / FakeTensor and torch.library.opcheck.
torch::Tensor apply_rotary_emb_meta(torch::Tensor& x, torch::Tensor& cos,
                                    torch::Tensor& sin, bool interleaved);

}  // namespace vllm_iluvatar
