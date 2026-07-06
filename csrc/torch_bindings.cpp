// SPDX-License-Identifier: Apache-2.0
// torch.ops._C bindings for the Iluvatar plugin.
//
// ixformer's native library used to occupy the ``_C`` torch namespace; it has
// since been renamed to ``_ixformer_C`` (opt_ds4_rebase), freeing ``_C`` for the
// plugin to register the vLLM-community-signature operators here. Importing the
// compiled module (``import vllm_iluvatar._C``) triggers these static
// TORCH_LIBRARY registrations, so community ``torch.ops._C.*`` call sites resolve
// to the Iluvatar implementations.
#include <torch/library.h>

#include "core/registration.h"
#include "ops.h"

// Schemas are copied verbatim from vLLM upstream:
//   * rms_norm: csrc/libtorch_stable/torch_bindings.cpp
//   * rotary_embedding: csrc/libtorch_stable/torch_bindings.cpp
//   * weak_ref_tensor / get_cuda_view_from_cpu_tensor: csrc/torch_bindings.cpp
// TORCH_LIBRARY_FRAGMENT (rather than TORCH_LIBRARY) is used so the plugin only
// extends the ``_C`` namespace and never conflicts with a stub definition.
TORCH_LIBRARY_FRAGMENT_EXPAND(_C, ops) {
  ops.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rope_dim_offset=0, bool inverse=False) -> ()");
  // Plugin-internal op (NOT a community torch.ops._C op): backs the
  // ApplyRotaryEmb OOT override; consumed only by the plugin, never by
  // community source. Kept in the ``_C`` namespace alongside the other native
  // ops for a single load point (import vllm_iluvatar._C).
  ops.def(
      "apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) "
      "-> Tensor");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("rms_norm", &vllm_iluvatar::rms_norm);
  ops.impl("rotary_embedding", &vllm_iluvatar::rotary_embedding);
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb);
  // Aliasing a CUDA tensor's storage; registered on the CUDA key like upstream.
  ops.impl("weak_ref_tensor", &vllm_iluvatar::weak_ref_tensor);
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CPU, ops) {
  // Input is a CPU tensor; registered on the CPU key like upstream.
  ops.impl("get_cuda_view_from_cpu_tensor",
           &vllm_iluvatar::get_cuda_view_from_cpu_tensor);
}

TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops) {
  // Shape propagation for the out-of-place apply_rotary_emb (torch.compile /
  // FakeTensor / opcheck): output mirrors x's shape.
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb_meta);
}

REGISTER_EXTENSION(_C)
