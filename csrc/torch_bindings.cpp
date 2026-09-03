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

#include "build_info.h"
#include "core/registration.h"
#include "community/ops.h"
#include "plugin/ops.h"

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
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");
  // Plugin-internal query op: arch-gated operators (e.g. the FP8 pair, which
  // only carries kernels on ivcore30) are bound on every arch, so callers and
  // tests need a way to ask which build this is. Device-independent, hence a
  // catch-all kernel registered together with the schema.
  ops.def("compiled_cuda_arch() -> str", &vllm_iluvatar::compiled_cuda_arch);
  ops.def(
      "reshape_and_cache_fp8("
      "Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, "
      "Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale"
      ") -> ()");
  // Plugin-internal op (NOT a community torch.ops._C op): backs the
  // ApplyRotaryEmb OOT override; consumed only by the plugin, never by
  // community source. Kept in the ``_C`` namespace alongside the other native
  // ops for a single load point (import vllm_iluvatar._C).
  ops.def(
      "apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) "
      "-> Tensor");
  ops.def(
      "fused_qk_rmsnorm_rope_gate("
      "Tensor q_gate, Tensor k, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, Tensor positions, float epsilon, "
      "int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim"
      ") -> (Tensor, Tensor, Tensor)");
  ops.def(
      "fused_qk_rmsnorm_gate("
      "Tensor q_gate, Tensor k, Tensor q_weight, Tensor k_weight, "
      "float epsilon, int num_q_heads, int num_kv_heads, int head_dim"
      ") -> (Tensor, Tensor, Tensor)");
  ops.def(
      "concat_and_cache_dsa_int8("
      "Tensor kv_int8, Tensor nope_scale, Tensor rope_scale, "
      "Tensor! kv_cache, Tensor! nope_scale_cache, Tensor! rope_scale_cache, "
      "Tensor slot_mapping) -> ()");
  ops.def(
      "reshape_and_cache_with_scale("
      "Tensor key, Tensor value, Tensor key_scale, Tensor! key_cache, "
      "Tensor! value_cache, Tensor! key_scale_cache, Tensor slot_mapping) "
      "-> ()");
  ops.def(
      "reshape_and_cache_mix("
      "Tensor key, Tensor value, Tensor key_scale, Tensor! key_cache, "
      "Tensor! value_cache, Tensor! key_scale_cache, Tensor slot_mapping) "
      "-> ()");
  // MiniMax QK-Norm two-stage ops (plugin-internal, TP>1 path).
  ops.def("minimax_qk_var(Tensor qkv, int q_size, int kv_size) -> Tensor");
  ops.def(
      "minimax_qk_rms_apply(Tensor qkv, Tensor qk_var, Tensor q_weight, "
      "Tensor k_weight, int q_size, int kv_size, int tp_world, "
      "float epsilon) -> (Tensor, Tensor)");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("rms_norm", &vllm_iluvatar::rms_norm);
  ops.impl("rotary_embedding", &vllm_iluvatar::rotary_embedding);
  ops.impl("static_scaled_fp8_quant",
           &vllm_iluvatar::static_scaled_fp8_quant);
  ops.impl("reshape_and_cache_fp8",
           &vllm_iluvatar::reshape_and_cache_fp8);
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb);
  ops.impl("fused_qk_rmsnorm_rope_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_rope_gate);
  ops.impl("fused_qk_rmsnorm_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_gate);
  ops.impl("concat_and_cache_dsa_int8",
           &vllm_iluvatar::concat_and_cache_dsa_int8);
  ops.impl("reshape_and_cache_with_scale",
           &vllm_iluvatar::reshape_and_cache_with_scale);
  ops.impl("reshape_and_cache_mix",
           &vllm_iluvatar::reshape_and_cache_mix);
  ops.impl("minimax_qk_var", &vllm_iluvatar::minimax_qk_var);
  ops.impl("minimax_qk_rms_apply", &vllm_iluvatar::minimax_qk_rms_apply);
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
  ops.impl("fused_qk_rmsnorm_rope_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_rope_gate_meta);
  ops.impl("fused_qk_rmsnorm_gate",
           &vllm_iluvatar::fused_qk_rmsnorm_gate_meta);
  ops.impl("minimax_qk_var", &vllm_iluvatar::minimax_qk_var_meta);
  ops.impl("minimax_qk_rms_apply",
           &vllm_iluvatar::minimax_qk_rms_apply_meta);
}

REGISTER_EXTENSION(_C)
