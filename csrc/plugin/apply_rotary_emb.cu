// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.apply_rotary_emb -- plugin-internal op (NO upstream counterpart)
//   apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor
//
// Backs the ApplyRotaryEmb OOT override with a real CUDA kernel (flash-attn
// style: separate cos/sin, out-of-place). Self-contained (no ixformer dep).
// Mirrors the community forward_static / vllm_flash_attn apply_rotary_emb
// semantics so it can replace the native PyTorch fallback with a real kernel.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "ops.h"
#include "utils.cuh"

namespace vllm_iluvatar {
namespace {

template <typename scalar_t, typename cache_t, bool IS_NEOX, int PACK_NUM>
__device__ __forceinline__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    bool inverse) {
  using vtype = typename TypeTraits<scalar_t, PACK_NUM>::type;
  using ctype = typename TypeTraits<cache_t, PACK_NUM>::type;

  if constexpr (IS_NEOX) {
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;
    ctype cos = reinterpret_cast<const ctype*>(cos_ptr + rot_offset)[0];
    ctype sin = reinterpret_cast<const ctype*>(sin_ptr + rot_offset)[0];
    vtype x = reinterpret_cast<vtype*>(arr + x_index)[0];
    vtype y = reinterpret_cast<vtype*>(arr + y_index)[0];
    vtype x_result, y_result;
#pragma unroll
    for (int i = 0; i < PACK_NUM; i++) {
      const float x_f = to_float(reinterpret_cast<scalar_t*>(&x)[i]);
      const float y_f = to_float(reinterpret_cast<scalar_t*>(&y)[i]);
      const float cos_f = to_float(reinterpret_cast<cache_t*>(&cos)[i]);
      float sin_f = to_float(reinterpret_cast<cache_t*>(&sin)[i]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&x_result)[i] =
          from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&y_result)[i] =
          from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
    }
    reinterpret_cast<vtype*>(arr + x_index)[0] = x_result;
    reinterpret_cast<vtype*>(arr + y_index)[0] = y_result;
  } else {
    // GPT-J interleaved: element 2r paired with 2r+1.
    const int index1 = 2 * rot_offset;
    const int index2 = 2 * rot_offset + PACK_NUM;
    ctype cos = reinterpret_cast<const ctype*>(cos_ptr + rot_offset)[0];
    ctype sin = reinterpret_cast<const ctype*>(sin_ptr + rot_offset)[0];
    vtype xy1 = reinterpret_cast<const vtype*>(arr + index1)[0];
    vtype xy2 = reinterpret_cast<const vtype*>(arr + index2)[0];
    vtype r1, r2;
    if constexpr (PACK_NUM == 2) {
      float x_f = to_float(reinterpret_cast<scalar_t*>(&xy1)[0]);
      float y_f = to_float(reinterpret_cast<scalar_t*>(&xy1)[1]);
      float cos_f = to_float(reinterpret_cast<cache_t*>(&cos)[0]);
      float sin_f = to_float(reinterpret_cast<cache_t*>(&sin)[0]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r1)[0] =
          from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r1)[1] =
          from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index1)[0] = r1;

      x_f = to_float(reinterpret_cast<scalar_t*>(&xy2)[0]);
      y_f = to_float(reinterpret_cast<scalar_t*>(&xy2)[1]);
      cos_f = to_float(reinterpret_cast<cache_t*>(&cos)[1]);
      sin_f = to_float(reinterpret_cast<cache_t*>(&sin)[1]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r2)[0] =
          from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r2)[1] =
          from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index2)[0] = r2;
    } else {
      const float x_f = to_float(reinterpret_cast<scalar_t*>(&xy1)[0]);
      const float y_f = to_float(reinterpret_cast<scalar_t*>(&xy2)[0]);
      const float cos_f = to_float(reinterpret_cast<cache_t*>(&cos)[0]);
      float sin_f = to_float(reinterpret_cast<cache_t*>(&sin)[0]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r1)[0] =
          from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r2)[0] =
          from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index1)[0] = r1;
      reinterpret_cast<vtype*>(arr + index2)[0] = r2;
    }
  }
}

// --- apply_rotary_emb kernel (flash-attn style: separate cos/sin, out-of-place)
// Rotates the first ``rot_dim`` channels of each head using per-seq cos/sin
// tables. Unlike ``rotary_embedding`` there is no positions gather: token at
// flat index t uses row ``t % seqlen`` of cos/sin (x laid out as
// [batch, seqlen, num_heads, head_dim] row-major). The pass-through channels
// ``x[..., rot_dim:]`` are already carried over by the caller's clone, so the
// kernel only touches the rotary slice.
template <typename scalar_t, typename cache_t, bool IS_NEOX, int PACK_NUM>
__global__ void apply_rotary_emb_kernel(
    scalar_t* __restrict__ out, const cache_t* __restrict__ cos,
    const cache_t* __restrict__ sin, const int seqlen, const int rot_dim,
    const int num_heads, const int64_t token_stride,
    const int64_t head_stride) {
  const int token_idx = blockIdx.x;
  const int s = token_idx % seqlen;
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cos + static_cast<int64_t>(s) * embed_dim;
  const cache_t* sin_ptr = sin + static_cast<int64_t>(s) * embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x * PACK_NUM; i < nq; i += blockDim.x * PACK_NUM) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        static_cast<int64_t>(token_idx) * token_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX, PACK_NUM>(
        out + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim,
        /*inverse=*/false);
  }
}

template <typename scalar_t, typename cache_t>
void launch_apply_rotary_emb(scalar_t* out, const cache_t* cos,
                             const cache_t* sin, int seqlen, int rot_dim,
                             int num_heads, int64_t token_stride,
                             int64_t head_stride, bool is_neox, int num_tokens,
                             cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(static_cast<int64_t>(num_heads) * rot_dim / 2,
                               512));

#define VLLM_ILU_LAUNCH_APPLY_ROPE(NEOX, PACK)                             \
  apply_rotary_emb_kernel<scalar_t, cache_t, NEOX, PACK>                   \
      <<<grid, block, 0, stream>>>(out, cos, sin, seqlen, rot_dim,         \
                                   num_heads, token_stride, head_stride)

  if constexpr (sizeof(scalar_t) == 2) {
    const bool can_pack = (rot_dim % 4 == 0) && (token_stride % 2 == 0) &&
                          (head_stride % 2 == 0);
    if (is_neox) {
      if (can_pack)
        VLLM_ILU_LAUNCH_APPLY_ROPE(true, 2);
      else
        VLLM_ILU_LAUNCH_APPLY_ROPE(true, 1);
    } else {
      if (can_pack)
        VLLM_ILU_LAUNCH_APPLY_ROPE(false, 2);
      else
        VLLM_ILU_LAUNCH_APPLY_ROPE(false, 1);
    }
  } else {
    if (is_neox)
      VLLM_ILU_LAUNCH_APPLY_ROPE(true, 1);
    else
      VLLM_ILU_LAUNCH_APPLY_ROPE(false, 1);
  }
#undef VLLM_ILU_LAUNCH_APPLY_ROPE
}

}  // namespace

torch::Tensor apply_rotary_emb(torch::Tensor& x, torch::Tensor& cos,
                               torch::Tensor& sin, bool interleaved) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "x must be [seqlen, num_heads, head_dim] or "
              "[batch, seqlen, num_heads, head_dim]");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2,
              "cos and sin must be [seqlen, rot_dim / 2]");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
              "cos and sin must share the same dtype");
  TORCH_CHECK(cos.size(0) == sin.size(0) && cos.size(1) == sin.size(1),
              "cos and sin must have the same shape");

  const int64_t head_dim = x.size(-1);
  const int64_t num_heads = x.size(-2);
  const int64_t seqlen = cos.size(0);
  const int64_t rot_dim = cos.size(1) * 2;
  TORCH_CHECK(rot_dim % 2 == 0, "rot_dim must be even");
  TORCH_CHECK(rot_dim <= head_dim, "rot_dim must not exceed head_dim");

  const int64_t num_tokens = x.numel() / (num_heads * head_dim);
  TORCH_CHECK(seqlen > 0 && num_tokens % seqlen == 0,
              "x token count must be a multiple of cos/sin seqlen");

  auto out = x.contiguous().clone();
  if (num_tokens == 0 || num_heads == 0 || rot_dim == 0) return out;

  const int64_t token_stride = num_heads * head_dim;
  const int64_t head_stride = head_dim;

  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto x_type = out.scalar_type();
  const auto cache_type = cos.scalar_type();
  void* out_ptr = out.data_ptr();
  const void* cos_ptr = cos.contiguous().data_ptr();
  const void* sin_ptr = sin.contiguous().data_ptr();

#define VLLM_ILU_DISPATCH_APPLY_ROPE(XT, CT)                                 \
  launch_apply_rotary_emb<XT, CT>(                                           \
      reinterpret_cast<XT*>(out_ptr), reinterpret_cast<const CT*>(cos_ptr),  \
      reinterpret_cast<const CT*>(sin_ptr), static_cast<int>(seqlen),        \
      static_cast<int>(rot_dim), static_cast<int>(num_heads), token_stride,  \
      head_stride, /*is_neox=*/!interleaved, static_cast<int>(num_tokens),   \
      stream)

#define VLLM_ILU_DISPATCH_APPLY_CACHE(XT)                                    \
  do {                                                                       \
    if (cache_type == at::kFloat) {                                          \
      VLLM_ILU_DISPATCH_APPLY_ROPE(XT, float);                               \
    } else if (cache_type == at::kHalf) {                                    \
      VLLM_ILU_DISPATCH_APPLY_ROPE(XT, half);                                \
    } else if (cache_type == at::kBFloat16) {                               \
      VLLM_ILU_DISPATCH_APPLY_ROPE(XT, __nv_bfloat16);                       \
    } else {                                                                 \
      TORCH_CHECK(false, "unsupported cos/sin dtype: ", cache_type);         \
    }                                                                        \
  } while (0)

  if (x_type == at::kFloat) {
    VLLM_ILU_DISPATCH_APPLY_CACHE(float);
  } else if (x_type == at::kHalf) {
    VLLM_ILU_DISPATCH_APPLY_CACHE(half);
  } else if (x_type == at::kBFloat16) {
    VLLM_ILU_DISPATCH_APPLY_CACHE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "unsupported x dtype: ", x_type);
  }

#undef VLLM_ILU_DISPATCH_APPLY_CACHE
#undef VLLM_ILU_DISPATCH_APPLY_ROPE

  return out;
}

torch::Tensor apply_rotary_emb_meta(torch::Tensor& x, torch::Tensor& /*cos*/,
                                    torch::Tensor& /*sin*/,
                                    bool /*interleaved*/) {
  return at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
}

}  // namespace vllm_iluvatar
