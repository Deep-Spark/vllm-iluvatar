// SPDX-License-Identifier: Apache-2.0
//
// torch.ops._C.rotary_embedding -- vLLM community signature
//   rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
//                    int head_size, Tensor cos_sin_cache, bool is_neox,
//                    int rope_dim_offset=0, bool inverse=False) -> ()
//
// Self-contained CUDA kernel for the Iluvatar plugin (no ixformer dependency).
// The host-side shape / stride derivation and validation are ported verbatim
// from vLLM upstream csrc/libtorch_stable/pos_encoding_kernels.cu so community
// vllm/_custom_ops.py and the RotaryEmbedding subclasses call this unchanged
// (supports positions 1D/2D, flat [*, H*D] or headed [*, H, D] query up to 4D,
// key=None, a cos_sin_cache dtype that may differ from q/k, rope_dim_offset and
// inverse). The device rotation math mirrors ixformer's PACK_NUM vectorization
// (src/kernels/infer/vllm/vllm_rotary_embedding.{h,cu}) for fp16/bf16, with a
// scalar (PACK=1) fallback, all accumulated in fp32 and written in place.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "ops.h"

namespace vllm_iluvatar {
namespace {

// --- minimal, self-contained type helpers (mirror ixformer TypeTraits) -------
template <typename T, int PACK_NUM>
struct RopeVec {};
template <>
struct RopeVec<float, 1> {
  using type = float;
};
template <>
struct RopeVec<float, 2> {
  using type = float2;
};
template <>
struct RopeVec<half, 1> {
  using type = half;
};
template <>
struct RopeVec<half, 2> {
  using type = half2;
};
template <>
struct RopeVec<__nv_bfloat16, 1> {
  using type = __nv_bfloat16;
};
template <>
struct RopeVec<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};

__device__ __forceinline__ float rope_to_float(float x) { return x; }
__device__ __forceinline__ float rope_to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float rope_to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T rope_from_float(float x);
template <>
__device__ __forceinline__ float rope_from_float<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ half rope_from_float<half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 rope_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX, int PACK_NUM>
__device__ __forceinline__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    bool inverse) {
  using vtype = typename RopeVec<scalar_t, PACK_NUM>::type;
  using ctype = typename RopeVec<cache_t, PACK_NUM>::type;

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
      const float x_f = rope_to_float(reinterpret_cast<scalar_t*>(&x)[i]);
      const float y_f = rope_to_float(reinterpret_cast<scalar_t*>(&y)[i]);
      const float cos_f = rope_to_float(reinterpret_cast<cache_t*>(&cos)[i]);
      float sin_f = rope_to_float(reinterpret_cast<cache_t*>(&sin)[i]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&x_result)[i] =
          rope_from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&y_result)[i] =
          rope_from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
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
      float x_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy1)[0]);
      float y_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy1)[1]);
      float cos_f = rope_to_float(reinterpret_cast<cache_t*>(&cos)[0]);
      float sin_f = rope_to_float(reinterpret_cast<cache_t*>(&sin)[0]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r1)[0] =
          rope_from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r1)[1] =
          rope_from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index1)[0] = r1;

      x_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy2)[0]);
      y_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy2)[1]);
      cos_f = rope_to_float(reinterpret_cast<cache_t*>(&cos)[1]);
      sin_f = rope_to_float(reinterpret_cast<cache_t*>(&sin)[1]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r2)[0] =
          rope_from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r2)[1] =
          rope_from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index2)[0] = r2;
    } else {
      const float x_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy1)[0]);
      const float y_f = rope_to_float(reinterpret_cast<scalar_t*>(&xy2)[0]);
      const float cos_f = rope_to_float(reinterpret_cast<cache_t*>(&cos)[0]);
      float sin_f = rope_to_float(reinterpret_cast<cache_t*>(&sin)[0]);
      if (inverse) sin_f = -sin_f;
      reinterpret_cast<scalar_t*>(&r1)[0] =
          rope_from_float<scalar_t>(x_f * cos_f - y_f * sin_f);
      reinterpret_cast<scalar_t*>(&r2)[0] =
          rope_from_float<scalar_t>(y_f * cos_f + x_f * sin_f);
      reinterpret_cast<vtype*>(arr + index1)[0] = r1;
      reinterpret_cast<vtype*>(arr + index2)[0] = r2;
    }
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX, int PACK_NUM>
__device__ __forceinline__ void apply_rotary_embedding(
    scalar_t* __restrict__ query, scalar_t* __restrict__ key,
    const cache_t* cache_ptr, const int num_heads, const int num_kv_heads,
    const int rot_dim, const int token_idx, const int64_t query_stride,
    const int64_t key_stride, const int64_t head_stride,
    const int64_t rope_dim_offset, const bool inverse) {
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x * PACK_NUM; i < nq; i += blockDim.x * PACK_NUM) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX, PACK_NUM>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x * PACK_NUM; i < nk; i += blockDim.x * PACK_NUM) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX, PACK_NUM>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
    }
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX, int PACK_NUM>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens] or [batch, seq]
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,  // nullptr or [..., num_kv_heads(*head_size)]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int64_t rope_dim_offset, const bool inverse) {
  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];
  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;
  apply_rotary_embedding<scalar_t, cache_t, IS_NEOX, PACK_NUM>(
      query, key, cache_ptr, num_heads, num_kv_heads, rot_dim, token_idx,
      query_stride, key_stride, head_stride, rope_dim_offset, inverse);
}

template <typename scalar_t, typename cache_t>
void launch_rotary_embedding(
    const int64_t* positions, scalar_t* query, scalar_t* key,
    const cache_t* cos_sin_cache, int rot_dim, int64_t query_stride,
    int64_t key_stride, int64_t head_stride, int num_heads, int num_kv_heads,
    int64_t rope_dim_offset, bool is_neox, bool inverse, int num_tokens,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(static_cast<int64_t>(num_heads) * rot_dim / 2,
                               512));

#define VLLM_ILU_LAUNCH_ROPE(NEOX, PACK)                                      \
  rotary_embedding_kernel<scalar_t, cache_t, NEOX, PACK>                      \
      <<<grid, block, 0, stream>>>(positions, query, key, cos_sin_cache,      \
                                   rot_dim, query_stride, key_stride,         \
                                   head_stride, num_heads, num_kv_heads,      \
                                   rope_dim_offset, inverse)

  if constexpr (sizeof(scalar_t) == 2) {
    const bool can_pack = (rot_dim % 4 == 0) && (query_stride % 2 == 0) &&
                          (key_stride % 2 == 0) && (head_stride % 2 == 0) &&
                          (rope_dim_offset % 2 == 0);
    if (is_neox) {
      if (can_pack)
        VLLM_ILU_LAUNCH_ROPE(true, 2);
      else
        VLLM_ILU_LAUNCH_ROPE(true, 1);
    } else {
      if (can_pack)
        VLLM_ILU_LAUNCH_ROPE(false, 2);
      else
        VLLM_ILU_LAUNCH_ROPE(false, 1);
    }
  } else {
    if (is_neox)
      VLLM_ILU_LAUNCH_ROPE(true, 1);
    else
      VLLM_ILU_LAUNCH_ROPE(false, 1);
  }
#undef VLLM_ILU_LAUNCH_ROPE
}

}  // namespace

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse) {
  const int64_t num_tokens = positions.numel();
  const int positions_ndim = positions.dim();

  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  TORCH_CHECK(positions.scalar_type() == at::kLong,
              "positions must be int64");

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size =
      key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim % 2 == 0, "cos_sin_cache last dim (rot_dim) must be even");
  TORCH_CHECK((rot_dim + rope_dim_offset) <= head_size,
              "rot_dim + rope_dim_offset must not exceed head_size");

  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

  const int query_ndim = query.dim();
  const int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  if (num_tokens == 0 || num_heads == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto q_type = query.scalar_type();
  const auto cache_type = cos_sin_cache.scalar_type();
  const int64_t* pos_ptr = positions.data_ptr<int64_t>();
  void* q_ptr = query.data_ptr();
  void* k_ptr = key.has_value() ? key->data_ptr() : nullptr;
  const void* cache_ptr = cos_sin_cache.data_ptr();

  TORCH_CHECK(key.has_value() ? key->scalar_type() == q_type : true,
              "query and key must share the same dtype");

#define VLLM_ILU_DISPATCH_ROPE(QT, CT)                                     \
  launch_rotary_embedding<QT, CT>(                                         \
      pos_ptr, reinterpret_cast<QT*>(q_ptr), reinterpret_cast<QT*>(k_ptr), \
      reinterpret_cast<const CT*>(cache_ptr), rot_dim, query_stride,       \
      key_stride, head_stride, num_heads, num_kv_heads, rope_dim_offset,   \
      is_neox, inverse, static_cast<int>(num_tokens), stream)

#define VLLM_ILU_DISPATCH_CACHE(QT)                     \
  do {                                                  \
    if (cache_type == at::kFloat) {                     \
      VLLM_ILU_DISPATCH_ROPE(QT, float);                \
    } else if (cache_type == at::kHalf) {               \
      VLLM_ILU_DISPATCH_ROPE(QT, half);                 \
    } else if (cache_type == at::kBFloat16) {           \
      VLLM_ILU_DISPATCH_ROPE(QT, __nv_bfloat16);        \
    } else {                                            \
      TORCH_CHECK(false, "unsupported cos_sin_cache dtype: ", cache_type); \
    }                                                   \
  } while (0)

  if (q_type == at::kFloat) {
    VLLM_ILU_DISPATCH_CACHE(float);
  } else if (q_type == at::kHalf) {
    VLLM_ILU_DISPATCH_CACHE(half);
  } else if (q_type == at::kBFloat16) {
    VLLM_ILU_DISPATCH_CACHE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "unsupported query dtype: ", q_type);
  }

#undef VLLM_ILU_DISPATCH_CACHE
#undef VLLM_ILU_DISPATCH_ROPE
}

}  // namespace vllm_iluvatar
