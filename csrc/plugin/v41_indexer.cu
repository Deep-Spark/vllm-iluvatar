// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 Lightning-Indexer ops:
//   v41_indexer_k_norm_rope_store -- k_norm + RoPE + paged store of index keys,
//                                    as a bf16, E4M3 or INT8 record
//   v41_indexer_q_rope            -- RoPE + weight rescale, BF16 query in place
//   v41_indexer_q_rope_fp8        -- RoPE + true E4M3 quant, q_scale folded into
//                                    the weights
//   v41_indexer_q_rope_int8       -- RoPE + amax/127 INT8 quant, same folding
//   v41_fill_short_context_topk   -- the identity selection, when the whole
//                                    compressed context fits in topk
//
// The K side mirrors ``off/vllm/vllm/models/deepseek_v41/common/ops/
// indexer_k_store.py`` and the Q side ``off/vllm/vllm/models/deepseek_v4/
// common/ops/fused_indexer_q.py``, minus the MXFP4 records: this platform
// stores index keys as plain BF16, or as per-token E4M3 or INT8 with one FP32
// scale, the layout the V3.2 paged scoring kernels already read.
//
// No quantize-dequantize anywhere. The rows that land here are the rows the
// scoring kernel reads.
//
// Like the compressor ops, slots are compressed-domain global row indices, so
// the page and row come out of the slot alone and no block table is needed.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <tuple>

#include "ops.h"
#include "utils.cuh"
#include "v41_common.cuh"

namespace vllm_iluvatar {
namespace {

using namespace v41;

// V4.1's indexer geometry. Both kernels below lay one GPT-J pair per thread, so
// a 128-wide row is 64 threads -- exactly one Iluvatar warp, which is what lets
// the norm and the amax be plain shuffles with no shared memory or barrier.
constexpr int kIndexHeadDim = 128;
constexpr int kIndexRopeDim = 64;
constexpr int kIndexNopeDim = kIndexHeadDim - kIndexRopeDim;
constexpr int kIndexThreads = kIndexHeadDim / kRopePacked;  // 64
constexpr int kIndexRopeThreads = kIndexRopeDim / kRopePacked;  // 32
// Per-token quantized record: 128 value bytes then one FP32 scale, segregated
// inside the page (all rows' values, then all rows' scales). The FP8 and INT8
// records share this geometry exactly, differing only in the quantizer, which
// is why both report the same row width upstream.
constexpr int kIndexScaleBytes = 4;

// INT8 quantization, ixformer's "scheme A": one ``amax / 127`` scale per row.
// ``kInt8AmaxEps`` is its ``kAmaxEps`` -- small enough to leave any row with
// signal untouched, large enough that an all-zero row does not divide by zero.
constexpr float kInt8AbsMax = 127.0f;
constexpr float kInt8AmaxEps = 1e-10f;

// Which record the index keys are stored as. The host maps the op's ``record``
// string onto this once per call and instantiates the matching kernel.
enum class IndexRecord { kBf16, kFp8, kInt8 };

// ---------------------------------------------------------------------------
// A1 / A2: index-key production
// ---------------------------------------------------------------------------

// Write one quantized key row into the page, and into the pipeline mirror when
// this rank has to hand the key on. FP8 and INT8 differ only in ``Byte2``
// (``uchar2`` vs ``char2``) and in how ``bytes`` and ``scale`` were produced.
template <typename Byte2>
__device__ inline void store_quant_key(uint8_t* __restrict__ page_base,
                                       uint8_t* __restrict__ mirror,
                                       int rows_per_page, int64_t row, int d0,
                                       Byte2 bytes, float scale) {
  *reinterpret_cast<Byte2*>(page_base + row * kIndexHeadDim + d0) = bytes;
  if (mirror != nullptr) {
    *reinterpret_cast<Byte2*>(mirror + d0) = bytes;
  }
  if (d0 == 0) {
    *reinterpret_cast<float*>(
        page_base + static_cast<int64_t>(rows_per_page) * kIndexHeadDim +
        row * kIndexScaleBytes) = scale;
    if (mirror != nullptr) {
      *reinterpret_cast<float*>(mirror + kIndexHeadDim) = scale;
    }
  }
}

template <IndexRecord RECORD>
__global__ void v41_indexer_k_norm_rope_store_kernel(
    const __nv_bfloat16* __restrict__ k_pre, int64_t k_pre_stride,
    const void* __restrict__ positions, bool positions_are_int64,
    const __nv_bfloat16* __restrict__ rms_norm_weight, float rms_norm_eps,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    void* __restrict__ k_cache, int64_t page_stride, int64_t row_stride,
    int rows_per_page, int64_t num_pages,
    const void* __restrict__ kv_slot_mapping, bool slots_are_int64,
    int compress_ratio, void* __restrict__ row_out) {
  const int64_t token = blockIdx.x;
  const int64_t slot = load_index(kv_slot_mapping, slots_are_int64, token);
  const int64_t pos = load_index(positions, positions_are_int64, token);
  // Only the last token of a group publishes that group's index key. Every
  // condition here is block-uniform, so the shuffles below keep every partner.
  if (slot < 0 || (pos + 1) % compress_ratio != 0) {
    return;
  }
  const int64_t page = slot / rows_per_page;
  if (page >= num_pages) {
    return;
  }
  const int64_t row = slot % rows_per_page;

  const int d0 = threadIdx.x * kRopePacked;
  const __nv_bfloat162 in =
      *reinterpret_cast<const __nv_bfloat162*>(k_pre + token * k_pre_stride + d0);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  // The BF16 and FP8 references round the normed row back through bf16 on the
  // way out, and again after the rotation for parity with the Q side.
  // ixformer's INT8 reference carries fp32 from the norm straight into the
  // quantizer, so for that record rounding would be a deviation.
  constexpr bool kRoundThroughBf16 = RECORD != IndexRecord::kInt8;

  // k_norm in fp32.
  const float sum_sq =
      warp_reduce_sum_span(even * even + odd * odd, kIndexThreads);
  const float inv_rms =
      rsqrtf(sum_sq / static_cast<float>(kIndexHeadDim) + rms_norm_eps);
  const __nv_bfloat162 w =
      *reinterpret_cast<const __nv_bfloat162*>(rms_norm_weight + d0);
  even = even * inv_rms * __bfloat162float(w.x);
  odd = odd * inv_rms * __bfloat162float(w.y);
  if constexpr (kRoundThroughBf16) {
    even = bf16_round(even);
    odd = bf16_round(odd);
  }

  // A latent stands for the first token of its group, so group j rotates
  // against position j * compress_ratio.
  if (d0 >= kIndexNopeDim) {
    const int64_t rope_pos = (pos / compress_ratio) * compress_ratio;
    rope_pair_at(even, odd, cos_sin_cache + rope_pos * cos_sin_stride,
                 (d0 - kIndexNopeDim) / 2, kIndexRopeDim);
  }
  if constexpr (kRoundThroughBf16) {
    even = bf16_round(even);
    odd = bf16_round(odd);
  }

  if constexpr (RECORD == IndexRecord::kBf16) {
    __nv_bfloat162 out;
    out.x = __float2bfloat16_rn(even);
    out.y = __float2bfloat16_rn(odd);
    *reinterpret_cast<__nv_bfloat162*>(
        static_cast<__nv_bfloat16*>(k_cache) + page * page_stride +
        row * row_stride + d0) = out;
    // The pipeline replica, when this rank has to hand the key to the next.
    if (row_out != nullptr) {
      *reinterpret_cast<__nv_bfloat162*>(
          static_cast<__nv_bfloat16*>(row_out) + token * kIndexHeadDim + d0) =
          out;
    }
    return;
  }

  // Unread when FP8 is compiled out, which leaves that instantiation empty.
  [[maybe_unused]] uint8_t* const page_base =
      static_cast<uint8_t*>(k_cache) + page * page_stride;
  [[maybe_unused]] uint8_t* const mirror =
      row_out == nullptr
          ? nullptr
          : static_cast<uint8_t*>(row_out) +
                token * (kIndexHeadDim + kIndexScaleBytes);

  if constexpr (RECORD == IndexRecord::kInt8) {
    // Scheme A: a plain ``amax / 127``, not a power of two, and the clamp
    // comes after the rounding so a row at the limit lands on +-127 rather
    // than wrapping to -128.
    const float absmax = fmaxf(
        warp_reduce_max(fmaxf(fabsf(even), fabsf(odd)), kIndexThreads),
        kInt8AmaxEps);
    const float inv_scale = kInt8AbsMax / absmax;
    constexpr int kInt8Limit = static_cast<int>(kInt8AbsMax);
    char2 bytes;
    bytes.x = static_cast<int8_t>(
        max(-kInt8Limit, min(kInt8Limit, __float2int_rn(even * inv_scale))));
    bytes.y = static_cast<int8_t>(
        max(-kInt8Limit, min(kInt8Limit, __float2int_rn(odd * inv_scale))));
    store_quant_key(page_base, mirror, rows_per_page, row, d0, bytes,
                    absmax / kInt8AbsMax);
    return;
  }

#ifdef ILUVATAR_HAS_FP8_FP4
  // Per-token power-of-two scale over the whole 128-wide row.
  const float absmax = fmaxf(
      warp_reduce_max(fmaxf(fabsf(even), fabsf(odd)), kIndexThreads), 1e-4f);
  const float exponent = ceilf(log2f(absmax / kE4m3Max));
  const float inv_scale = exp2f(-exponent);

  uchar2 bytes;
  bytes.x = e4m3_bits(
      fminf(fmaxf(even * inv_scale, -kE4m3Max), kE4m3Max));
  bytes.y = e4m3_bits(
      fminf(fmaxf(odd * inv_scale, -kE4m3Max), kE4m3Max));
  store_quant_key(page_base, mirror, rows_per_page, row, d0, bytes,
                  exp2f(exponent));
#endif
}

// ---------------------------------------------------------------------------
// A3: index-query RoPE
// ---------------------------------------------------------------------------

// BF16 query, rotated in place. Only the RoPE band is touched; the NoPE half
// is passed through, so the block is half as wide as the FP8 kernel's.
// ``weights_out`` may alias ``weights_in``: each element is read and written by
// exactly one thread.
__global__ void v41_indexer_q_rope_kernel(
    __nv_bfloat16* __restrict__ q, int64_t q_token_stride,
    int64_t q_head_stride, const void* __restrict__ positions,
    bool positions_are_int64, const float* __restrict__ cos_sin_cache,
    int64_t cos_sin_stride, const __nv_bfloat16* __restrict__ weights_in,
    __nv_bfloat16* __restrict__ weights_out, int64_t weights_stride,
    float weights_scale) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int d0 = kIndexNopeDim + threadIdx.x * kRopePacked;

  __nv_bfloat16* const row =
      q + token * q_token_stride + head * q_head_stride + d0;
  const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(row);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  // The query has no slot to gate on, so graph-capture padding reaches this
  // kernel with whatever position the runner left behind. Clamping keeps a
  // negative one from indexing before the table; the row is discarded anyway.
  const int64_t raw_pos = load_index(positions, positions_are_int64, token);
  const int64_t pos = raw_pos < 0 ? 0 : raw_pos;
  rope_pair_at(even, odd, cos_sin_cache + pos * cos_sin_stride,
               (d0 - kIndexNopeDim) / 2, kIndexRopeDim);

  __nv_bfloat162 out;
  out.x = __float2bfloat16_rn(even);
  out.y = __float2bfloat16_rn(odd);
  *reinterpret_cast<__nv_bfloat162*>(row) = out;

  if (threadIdx.x == 0) {
    const int64_t w = token * weights_stride + head;
    weights_out[w] =
        __float2bfloat16_rn(__bfloat162float(weights_in[w]) * weights_scale);
  }
}

#ifdef ILUVATAR_HAS_FP8_FP4
// FP8 query: rotate, pick one power-of-two scale for the whole 128-wide row,
// write E4M3, and fold that scale into the weights. The downstream FP8
// mqa-logits kernel takes no companion scale tensor -- it multiplies the
// per-row q_scale in through ``weights``, which is why it lands there.
__global__ void v41_indexer_q_rope_fp8_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t q_token_stride,
    int64_t q_head_stride, const void* __restrict__ positions,
    bool positions_are_int64, const float* __restrict__ cos_sin_cache,
    int64_t cos_sin_stride, const __nv_bfloat16* __restrict__ weights_in,
    int64_t weights_in_stride, uint8_t* __restrict__ q_fp8,
    float* __restrict__ weights_out, int64_t weights_out_stride,
    float weights_scale) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int num_heads = gridDim.y;
  const int d0 = threadIdx.x * kRopePacked;

  const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(
      q + token * q_token_stride + head * q_head_stride + d0);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  if (d0 >= kIndexNopeDim) {
    // Clamped for the same reason as the bf16 kernel: padding rows carry no
    // slot to gate on and must not index before the table.
    const int64_t raw_pos = load_index(positions, positions_are_int64, token);
    const int64_t pos = raw_pos < 0 ? 0 : raw_pos;
    rope_pair_at(even, odd, cos_sin_cache + pos * cos_sin_stride,
                 (d0 - kIndexNopeDim) / 2, kIndexRopeDim);
    // The reference rounds the rotated pair to bf16 before the absmax; the
    // NoPE half is already exactly bf16, having come straight off the tensor.
    even = bf16_round(even);
    odd = bf16_round(odd);
  }

  const float amax = fmaxf(
      warp_reduce_max(fmaxf(fabsf(even), fabsf(odd)), kIndexThreads), 1e-4f);
  const float q_scale = exp2f(ceilf(log2f(amax / kE4m3Max)));

  uchar2 bytes;
  bytes.x = e4m3_bits(__fdiv_rn(even, q_scale));
  bytes.y = e4m3_bits(__fdiv_rn(odd, q_scale));
  *reinterpret_cast<uchar2*>(
      q_fp8 + (token * num_heads + head) * kIndexHeadDim + d0) = bytes;

  if (d0 == 0) {
    weights_out[token * weights_out_stride + head] =
        __bfloat162float(weights_in[token * weights_in_stride + head]) *
        q_scale * weights_scale;
  }
}
#endif  // ILUVATAR_HAS_FP8_FP4

// INT8 query. Same shape as the FP8 kernel -- rotate, reduce one scale over the
// whole 128-wide row, store, fold that scale into the weights -- and for the
// same reason: ixformer's INT8 paged mqa-logits kernel takes no companion scale
// tensor either. Two things differ, and both come from the reference
// (``ds4_fused_indexer_q_rope.cu``, whose folding this reproduces):
//
//   * the scale is the plain ``amax / 127`` ratio, not a power of two, so the
//     quantization error is spread rather than biased low;
//   * the rotated pair is not rounded to bf16 before the amax. The FP8 kernel
//     rounds because its reference stores an intermediate bf16 row; this one's
//     reference carries fp32 from the rotation straight into the quantizer.
//
// No FP8 hardware is involved, so unlike the kernel above this one builds and
// runs everywhere.
__global__ void v41_indexer_q_rope_int8_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t q_token_stride,
    int64_t q_head_stride, const void* __restrict__ positions,
    bool positions_are_int64, const float* __restrict__ cos_sin_cache,
    int64_t cos_sin_stride, const __nv_bfloat16* __restrict__ weights_in,
    int64_t weights_in_stride, int8_t* __restrict__ q_int8,
    float* __restrict__ weights_out, int64_t weights_out_stride,
    float weights_scale) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int num_heads = gridDim.y;
  const int d0 = threadIdx.x * kRopePacked;

  const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(
      q + token * q_token_stride + head * q_head_stride + d0);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  if (d0 >= kIndexNopeDim) {
    // Clamped for the same reason as the kernels above: graph-capture padding
    // carries no slot to gate on and must not index before the table.
    const int64_t raw_pos = load_index(positions, positions_are_int64, token);
    const int64_t pos = raw_pos < 0 ? 0 : raw_pos;
    rope_pair_at(even, odd, cos_sin_cache + pos * cos_sin_stride,
                 (d0 - kIndexNopeDim) / 2, kIndexRopeDim);
  }

  const float amax = fmaxf(
      warp_reduce_max(fmaxf(fabsf(even), fabsf(odd)), kIndexThreads),
      kInt8AmaxEps);
  const float q_scale = amax / kInt8AbsMax;
  // ``x * (127 / amax)``, never ``x / (amax / 127)``: the reference multiplies,
  // and on this device the two disagree at values that land halfway between
  // neighbouring integers. Clamp after the round, not before.
  const float inv_scale = kInt8AbsMax / amax;

  constexpr int kInt8Limit = static_cast<int>(kInt8AbsMax);
  char2 bytes;
  bytes.x = static_cast<int8_t>(
      max(-kInt8Limit, min(kInt8Limit, __float2int_rn(even * inv_scale))));
  bytes.y = static_cast<int8_t>(
      max(-kInt8Limit, min(kInt8Limit, __float2int_rn(odd * inv_scale))));
  *reinterpret_cast<char2*>(
      q_int8 + (token * num_heads + head) * kIndexHeadDim + d0) = bytes;

  if (d0 == 0) {
    weights_out[token * weights_out_stride + head] =
        __bfloat162float(weights_in[token * weights_in_stride + head]) *
        q_scale * weights_scale;
  }
}

// ---------------------------------------------------------------------------
// A4: the short-context bypass
// ---------------------------------------------------------------------------

// When a layer's whole compressed context fits in ``topk``, the selection is
// forced: every group is picked whatever it scores, so the q projection, RoPE,
// quantization and paged scoring can all be skipped and the top-k written
// outright. Mirrors the reference's ``_fill_short_context_topk_indices``, with
// the companion ``topk_lens`` this platform's sparse kernels take.
//
// One block per token, striding over the row: V4.1's topk is 512.
//
// The row is pure output, so it goes out two columns at a time in 8-byte
// stores. Not four: a whole row is only 512 columns, and at 16 bytes a lane
// the grid shrinks to 192 warps, which leaves nothing to hide store latency
// behind. 8 bytes halves the store count and keeps the warps. ``pairs`` is
// zero when the row stride cannot carry an aligned ``int2``, and the scalar
// loop below then covers the whole row.
//
// ``compress_shift`` is the log2 of ``compress_ratio`` (1 or 2 on V4.1), or
// -1; the shift keeps a 64-bit software divide out of the kernel.
__global__ void v41_fill_short_context_topk_kernel(
    int32_t* __restrict__ topk_idxs, int64_t idxs_stride,
    int32_t* __restrict__ topk_lens, const void* __restrict__ positions,
    bool positions_are_int64, int topk, int pairs, int compress_ratio,
    int compress_shift) {
  const int64_t token = blockIdx.x;
  const int64_t pos = load_index(positions, positions_are_int64, token);
  // A token sees this many groups under causality. A negative position is
  // graph-capture padding and yields an empty row.
  const int64_t groups =
      pos < 0 ? 0
              : (compress_shift >= 0 ? ((pos + 1) >> compress_shift)
                                     : (pos + 1) / compress_ratio);
  int32_t* const row = topk_idxs + token * idxs_stride;

  for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const int j = pair * 2;
    int2 packed;
    packed.x = j + 0 < groups ? j + 0 : -1;
    packed.y = j + 1 < groups ? j + 1 : -1;
    *reinterpret_cast<int2*>(row + j) = packed;
  }
  for (int j = pairs * 2 + threadIdx.x; j < topk; j += blockDim.x) {
    row[j] = j < groups ? j : -1;
  }
  if (threadIdx.x == 0) {
    topk_lens[token] =
        static_cast<int32_t>(groups < topk ? groups : topk);
  }
}

void check_index_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kLong || t.scalar_type() == at::kInt,
              name, " must be int32 or int64");
}

void check_slot_tensor(const torch::Tensor& t, const char* name) {
  check_index_tensor(t, name);
  TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64");
}

void check_cos_sin(const torch::Tensor& cos_sin_cache) {
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.scalar_type() == at::kFloat &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be a 2D fp32 CUDA tensor, last dim "
              "contiguous");
  TORCH_CHECK(cos_sin_cache.size(1) >= kIndexRopeDim,
              "cos_sin_cache must be at least ", kIndexRopeDim, " wide");
}

// [T, H, 128] bf16, head rows contiguous.
void check_index_q(const torch::Tensor& q) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3, "q must be a 3D CUDA tensor");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(q.size(2) == kIndexHeadDim, "index head_dim must be ",
              kIndexHeadDim, ", got ", q.size(2));
  TORCH_CHECK(q.stride(2) == 1, "q head rows must be contiguous");
}

}  // namespace

void v41_indexer_k_norm_rope_store(torch::Tensor& k_pre,
                                   torch::Tensor& positions,
                                   torch::Tensor& cos_sin_cache,
                                   torch::Tensor& rms_norm_weight,
                                   double rms_norm_eps, torch::Tensor& k_cache,
                                   torch::Tensor& kv_slot_mapping,
                                   int64_t compress_ratio,
                                   const std::string& record,
                                   const std::optional<torch::Tensor>& row_out) {
  const IndexRecord kind = record == "bf16"   ? IndexRecord::kBf16
                           : record == "fp8"  ? IndexRecord::kFp8
                           : record == "int8" ? IndexRecord::kInt8
                                              : IndexRecord::kBf16;
  TORCH_CHECK(record == "bf16" || record == "fp8" || record == "int8",
              "record must be 'bf16', 'fp8' or 'int8', got '", record, "'");
  TORCH_CHECK(compress_ratio == 1 || compress_ratio == 2,
              "compress_ratio must be 1 or 2, got ", compress_ratio);
  TORCH_CHECK(k_pre.is_cuda() && k_pre.dim() == 2 &&
                  k_pre.scalar_type() == at::kBFloat16,
              "k_pre must be a 2D bf16 CUDA tensor");
  TORCH_CHECK(k_pre.size(1) == kIndexHeadDim && k_pre.stride(1) == 1,
              "k_pre must be [T, 128] with a contiguous last dim");
  TORCH_CHECK(rms_norm_weight.is_cuda() && rms_norm_weight.is_contiguous() &&
                  rms_norm_weight.numel() == kIndexHeadDim &&
                  rms_norm_weight.scalar_type() == at::kBFloat16,
              "rms_norm_weight must be a contiguous bf16 [128] CUDA tensor");
  TORCH_CHECK(k_cache.is_cuda() && k_cache.dim() == 3,
              "k_cache must be a 3D CUDA tensor [pages, rows, width]");
  check_cos_sin(cos_sin_cache);
  check_index_tensor(positions, "positions");
  check_slot_tensor(kv_slot_mapping, "kv_slot_mapping");

  const int64_t num_tokens = kv_slot_mapping.numel();
  TORCH_CHECK(num_tokens <= k_pre.size(0) && num_tokens <= positions.numel(),
              "kv_slot_mapping is longer than k_pre / positions");
  if (num_tokens == 0) {
    return;
  }

  if (kind == IndexRecord::kBf16) {
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16,
                "a bf16 indexer cache is bfloat16");
    TORCH_CHECK(k_cache.size(2) == kIndexHeadDim && k_cache.stride(2) == 1,
                "a bf16 indexer row is 128 wide and contiguous");
  } else {
    TORCH_CHECK(k_cache.scalar_type() == at::kByte, "a ", record,
                " indexer cache is uint8");
    TORCH_CHECK(k_cache.size(2) == kIndexHeadDim + kIndexScaleBytes, "a ",
                record, " indexer row is ",
                kIndexHeadDim + kIndexScaleBytes,
                " bytes (128 value bytes + one fp32 scale), got ",
                k_cache.size(2));
    TORCH_CHECK(k_cache.stride(2) == 1 &&
                    k_cache.stride(1) == kIndexHeadDim + kIndexScaleBytes,
                "the ", record, " indexer page must be internally contiguous");
#ifndef ILUVATAR_HAS_FP8_FP4
    TORCH_CHECK(kind != IndexRecord::kFp8,
                "the fp8 indexer cache needs an ivcore30 build");
#endif
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(k_pre));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned grid = static_cast<unsigned>(num_tokens);
  const auto* k_pre_ptr =
      reinterpret_cast<const __nv_bfloat16*>(k_pre.data_ptr());
  const auto* weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(rms_norm_weight.data_ptr());
  const bool positions_are_int64 = positions.scalar_type() == at::kLong;
  const bool slots_are_int64 = kv_slot_mapping.scalar_type() == at::kLong;

  // ``row_out`` mirrors the key in token order for the next pipeline rank.
  void* row_ptr = nullptr;
  if (row_out.has_value()) {
    const torch::Tensor& rows = *row_out;
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous(),
                "row_out must be a 2D contiguous CUDA tensor");
    TORCH_CHECK(rows.size(0) >= num_tokens,
                "row_out is shorter than the token count");
    TORCH_CHECK(rows.scalar_type() == k_cache.scalar_type(),
                "row_out must share the indexer cache dtype");
    TORCH_CHECK(rows.size(1) == k_cache.size(2),
                "row_out rows are ", rows.size(1), " wide but the indexer "
                "record is ", k_cache.size(2));
    row_ptr = rows.data_ptr();
  }

#define VLLM_ILU_LAUNCH_K_STORE(RECORD)                                     \
  v41_indexer_k_norm_rope_store_kernel<RECORD>                              \
      <<<grid, kIndexThreads, 0, stream>>>(                                 \
          k_pre_ptr, k_pre.stride(0), positions.data_ptr(),                 \
          positions_are_int64, weight_ptr,                                  \
          static_cast<float>(rms_norm_eps), cos_sin_cache.data_ptr<float>(), \
          cos_sin_cache.stride(0), k_cache.data_ptr(), k_cache.stride(0),   \
          k_cache.stride(1), static_cast<int>(k_cache.size(1)),             \
          k_cache.size(0), kv_slot_mapping.data_ptr(), slots_are_int64,     \
          static_cast<int>(compress_ratio), row_ptr)

  switch (kind) {
    case IndexRecord::kBf16:
      VLLM_ILU_LAUNCH_K_STORE(IndexRecord::kBf16);
      break;
    case IndexRecord::kFp8:
      VLLM_ILU_LAUNCH_K_STORE(IndexRecord::kFp8);
      break;
    case IndexRecord::kInt8:
      VLLM_ILU_LAUNCH_K_STORE(IndexRecord::kInt8);
      break;
  }
#undef VLLM_ILU_LAUNCH_K_STORE
}

void v41_indexer_q_rope(torch::Tensor& q, torch::Tensor& positions,
                        torch::Tensor& cos_sin_cache,
                        torch::Tensor& weights_in, torch::Tensor& weights_out,
                        double weights_scale) {
  check_index_q(q);
  check_cos_sin(cos_sin_cache);
  check_index_tensor(positions, "positions");
  TORCH_CHECK(weights_in.is_cuda() && weights_in.dim() == 2 &&
                  weights_in.scalar_type() == at::kBFloat16,
              "weights_in must be a 2D bf16 CUDA tensor");
  TORCH_CHECK(weights_out.is_cuda() && weights_out.dim() == 2 &&
                  weights_out.scalar_type() == at::kBFloat16,
              "weights_out must be a 2D bf16 CUDA tensor");
  TORCH_CHECK(weights_in.sizes() == weights_out.sizes() &&
                  weights_in.strides() == weights_out.strides() &&
                  weights_in.stride(1) == 1,
              "weights_in / weights_out must match in shape and stride and "
              "have a contiguous last dim");
  TORCH_CHECK(weights_in.size(0) == q.size(0) &&
                  weights_in.size(1) == q.size(1),
              "weights must be [T, H] matching q");
  TORCH_CHECK(q.size(0) <= positions.numel(),
              "positions is shorter than q's token count");
  if (q.size(0) == 0 || q.size(1) == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(q.size(0)),
                  static_cast<unsigned>(q.size(1)));
  v41_indexer_q_rope_kernel<<<grid, kIndexRopeThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()), q.stride(0), q.stride(1),
      positions.data_ptr(), positions.scalar_type() == at::kLong,
      cos_sin_cache.data_ptr<float>(), cos_sin_cache.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(weights_in.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(weights_out.data_ptr()),
      weights_in.stride(0), static_cast<float>(weights_scale));
}

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_fp8(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale) {
  check_index_q(q);
  check_cos_sin(cos_sin_cache);
  check_index_tensor(positions, "positions");
  TORCH_CHECK(weights_in.is_cuda() && weights_in.dim() == 2 &&
                  weights_in.scalar_type() == at::kBFloat16 &&
                  weights_in.stride(1) == 1,
              "weights_in must be a 2D bf16 CUDA tensor with a contiguous "
              "last dim");
  TORCH_CHECK(weights_in.size(0) == q.size(0) &&
                  weights_in.size(1) == q.size(1),
              "weights_in must be [T, H] matching q");
  TORCH_CHECK(q.size(0) <= positions.numel(),
              "positions is shorter than q's token count");

  auto q_fp8 = torch::empty(q.sizes(),
                            q.options().dtype(torch::kFloat8_e4m3fn));
  auto weights_out =
      torch::empty(weights_in.sizes(), weights_in.options().dtype(
                                           torch::kFloat32));
  if (q.size(0) == 0 || q.size(1) == 0) {
    return {q_fp8, weights_out};
  }
#ifndef ILUVATAR_HAS_FP8_FP4
  TORCH_CHECK(false, "the fp8 indexer query needs an ivcore30 build");
#else
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(q.size(0)),
                  static_cast<unsigned>(q.size(1)));
  v41_indexer_q_rope_fp8_kernel<<<grid, kIndexThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), q.stride(0),
      q.stride(1), positions.data_ptr(),
      positions.scalar_type() == at::kLong, cos_sin_cache.data_ptr<float>(),
      cos_sin_cache.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(weights_in.data_ptr()),
      weights_in.stride(0), reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
      weights_out.data_ptr<float>(), weights_out.stride(0),
      static_cast<float>(weights_scale));
#endif
  return {q_fp8, weights_out};
}

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_fp8_meta(
    torch::Tensor& q, torch::Tensor& /*positions*/,
    torch::Tensor& /*cos_sin_cache*/, torch::Tensor& weights_in,
    double /*weights_scale*/) {
  return {torch::empty(q.sizes(), q.options().dtype(torch::kFloat8_e4m3fn)),
          torch::empty(weights_in.sizes(),
                       weights_in.options().dtype(torch::kFloat32))};
}

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_int8(
    torch::Tensor& q, torch::Tensor& positions, torch::Tensor& cos_sin_cache,
    torch::Tensor& weights_in, double weights_scale) {
  check_index_q(q);
  check_cos_sin(cos_sin_cache);
  check_index_tensor(positions, "positions");
  TORCH_CHECK(weights_in.is_cuda() && weights_in.dim() == 2 &&
                  weights_in.scalar_type() == at::kBFloat16 &&
                  weights_in.stride(1) == 1,
              "weights_in must be a 2D bf16 CUDA tensor with a contiguous "
              "last dim");
  TORCH_CHECK(weights_in.size(0) == q.size(0) &&
                  weights_in.size(1) == q.size(1),
              "weights_in must be [T, H] matching q");
  TORCH_CHECK(q.size(0) <= positions.numel(),
              "positions is shorter than q's token count");

  auto q_int8 = torch::empty(q.sizes(), q.options().dtype(torch::kChar));
  auto weights_out =
      torch::empty(weights_in.sizes(),
                   weights_in.options().dtype(torch::kFloat32));
  if (q.size(0) == 0 || q.size(1) == 0) {
    return {q_int8, weights_out};
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(q.size(0)),
                  static_cast<unsigned>(q.size(1)));
  v41_indexer_q_rope_int8_kernel<<<grid, kIndexThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), q.stride(0),
      q.stride(1), positions.data_ptr(),
      positions.scalar_type() == at::kLong, cos_sin_cache.data_ptr<float>(),
      cos_sin_cache.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(weights_in.data_ptr()),
      weights_in.stride(0), reinterpret_cast<int8_t*>(q_int8.data_ptr()),
      weights_out.data_ptr<float>(), weights_out.stride(0),
      static_cast<float>(weights_scale));
  return {q_int8, weights_out};
}

std::tuple<torch::Tensor, torch::Tensor> v41_indexer_q_rope_int8_meta(
    torch::Tensor& q, torch::Tensor& /*positions*/,
    torch::Tensor& /*cos_sin_cache*/, torch::Tensor& weights_in,
    double /*weights_scale*/) {
  return {torch::empty(q.sizes(), q.options().dtype(torch::kChar)),
          torch::empty(weights_in.sizes(),
                       weights_in.options().dtype(torch::kFloat32))};
}

void v41_fill_short_context_topk(torch::Tensor& topk_idxs,
                                 torch::Tensor& topk_lens,
                                 torch::Tensor& positions,
                                 int64_t compress_ratio) {
  TORCH_CHECK(compress_ratio == 1 || compress_ratio == 2,
              "compress_ratio must be 1 or 2, got ", compress_ratio);
  TORCH_CHECK(topk_idxs.is_cuda() && topk_idxs.dim() == 2 &&
                  topk_idxs.scalar_type() == at::kInt &&
                  topk_idxs.stride(1) == 1,
              "topk_idxs must be a 2D int32 CUDA tensor [T, topk] with a "
              "contiguous last dim");
  TORCH_CHECK(topk_lens.is_cuda() && topk_lens.dim() == 1 &&
                  topk_lens.is_contiguous() &&
                  topk_lens.scalar_type() == at::kInt,
              "topk_lens must be a contiguous 1D int32 CUDA tensor");
  check_index_tensor(positions, "positions");

  const int64_t num_tokens = topk_idxs.size(0);
  TORCH_CHECK(topk_lens.numel() >= num_tokens &&
                  positions.numel() >= num_tokens,
              "topk_lens / positions are shorter than topk_idxs' token count");
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_idxs));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // 256 lanes of int2 cover V4.1's 512-wide row in one store apiece.
  constexpr unsigned kFillThreads = 256;
  const int topk = static_cast<int>(topk_idxs.size(1));
  const bool wide = topk_idxs.stride(0) % 2 == 0;
  int compress_shift = -1;
  for (int shift = 0; shift < 31; ++shift) {
    if ((int64_t{1} << shift) == compress_ratio) {
      compress_shift = shift;
      break;
    }
  }
  v41_fill_short_context_topk_kernel<<<static_cast<unsigned>(num_tokens),
                                      kFillThreads, 0, stream>>>(
      topk_idxs.data_ptr<int32_t>(), topk_idxs.stride(0),
      topk_lens.data_ptr<int32_t>(), positions.data_ptr(),
      positions.scalar_type() == at::kLong, topk, wide ? topk / 2 : 0,
      static_cast<int>(compress_ratio), compress_shift);
}

}  // namespace vllm_iluvatar
