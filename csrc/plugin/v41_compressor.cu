// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 compressor ops:
//   v41_compress_norm_cr1 -- ratio-1 per-token RMSNorm into the BF16 latent
//   v41_rope_insert       -- GPT-J RoPE + publish a latent into the compressed
//                            KV cache (packed fp8_ds_mla, plain bf16, plain fp8)
//
// Both mirror ``off/vllm/vllm/models/deepseek_v41/common/ops/
// fused_compress_quant_cache.py``: ``_store_latent`` in the ratio-1 degenerate
// case, and ``rope_quant_insert`` / ``_rope_plain_insert_kernel``.
//
// Slots are compressed-domain global row indices (``get_compressed_slot_mapping``
// emits ``block_number * rows_per_page + row``), so the page and the row inside
// it come out of the slot alone -- no block table, exactly like the reference.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "ops.h"
#include "utils.cuh"
#include "v41_common.cuh"

namespace vllm_iluvatar {
namespace {

using namespace v41;

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
// One thread per GPT-J pair covers a 512-wide row with four 64-lane warps.
constexpr int kLatentThreads = kLatentDim / kRopePacked;

// ---------------------------------------------------------------------------
// B1: ratio-1 latent RMSNorm
// ---------------------------------------------------------------------------

// One block per token. Ratio 1 pools nothing and carries no state, so the
// latent is just the normalized row; ``slot_mapping`` only marks valid tokens
// (the reference leaves an invalid row untouched, and both consumers skip it).
__global__ void v41_compress_norm_cr1_kernel(
    const float* __restrict__ kv_score, int64_t row_stride,
    const void* __restrict__ slot_mapping, bool slots_are_int64,
    const __nv_bfloat16* __restrict__ weight, float eps,
    __nv_bfloat16* __restrict__ latent_out, int64_t out_stride) {
  const int64_t token = blockIdx.x;
  if (load_index(slot_mapping, slots_are_int64, token) < 0) {
    return;
  }

  const int d0 = threadIdx.x * 2;
  const float2 row =
      *reinterpret_cast<const float2*>(kv_score + token * row_stride + d0);

  __shared__ float smem[kMaxThreadsPerBlock / warpSize];
  const float sum_sq =
      block_reduce_sum(row.x * row.x + row.y * row.y, smem);
  const float inv_rms =
      rsqrtf(sum_sq / static_cast<float>(kLatentDim) + eps);

  const __nv_bfloat162 w = *reinterpret_cast<const __nv_bfloat162*>(weight + d0);
  __nv_bfloat162 out;
  out.x = __float2bfloat16_rn(row.x * inv_rms * __bfloat162float(w.x));
  out.y = __float2bfloat16_rn(row.y * inv_rms * __bfloat162float(w.y));
  *reinterpret_cast<__nv_bfloat162*>(latent_out + token * out_stride + d0) = out;
}

// ---------------------------------------------------------------------------
// B2: RoPE + compressed-cache insert
// ---------------------------------------------------------------------------

// Shared prologue: decide whether this token publishes, and where.
// Returns false for the whole block, so the shuffles in the packed kernel never
// lose a partner.
__device__ __forceinline__ bool resolve_slot(
    const void* positions, bool positions_are_int64, const void* slot_mapping,
    bool slots_are_int64, int64_t token, int compress_ratio, int rows_per_page,
    int64_t num_pages, int64_t& page, int64_t& row, int64_t& rope_pos) {
  const int64_t pos = load_index(positions, positions_are_int64, token);
  const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
  if (slot < 0 || (pos + 1) % compress_ratio != 0) {
    return false;
  }
  page = slot / rows_per_page;
  if (page >= num_pages) {
    return false;
  }
  row = slot % rows_per_page;
  // A compressed row stands for a whole group; it is rotated against the
  // group's first raw position.
  rope_pos = (pos / compress_ratio) * compress_ratio;
  return true;
}

// Packed fp8_ds_mla record: 448 E4M3 NoPE bytes, 64 BF16 RoPE values and 7
// UE8M0 group scales, segregated inside the page (all rows' data, then all
// rows' scales).
//
// Threads own a GPT-J pair, so a 64-value quantization group is 32 of them.
// That places every group inside one warp and puts the NoPE/RoPE split on a
// group boundary (448 = 7 groups = 224 threads), so the amax reduction is five
// shuffles with no barrier and no lane of one group ever reads a lane of
// another. Nothing returns early past this point for the same reason.
__global__ void v41_rope_insert_packed_kernel(
    const __nv_bfloat16* __restrict__ latent, int64_t latent_stride,
    const void* __restrict__ positions, bool positions_are_int64,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ cache, int64_t page_stride, int rows_per_page,
    int64_t num_pages, const void* __restrict__ slot_mapping,
    bool slots_are_int64, int compress_ratio,
    uint8_t* __restrict__ row_out) {
  const int64_t token = blockIdx.x;
  int64_t page, row, rope_pos;
  if (!resolve_slot(positions, positions_are_int64, slot_mapping,
                    slots_are_int64, token, compress_ratio, rows_per_page,
                    num_pages, page, row, rope_pos)) {
    return;
  }

  const int d0 = threadIdx.x * kRopePacked;
  const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(
      latent + token * latent_stride + d0);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  constexpr int kRopeStart = kLatentDim - kRopeDim;
  const bool rotates = d0 >= kRopeStart;
  if (rotates) {
    rope_pair_at(even, odd, cos_sin_cache + rope_pos * cos_sin_stride,
                 (d0 - kRopeStart) / 2, kRopeDim);
  }

  // Reduced by every thread, used by the NoPE ones. The RoPE threads occupy
  // lanes the NoPE groups never name, so their amax joins no group but theirs.
  const float amax = warp_reduce_max(fmaxf(fabsf(even), fabsf(odd)),
                                     kFp8QuantGroup / kRopePacked);
  (void)amax;

  uint8_t* const data = cache + page * page_stride + row * kFp8TokenDataBytes;
  // The pipeline replica, when this rank has to hand the record to the next
  // one. Same bytes, addressed by token instead of by slot, so the record is
  // written once and read twice rather than recomputed downstream.
  uint8_t* const mirror =
      row_out == nullptr ? nullptr : row_out + token * kFp8RecordBytes;

  if (rotates) {
    // The record keeps the RoPE tail in BF16, so the rotation is stored as it
    // comes out rather than pushed through a quantization grid.
    __nv_bfloat162 out;
    out.x = __float2bfloat16_rn(even);
    out.y = __float2bfloat16_rn(odd);
    const int rope_byte = kFp8NopeBytes + (d0 - kRopeStart) * 2;
    *reinterpret_cast<__nv_bfloat162*>(data + rope_byte) = out;
    if (mirror != nullptr) {
      *reinterpret_cast<__nv_bfloat162*>(mirror + rope_byte) = out;
    }
    return;
  }

#ifdef ILUVATAR_HAS_FP8_FP4
  const uint8_t scale_bits = ue8m0_bits(amax);
  const float inv_scale = 1.0f / ue8m0_scale(scale_bits);
  uchar2 bytes;
  bytes.x = e4m3_bits(even * inv_scale);
  bytes.y = e4m3_bits(odd * inv_scale);
  *reinterpret_cast<uchar2*>(data + d0) = bytes;
  if (mirror != nullptr) {
    *reinterpret_cast<uchar2*>(mirror + d0) = bytes;
  }

  if (d0 % kFp8QuantGroup == 0) {
    uint8_t* const scales = cache + page * page_stride +
                            static_cast<int64_t>(rows_per_page) *
                                kFp8TokenDataBytes +
                            row * kFp8ScaleBytes;
    scales[d0 / kFp8QuantGroup] = scale_bits;
    // The eighth byte exists to make the region a multiple of eight; the
    // readers skip it, but a page of uninitialised memory should not carry it.
    if (d0 == 0) scales[kFp8NumScales] = 0;
    if (mirror != nullptr) {
      mirror[kFp8TokenDataBytes + d0 / kFp8QuantGroup] = scale_bits;
      if (d0 == 0) mirror[kFp8TokenDataBytes + kFp8NumScales] = 0;
    }
  }
#endif
}

// Plain [448 NoPE | 64 RoPE] bf16 row, the reference's
// ``_rope_plain_insert_kernel`` without its per-tensor-fp8 variant: this
// platform's two main-KV records are the packed 584-byte one and this.
__global__ void v41_rope_insert_plain_kernel(
    const __nv_bfloat16* __restrict__ latent, int64_t latent_stride,
    const void* __restrict__ positions, bool positions_are_int64,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    __nv_bfloat16* __restrict__ cache, int64_t page_stride, int64_t row_stride,
    int rows_per_page, int64_t num_pages,
    const void* __restrict__ slot_mapping, bool slots_are_int64,
    int compress_ratio, __nv_bfloat16* __restrict__ row_out) {
  const int64_t token = blockIdx.x;
  int64_t page, row, rope_pos;
  if (!resolve_slot(positions, positions_are_int64, slot_mapping,
                    slots_are_int64, token, compress_ratio, rows_per_page,
                    num_pages, page, row, rope_pos)) {
    return;
  }

  const int d0 = threadIdx.x * kRopePacked;
  const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(
      latent + token * latent_stride + d0);
  float even = __bfloat162float(in.x);
  float odd = __bfloat162float(in.y);

  constexpr int kRopeStart = kLatentDim - kRopeDim;
  if (d0 >= kRopeStart) {
    rope_pair_at(even, odd, cos_sin_cache + rope_pos * cos_sin_stride,
                 (d0 - kRopeStart) / 2, kRopeDim);
  }

  __nv_bfloat162 out;
  out.x = __float2bfloat16_rn(even);
  out.y = __float2bfloat16_rn(odd);
  *reinterpret_cast<__nv_bfloat162*>(
      cache + page * page_stride + row * row_stride + d0) = out;
  if (row_out != nullptr) {
    *reinterpret_cast<__nv_bfloat162*>(row_out + token * kLatentDim + d0) = out;
  }
}

void check_index_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kLong || t.scalar_type() == at::kInt,
              name, " must be int32 or int64");
}

}  // namespace

void v41_compress_norm_cr1(torch::Tensor& kv_score,
                           torch::Tensor& slot_mapping,
                           torch::Tensor& rms_norm_weight, double rms_norm_eps,
                           torch::Tensor& latent_out) {
  TORCH_CHECK(kv_score.is_cuda() && kv_score.dim() == 2,
              "kv_score must be a 2D CUDA tensor");
  TORCH_CHECK(kv_score.scalar_type() == at::kFloat, "kv_score must be float32");
  TORCH_CHECK(kv_score.size(1) == kLatentDim && kv_score.stride(1) == 1,
              "kv_score must be [T, 512] with a contiguous last dim");
  TORCH_CHECK(kv_score.stride(0) % 2 == 0,
              "kv_score rows must be 8-byte aligned for the float2 load");
  TORCH_CHECK(latent_out.is_cuda() && latent_out.dim() == 2 &&
                  latent_out.scalar_type() == at::kBFloat16,
              "latent_out must be a 2D bf16 CUDA tensor");
  TORCH_CHECK(latent_out.size(1) == kLatentDim && latent_out.stride(1) == 1,
              "latent_out must be [T, 512] with a contiguous last dim");
  TORCH_CHECK(rms_norm_weight.is_cuda() && rms_norm_weight.is_contiguous() &&
                  rms_norm_weight.numel() == kLatentDim &&
                  rms_norm_weight.scalar_type() == at::kBFloat16,
              "rms_norm_weight must be a contiguous bf16 [512] CUDA tensor");
  check_index_tensor(slot_mapping, "slot_mapping");

  const int64_t num_tokens = slot_mapping.numel();
  TORCH_CHECK(num_tokens <= kv_score.size(0) &&
                  num_tokens <= latent_out.size(0),
              "slot_mapping is longer than kv_score / latent_out");
  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_score));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  v41_compress_norm_cr1_kernel<<<static_cast<unsigned>(num_tokens),
                                 kLatentThreads, 0, stream>>>(
      kv_score.data_ptr<float>(), kv_score.stride(0), slot_mapping.data_ptr(),
      slot_mapping.scalar_type() == at::kLong,
      reinterpret_cast<const __nv_bfloat16*>(rms_norm_weight.data_ptr()),
      static_cast<float>(rms_norm_eps),
      reinterpret_cast<__nv_bfloat16*>(latent_out.data_ptr()),
      latent_out.stride(0));
}

void v41_rope_insert(torch::Tensor& latent, torch::Tensor& positions,
                     torch::Tensor& cos_sin_cache, torch::Tensor& kv_cache,
                     torch::Tensor& slot_mapping, int64_t compress_ratio,
                     const std::optional<torch::Tensor>& row_out) {
  TORCH_CHECK(compress_ratio == 1 || compress_ratio == 2,
              "compress_ratio must be 1 or 2, got ", compress_ratio);
  TORCH_CHECK(latent.is_cuda() && latent.dim() == 2 &&
                  latent.scalar_type() == at::kBFloat16,
              "latent must be a 2D bf16 CUDA tensor");
  TORCH_CHECK(latent.size(1) == kLatentDim && latent.stride(1) == 1,
              "latent must be [T, 512] with a contiguous last dim");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.scalar_type() == at::kFloat &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be a 2D fp32 CUDA tensor, last dim "
              "contiguous");
  TORCH_CHECK(cos_sin_cache.size(1) >= kRopeDim,
              "cos_sin_cache must be at least 64 wide");
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dim() == 3,
              "kv_cache must be a 3D CUDA tensor [pages, rows, width]");
  check_index_tensor(positions, "positions");
  check_index_tensor(slot_mapping, "slot_mapping");

  const int64_t num_tokens = slot_mapping.numel();
  TORCH_CHECK(num_tokens <= latent.size(0) && num_tokens <= positions.numel(),
              "slot_mapping is longer than latent / positions");
  if (num_tokens == 0) {
    return;
  }

  const int64_t num_pages = kv_cache.size(0);
  const int rows_per_page = static_cast<int>(kv_cache.size(1));
  const bool positions_are_int64 = positions.scalar_type() == at::kLong;
  const bool slots_are_int64 = slot_mapping.scalar_type() == at::kLong;

  // ``row_out`` mirrors the record in token order for the next pipeline rank.
  // Its width is the record's, not the cache's page geometry.
  void* row_ptr = nullptr;
  if (row_out.has_value()) {
    const torch::Tensor& rows = *row_out;
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous(),
                "row_out must be a 2D contiguous CUDA tensor");
    TORCH_CHECK(rows.size(0) >= num_tokens,
                "row_out is shorter than the token count");
    TORCH_CHECK(rows.scalar_type() == kv_cache.scalar_type(),
                "row_out must share the cache dtype");
    TORCH_CHECK(rows.size(1) == kv_cache.size(2),
                "row_out rows are ", rows.size(1), " wide but the cache "
                "record is ", kv_cache.size(2));
    row_ptr = rows.data_ptr();
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(latent));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned grid = static_cast<unsigned>(num_tokens);
  const auto* latent_ptr =
      reinterpret_cast<const __nv_bfloat16*>(latent.data_ptr());
  const float* cos_sin_ptr = cos_sin_cache.data_ptr<float>();

  if (kv_cache.scalar_type() == at::kByte) {
    TORCH_CHECK(kv_cache.size(2) == kFp8RecordBytes,
                "the packed record is ", kFp8RecordBytes, " bytes per row, "
                "got a cache of width ", kv_cache.size(2));
    TORCH_CHECK(kv_cache.stride(2) == 1 &&
                    kv_cache.stride(1) == kFp8RecordBytes,
                "the packed cache page must be internally contiguous");
#ifndef ILUVATAR_HAS_FP8_FP4
    TORCH_CHECK(false,
                "the packed fp8_ds_mla record needs an ivcore30 build");
#endif
    v41_rope_insert_packed_kernel<<<grid, kLatentThreads, 0, stream>>>(
        latent_ptr, latent.stride(0), positions.data_ptr(),
        positions_are_int64, cos_sin_ptr, cos_sin_cache.stride(0),
        reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache.stride(0),
        rows_per_page, num_pages, slot_mapping.data_ptr(), slots_are_int64,
        static_cast<int>(compress_ratio),
        reinterpret_cast<uint8_t*>(row_ptr));
    return;
  }

  TORCH_CHECK(kv_cache.scalar_type() == at::kBFloat16,
              "the compressed KV cache is either the packed uint8 record or a "
              "plain bf16 row; got ", kv_cache.scalar_type());
  TORCH_CHECK(kv_cache.size(2) == kLatentDim && kv_cache.stride(2) == 1,
              "a plain compressed cache row is 512 wide and contiguous");
  v41_rope_insert_plain_kernel<<<grid, kLatentThreads, 0, stream>>>(
      latent_ptr, latent.stride(0), positions.data_ptr(), positions_are_int64,
      cos_sin_ptr, cos_sin_cache.stride(0),
      reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr()),
      kv_cache.stride(0), kv_cache.stride(1), rows_per_page, num_pages,
      slot_mapping.data_ptr(), slots_are_int64,
      static_cast<int>(compress_ratio),
      reinterpret_cast<__nv_bfloat16*>(row_ptr));
}

}  // namespace vllm_iluvatar
