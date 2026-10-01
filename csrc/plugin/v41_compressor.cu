// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 compressor ops:
//   v41_compress_norm_cr1  -- ratio-1 per-token RMSNorm into the BF16 latent
//   v41_compress_pool_norm -- ratio-2 group pooling over the state ring, then
//                             the same RMSNorm
//   v41_rope_insert        -- GPT-J RoPE + publish a latent into the compressed
//                             KV cache (packed fp8_ds_mla, plain bf16, plain fp8)
//
// All three mirror ``off/vllm/vllm/models/deepseek_v41/common/ops/
// fused_compress_quant_cache.py``: ``_store_latent`` in the ratio-1 degenerate
// case, ``_fused_save_compress_norm_kernel``'s two program kinds, and
// ``rope_quant_insert`` / ``_rope_plain_insert_kernel``.
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
// The pooling op's group size. Ratio 1 pools nothing and has its own op.
constexpr int kPoolRatio = 2;

// ---------------------------------------------------------------------------
// B1: latent RMSNorm
// ---------------------------------------------------------------------------

// The reference's ``_store_latent``: normalize a row held two channels per
// thread and publish it as the bf16 latent. Every thread of the block has to
// call this, the variance being a block reduction.
__device__ __forceinline__ void store_latent(
    float even, float odd, int d0, const __nv_bfloat16* __restrict__ weight,
    float eps, __nv_bfloat16* __restrict__ latent_out, int64_t out_stride,
    int64_t token, float* smem) {
  const float sum_sq = block_reduce_sum(even * even + odd * odd, smem);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(kLatentDim) + eps);
  const __nv_bfloat162 w = *reinterpret_cast<const __nv_bfloat162*>(weight + d0);
  __nv_bfloat162 out;
  out.x = __float2bfloat16_rn(even * inv_rms * __bfloat162float(w.x));
  out.y = __float2bfloat16_rn(odd * inv_rms * __bfloat162float(w.y));
  *reinterpret_cast<__nv_bfloat162*>(latent_out + token * out_stride + d0) = out;
}

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

  const int d0 = threadIdx.x * kRopePacked;
  const float2 row =
      *reinterpret_cast<const float2*>(kv_score + token * row_stride + d0);

  __shared__ float smem[kMaxThreadsPerBlock / warpSize];
  store_latent(row.x, row.y, d0, weight, eps, latent_out, out_stride, token,
               smem);
}

// ---------------------------------------------------------------------------
// B1b: ratio-2 group pooling
// ---------------------------------------------------------------------------

// Softmax over a group's two score rows, per channel, then the weighted sum of
// the kv rows. Subtracting the peak is what keeps a large score out of the
// exponential's overflow; the algebra is the reference's ``tl.softmax``.
__device__ __forceinline__ float pool_channel(float kv_a, float score_a,
                                              float kv_b, float score_b) {
  const float peak = fmaxf(score_a, score_b);
  const float weight_a = expf(score_a - peak);
  const float weight_b = expf(score_b - peak);
  return (kv_a * weight_a + kv_b * weight_b) / (weight_a + weight_b);
}

// A state row is [kv | score], both ``kLatentDim`` wide.
__device__ __forceinline__ void pool_rows(const float* __restrict__ row_a,
                                          const float* __restrict__ row_b,
                                          int d0, float& even, float& odd) {
  const float2 kv_a = *reinterpret_cast<const float2*>(row_a + d0);
  const float2 score_a = *reinterpret_cast<const float2*>(row_a + kLatentDim + d0);
  const float2 kv_b = *reinterpret_cast<const float2*>(row_b + d0);
  const float2 score_b = *reinterpret_cast<const float2*>(row_b + kLatentDim + d0);
  even = pool_channel(kv_a.x, score_a.x, kv_b.x, score_b.x);
  odd = pool_channel(kv_a.y, score_a.y, kv_b.y, score_b.y);
}

// Ratio 2 keeps one ring block per request holding the open group's rows:
// position ``p`` lives in row ``p % capacity``, and the slot encodes
// ``block * capacity + p % capacity``.
//
// The grid is the reference's, one program per request followed by one per pair
// of packed tokens, and so is the division of labour. A request program pools
// the group the chunk's first token closes with its predecessor -- the only row
// that has to come out of the ring -- and then stores the chunk's last
// ``capacity`` rows, the only rows a later chunk can ask for. Both in the same
// program, so a ring row is never read and written at the same time. A pair
// program pools a group that ends inside its own pair and reads both rows from
// the raw input, so a mid-chunk group never touches the ring at all.
//
// That last point is the reason this kernel exists rather than a call into
// ixformer: ``ds4_v41_compress_prepare_out`` gathers every group from the state
// cache by absolute position (``block_table[pos / capacity]``), which needs a
// state cache paged over the whole context. ``CircularBufferSpec`` reserves one
// block per request, so that gather runs off the end of the block-table row.
__global__ void v41_compress_pool_norm_kernel(
    const float* __restrict__ kv_score, int64_t raw_stride,
    const void* __restrict__ positions, bool positions_are_int64,
    float* __restrict__ state, int64_t page_stride, int64_t row_stride,
    int rows_per_page, int64_t num_pages,
    const void* __restrict__ slot_mapping, bool slots_are_int64,
    const void* __restrict__ query_start_loc, bool starts_are_int64,
    const void* __restrict__ token_to_req, bool reqs_are_int64,
    const __nv_bfloat16* __restrict__ weight, float eps,
    __nv_bfloat16* __restrict__ latent_out, int64_t out_stride,
    int64_t num_tokens, int64_t num_reqs) {
  __shared__ float smem[kMaxThreadsPerBlock / warpSize];
  const int d0 = threadIdx.x * kRopePacked;

  if (blockIdx.x < num_reqs) {
    const int64_t req = blockIdx.x;
    const int64_t start = load_index(query_start_loc, starts_are_int64, req);
    const int64_t end = load_index(query_start_loc, starts_are_int64, req + 1);
    if (start >= end) {
      return;
    }
    const int64_t first_slot =
        load_index(slot_mapping, slots_are_int64, start);
    if (first_slot < 0) {
      return;
    }
    const int64_t page = first_slot / rows_per_page;
    if (page >= num_pages) {
      return;
    }

    // Every test above is block-uniform, so the barrier and the reduction
    // below keep all their partners.
    const int64_t pos = load_index(positions, positions_are_int64, start);
    if ((pos + 1) % kPoolRatio == 0) {
      // A group of two closes on an odd position, so the predecessor row is
      // there and ``pos - 1`` never goes negative.
      const float* prev_row = state + page * page_stride +
                              ((pos - 1) % rows_per_page) * row_stride;
      float even, odd;
      pool_rows(prev_row, kv_score + start * raw_stride, d0, even, odd);
      // Every lane has read the ring before the tail store below writes it.
      __syncthreads();
      store_latent(even, odd, d0, weight, eps, latent_out, out_stride, start,
                   smem);
    }

    const int64_t span = end - start;
    const int64_t num_rows =
        span < static_cast<int64_t>(rows_per_page) ? span : rows_per_page;
    for (int64_t k = 0; k < num_rows; ++k) {
      const int64_t token = end - num_rows + k;
      const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
      if (slot < 0) {
        continue;
      }
      const int64_t tail_page = slot / rows_per_page;
      if (tail_page >= num_pages) {
        continue;
      }
      float* dst = state + tail_page * page_stride +
                   (slot % rows_per_page) * row_stride;
      const float* src = kv_score + token * raw_stride;
      *reinterpret_cast<float2*>(dst + d0) =
          *reinterpret_cast<const float2*>(src + d0);
      *reinterpret_cast<float2*>(dst + kLatentDim + d0) =
          *reinterpret_cast<const float2*>(src + kLatentDim + d0);
    }
    return;
  }

  int64_t t = (blockIdx.x - num_reqs) * kPoolRatio;
  if (t >= num_tokens) {
    return;
  }
  // Snap to the token that closes a group; its partner publishes nothing.
  t += (load_index(positions, positions_are_int64, t) + 1) % kPoolRatio;
  if (t >= num_tokens) {
    return;
  }
  if (load_index(slot_mapping, slots_are_int64, t) < 0) {
    return;
  }
  // A chunk's first token has no in-batch predecessor; its request program
  // pooled it against the ring.
  const int64_t req = load_index(token_to_req, reqs_are_int64, t);
  if (t == load_index(query_start_loc, starts_are_int64, req)) {
    return;
  }

  float even, odd;
  pool_rows(kv_score + (t - 1) * raw_stride, kv_score + t * raw_stride, d0,
            even, odd);
  store_latent(even, odd, d0, weight, eps, latent_out, out_stride, t, smem);
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

void check_slot_tensor(const torch::Tensor& t, const char* name) {
  check_index_tensor(t, name);
  TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64");
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
  check_slot_tensor(slot_mapping, "slot_mapping");

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

void v41_compress_pool_norm(torch::Tensor& kv_score, torch::Tensor& positions,
                            torch::Tensor& state_cache,
                            torch::Tensor& slot_mapping,
                            torch::Tensor& query_start_loc,
                            torch::Tensor& token_to_req_indices,
                            torch::Tensor& rms_norm_weight,
                            double rms_norm_eps, torch::Tensor& latent_out) {
  TORCH_CHECK(kv_score.is_cuda() && kv_score.dim() == 2,
              "kv_score must be a 2D CUDA tensor");
  TORCH_CHECK(kv_score.scalar_type() == at::kFloat, "kv_score must be float32");
  TORCH_CHECK(kv_score.size(1) == kPoolRatio * kLatentDim &&
                  kv_score.stride(1) == 1,
              "kv_score must be [T, ", kPoolRatio * kLatentDim,
              "] with a contiguous last dim");
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
  TORCH_CHECK(state_cache.is_cuda() && state_cache.dim() == 3 &&
                  state_cache.scalar_type() == at::kFloat,
              "state_cache must be a 3D fp32 CUDA tensor [pages, rows, width]");
  TORCH_CHECK(state_cache.size(2) == kPoolRatio * kLatentDim &&
                  state_cache.stride(2) == 1,
              "a state row is ", kPoolRatio * kLatentDim,
              " wide and contiguous");
  TORCH_CHECK(state_cache.stride(0) % 2 == 0 && state_cache.stride(1) % 2 == 0,
              "state_cache pages and rows must be 8-byte aligned");
  check_index_tensor(positions, "positions");
  check_slot_tensor(slot_mapping, "slot_mapping");
  check_index_tensor(query_start_loc, "query_start_loc");
  check_index_tensor(token_to_req_indices, "token_to_req_indices");

  const int64_t num_tokens = slot_mapping.numel();
  TORCH_CHECK(num_tokens <= kv_score.size(0) &&
                  num_tokens <= latent_out.size(0) &&
                  num_tokens <= positions.numel() &&
                  num_tokens <= token_to_req_indices.numel(),
              "slot_mapping is longer than the per-token tensors");
  TORCH_CHECK(query_start_loc.numel() >= 1,
              "query_start_loc must hold at least the leading zero");
  if (num_tokens == 0) {
    return;
  }
  const int64_t num_reqs = query_start_loc.numel() - 1;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_score));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // The reference grid: a program per request, then one per packed pair.
  const unsigned grid = static_cast<unsigned>(
      num_reqs + (num_tokens + kPoolRatio - 1) / kPoolRatio);
  v41_compress_pool_norm_kernel<<<grid, kLatentThreads, 0, stream>>>(
      kv_score.data_ptr<float>(), kv_score.stride(0), positions.data_ptr(),
      positions.scalar_type() == at::kLong, state_cache.data_ptr<float>(),
      state_cache.stride(0), state_cache.stride(1),
      static_cast<int>(state_cache.size(1)), state_cache.size(0),
      slot_mapping.data_ptr(), slot_mapping.scalar_type() == at::kLong,
      query_start_loc.data_ptr(),
      query_start_loc.scalar_type() == at::kLong,
      token_to_req_indices.data_ptr(),
      token_to_req_indices.scalar_type() == at::kLong,
      reinterpret_cast<const __nv_bfloat16*>(rms_norm_weight.data_ptr()),
      static_cast<float>(rms_norm_eps),
      reinterpret_cast<__nv_bfloat16*>(latent_out.data_ptr()),
      latent_out.stride(0), num_tokens, num_reqs);
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
  check_slot_tensor(slot_mapping, "slot_mapping");

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
