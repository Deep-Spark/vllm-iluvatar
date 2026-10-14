// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 two-level indexer, block half:
//   v41_candidate_block_scores -- per-candidate-block max of the index logits
//   v41_apply_candidate_mask   -- -inf everything outside the candidate blocks
//
// V4.1 scores its index logits twice. The layer at ``candidate_source_layer_id``
// publishes a coarse shortlist -- the best ``candidate_topk_blocks`` blocks of
// ``candidate_block_size`` columns each -- and the index sources above it keep
// only those blocks before taking their own top-k. Official spells both halves
// in Triton (``vllm/model_executor/kernels/attention/dsa/candidate_blocks.py``),
// which does not run on corex, so they are reproduced here; the arithmetic
// below is a transcription of those kernels, down to the two special cases:
//
//   * the block holding a row's newest column always scores ``+inf``, so the
//     shortlist cannot drop the most recent context;
//   * a candidate whose first column lands past the logits width is folded into
//     a single "edge" flag that keeps the last column instead.
//
// Columns are *packed*: a row's context starts at ``row_ks`` and ends at
// ``row_ke``, and block ``b`` covers ``[start + b*block_size, ...)``. Rows share
// bounds in groups of ``row_repeat`` (decode batches that flatten several query
// rows per request); V4.1 passes per-token bounds, so ``row_repeat`` is 1.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "ops.h"
#include "utils.cuh"

namespace vllm_iluvatar {
namespace {

constexpr int kThreads = 256;

inline int64_t cdiv_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The index logits come out of whichever ixformer scoring kernel ran: bf16 on
// this platform, fp32 from the reference path. Both have to be maskable in
// place, so the two kernels below are generic over the element type and do
// their arithmetic in fp32.
template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v);
template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half_rn(v);
}

#define VLLM_ILU_DISPATCH_LOGITS(TYPE, ...)                            \
  switch (TYPE) {                                                      \
    case at::kFloat: {                                                 \
      using scalar_t = float;                                          \
      __VA_ARGS__;                                                     \
      break;                                                           \
    }                                                                  \
    case at::kBFloat16: {                                              \
      using scalar_t = __nv_bfloat16;                                  \
      __VA_ARGS__;                                                     \
      break;                                                           \
    }                                                                  \
    case at::kHalf: {                                                  \
      using scalar_t = __half;                                         \
      __VA_ARGS__;                                                     \
      break;                                                           \
    }                                                                  \
    default:                                                           \
      TORCH_CHECK(false, "logits must be fp32, bf16 or fp16, got ",    \
                  TYPE);                                               \
  }

// Triton's ``tl.maximum(..., propagate_nan=ALL)``. A NaN anywhere in a block
// has to survive the reduction, or a warmup row of NaNs would score as a
// perfectly ordinary block and be shortlisted.
__device__ __forceinline__ float nan_max(float a, float b) {
  if (isnan(a) || isnan(b)) {
    return CUDART_NAN_F;
  }
  return fmaxf(a, b);
}

__device__ __forceinline__ int load_bound(const int32_t* p, int64_t stride,
                                          int64_t row, int row_repeat) {
  return static_cast<int>(p[(row / row_repeat) * stride]);
}

template <typename scalar_t>
__global__ void v41_candidate_block_scores_kernel(
    const scalar_t* __restrict__ logits, int64_t stride_row,
    int64_t stride_col, const int32_t* __restrict__ starts,
    int64_t start_stride, const int32_t* __restrict__ ends, int64_t end_stride,
    scalar_t* __restrict__ scores, int width, int nblocks, int block_size,
    int row_repeat) {
  const int64_t row = blockIdx.x;
  const int block = blockIdx.y * blockDim.x + threadIdx.x;
  if (block >= nblocks) {
    return;
  }

  const int start =
      starts == nullptr ? 0 : load_bound(starts, start_stride, row, row_repeat);
  const int end = load_bound(ends, end_stride, row, row_repeat);

  float acc = -INFINITY;
  const int64_t base = static_cast<int64_t>(start) +
                       static_cast<int64_t>(block) * block_size;
  for (int j = 0; j < block_size; ++j) {
    const int64_t col = base + j;
    if (col < end && col < width) {
      acc = nan_max(acc,
                    to_float(logits[row * stride_row + col * stride_col]));
    }
  }

  // Pin the row's newest block: whatever it scored, it is always shortlisted.
  if (end > start && block == (end - start - 1) / block_size) {
    acc = INFINITY;
  }
  scores[row * static_cast<int64_t>(nblocks) + block] = from_float<scalar_t>(acc);
}

// One block per row. Zeroes the row's flags, then marks the shortlisted
// blocks; slot ``nblocks`` is the edge flag described in the header.
__global__ void v41_candidate_flags_kernel(
    const int32_t* __restrict__ candidates, int64_t cand_stride_row,
    int64_t cand_stride_col, const int32_t* __restrict__ starts,
    int64_t start_stride, uint8_t* __restrict__ flags, int width, int nblocks,
    int block_size, int num_candidates, int row_repeat) {
  const int64_t row = blockIdx.x;
  uint8_t* const row_flags = flags + row * static_cast<int64_t>(nblocks + 1);
  for (int i = threadIdx.x; i <= nblocks; i += blockDim.x) {
    row_flags[i] = 0;
  }
  __syncthreads();

  const int start =
      starts == nullptr ? 0 : load_bound(starts, start_stride, row, row_repeat);
  for (int c = threadIdx.x; c < num_candidates; c += blockDim.x) {
    int64_t block = candidates[row * cand_stride_row + c * cand_stride_col];
    if (start + block * block_size >= width) {
      block = nblocks;
    }
    if (block >= 0) {
      row_flags[block] = 1;
    }
  }
}

template <typename scalar_t>
__global__ void v41_mask_candidates_kernel(
    scalar_t* __restrict__ logits, int64_t stride_row, int64_t stride_col,
    const int32_t* __restrict__ starts, int64_t start_stride,
    const int32_t* __restrict__ ends, int64_t end_stride,
    const uint8_t* __restrict__ flags, int width, int nblocks, int block_size,
    int row_repeat) {
  const int64_t row = blockIdx.x;
  const int64_t col =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (col >= width) {
    return;
  }

  const int start =
      starts == nullptr ? 0 : load_bound(starts, start_stride, row, row_repeat);
  const int end = load_bound(ends, end_stride, row, row_repeat);
  const uint8_t* const row_flags =
      flags + row * static_cast<int64_t>(nblocks + 1);

  const bool valid = col >= start && col < end;
  bool keep = false;
  if (valid) {
    keep = row_flags[(col - start) / block_size] != 0;
  }
  keep = keep || (col == width - 1 && row_flags[nblocks] != 0);

  if (!(valid && keep)) {
    logits[row * stride_row + col * stride_col] =
        from_float<scalar_t>(-INFINITY);
  }
}

// The INT8 decode scorer already initializes the complete exterior to -inf.
// Keep the private flag and mask work bounded by the current GPU row length.
__global__ void v41_initialized_decode_flags_kernel(
    const int32_t* candidates, const int32_t* ends, uint8_t* flags,
    int width, int num_candidates, int64_t end_stride,
    int64_t candidate_row_stride, int64_t candidate_col_stride) {
  const int row = blockIdx.x;
  const int nblocks = (width + 7) / 8;
  const int length = max(0, min(ends[row * end_stride], width));
  const int live_blocks = (length + 7) / 8;
  uint8_t* row_flags = flags + static_cast<int64_t>(row) * (nblocks + 1);
  for (int i = threadIdx.x; i < live_blocks; i += blockDim.x) row_flags[i] = 0;
  if (threadIdx.x == 0) row_flags[nblocks] = 0;
  __syncthreads();
  for (int c = threadIdx.x; c < num_candidates; c += blockDim.x) {
    const int64_t block = candidates[static_cast<int64_t>(row) *
        candidate_row_stride + static_cast<int64_t>(c) * candidate_col_stride];
    if (block < 0) continue;
    if (block * 8 >= width) row_flags[nblocks] = 1;
    else if (block < live_blocks) row_flags[block] = 1;
  }
}

__global__ void v41_initialized_decode_mask_kernel(
    int16_t* logits, const int32_t* ends, const uint8_t* flags,
    int width, int64_t logits_row_stride, int64_t end_stride) {
  const int row = blockIdx.x;
  const int length = max(0, min(ends[row * end_stride], width));
  const int nblocks = (width + 7) / 8;
  const uint8_t* row_flags = flags + static_cast<int64_t>(row) * (nblocks + 1);
  int16_t* row_logits = logits + static_cast<int64_t>(row) * logits_row_stride;
#if __IVCORE_ARCH__
  const auto desc = __build_desc(row_logits);
#endif
  for (int col = blockIdx.y * blockDim.x + threadIdx.x;
       col < length; col += gridDim.y * blockDim.x) {
    const bool keep = row_flags[col / 8] != 0 ||
        (col == width - 1 && row_flags[nblocks] != 0);
    if (!keep) {
#if __IVCORE_ARCH__
      // Preserve coherence through the native score/selection pipeline.
      __ivcorex_ml_mem_store_i16(static_cast<int16_t>(0xff80), desc, col * 2, 0, 0);
#else
      row_logits[col] = static_cast<int16_t>(0xff80);
#endif
    }
  }
}

// Fold ``torch.topk``'s two outputs into the shortlist row ``out`` expects.
// ``out`` is wider than ``k`` (it holds ``topk_blocks`` columns); the first
// ``k`` take an index, unless that slot scored ``-inf`` -- a block with no
// live column, which is padding, not a candidate -- and the tail is ``-1``.
//
// A NaN is left as a kept index: ``values != -inf`` is true for it, which is
// the tie behaviour the reference keeps. The whole point of the kernel is that
// the select, the int32 cast, the row copy and the tail fill the python used
// to spell as four torch ops over the slice become one launch that writes
// ``out`` once.
template <typename scalar_t>
__global__ void v41_write_candidate_topk_kernel(
    const scalar_t* __restrict__ values, int64_t val_stride_row,
    int64_t val_stride_col, const int64_t* __restrict__ indices,
    int64_t idx_stride_row, int64_t idx_stride_col, int32_t* __restrict__ out,
    int64_t out_stride_row, int64_t out_stride_col, int k, int out_width) {
  const int64_t row = blockIdx.x;
  for (int col = blockIdx.y * blockDim.x + threadIdx.x; col < out_width;
       col += static_cast<int>(gridDim.y) * blockDim.x) {
    int32_t v = -1;
    if (col < k) {
      const float score =
          to_float(values[row * val_stride_row + col * val_stride_col]);
      if (score != -INFINITY) {
        v = static_cast<int32_t>(
            indices[row * idx_stride_row + col * idx_stride_col]);
      }
    }
    out[row * out_stride_row + col * out_stride_col] = v;
  }
}

void check_logits(const torch::Tensor& logits) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2,
              "logits must be a 2D CUDA tensor [rows, width]");
}

void check_bound(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.scalar_type() == at::kInt, name,
              " must be a 1D int32 CUDA tensor");
}

void check_row_groups(int64_t rows, int64_t bound_rows, int64_t row_repeat,
                      const char* name) {
  TORCH_CHECK(row_repeat >= 1, "row_repeat must be positive, got ", row_repeat);
  TORCH_CHECK((rows + row_repeat - 1) / row_repeat <= bound_rows, name,
              " holds ", bound_rows, " bounds but ", rows,
              " rows in groups of ", row_repeat, " need ",
              (rows + row_repeat - 1) / row_repeat);
}

}  // namespace

torch::Tensor v41_candidate_block_scores(
    torch::Tensor& logits, const std::optional<torch::Tensor>& row_ks,
    torch::Tensor& row_ke, int64_t block_size, int64_t row_repeat) {
  check_logits(logits);
  check_bound(row_ke, "row_ke");
  TORCH_CHECK(block_size > 0, "block_size must be positive, got ", block_size);

  const int64_t rows = logits.size(0);
  const int64_t width = logits.size(1);
  const int64_t nblocks = cdiv_i64(width, block_size);
  auto scores = logits.new_empty({rows, nblocks});
  if (rows == 0 || width == 0) {
    return scores;
  }
  check_row_groups(rows, row_ke.numel(), row_repeat, "row_ke");

  const int32_t* starts_ptr = nullptr;
  int64_t start_stride = 0;
  if (row_ks.has_value()) {
    const torch::Tensor& ks = *row_ks;
    check_bound(ks, "row_ks");
    check_row_groups(rows, ks.numel(), row_repeat, "row_ks");
    starts_ptr = ks.data_ptr<int32_t>();
    start_stride = ks.stride(0);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(cdiv_i64(nblocks, kThreads)));
  VLLM_ILU_DISPATCH_LOGITS(
      logits.scalar_type(),
      v41_candidate_block_scores_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(logits.data_ptr()),
          logits.stride(0), logits.stride(1), starts_ptr, start_stride,
          row_ke.data_ptr<int32_t>(), row_ke.stride(0),
          reinterpret_cast<scalar_t*>(scores.data_ptr()),
          static_cast<int>(width), static_cast<int>(nblocks),
          static_cast<int>(block_size), static_cast<int>(row_repeat)));
  return scores;
}

void v41_write_candidate_topk(torch::Tensor& out, torch::Tensor& values,
                              torch::Tensor& indices, int64_t k) {
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 &&
                  out.scalar_type() == at::kInt,
              "out must be a 2D int32 CUDA tensor [rows, topk_blocks]");
  TORCH_CHECK(values.is_cuda() && values.dim() == 2,
              "values must be a 2D CUDA tensor [rows, k]");
  TORCH_CHECK(indices.is_cuda() && indices.dim() == 2 &&
                  indices.scalar_type() == at::kLong,
              "indices must be a 2D int64 CUDA tensor [rows, k]");

  const int64_t rows = out.size(0);
  const int64_t out_width = out.size(1);
  TORCH_CHECK(k >= 0 && k <= out_width, "k must be in [0, out_width], got ", k,
              " with out_width ", out_width);
  TORCH_CHECK(values.size(0) >= rows && indices.size(0) >= rows,
              "values / indices have fewer rows than out");
  TORCH_CHECK(values.size(1) >= k && indices.size(1) >= k,
              "values / indices are narrower than k");
  if (rows == 0 || out_width == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(cdiv_i64(out_width, kThreads)));
  VLLM_ILU_DISPATCH_LOGITS(
      values.scalar_type(),
      v41_write_candidate_topk_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(values.data_ptr()),
          values.stride(0), values.stride(1), indices.data_ptr<int64_t>(),
          indices.stride(0), indices.stride(1), out.data_ptr<int32_t>(),
          out.stride(0), out.stride(1), static_cast<int>(k),
          static_cast<int>(out_width)));
}

void v41_apply_candidate_mask(torch::Tensor& logits,
                              const std::optional<torch::Tensor>& row_ks,
                              torch::Tensor& row_ke,
                              torch::Tensor& candidate_blocks,
                              int64_t block_size, int64_t row_repeat) {
  check_logits(logits);
  check_bound(row_ke, "row_ke");
  TORCH_CHECK(block_size > 0, "block_size must be positive, got ", block_size);
  TORCH_CHECK(candidate_blocks.is_cuda() && candidate_blocks.dim() == 2 &&
                  candidate_blocks.scalar_type() == at::kInt,
              "candidate_blocks must be a 2D int32 CUDA tensor [rows, k]");

  const int64_t rows = logits.size(0);
  const int64_t width = logits.size(1);
  if (rows == 0 || width == 0) {
    return;
  }
  TORCH_CHECK(candidate_blocks.size(0) >= rows, "candidate_blocks holds ",
              candidate_blocks.size(0), " rows but the logits have ", rows);
  check_row_groups(rows, row_ke.numel(), row_repeat, "row_ke");

  const int32_t* starts_ptr = nullptr;
  int64_t start_stride = 0;
  if (row_ks.has_value()) {
    const torch::Tensor& ks = *row_ks;
    check_bound(ks, "row_ks");
    check_row_groups(rows, ks.numel(), row_repeat, "row_ks");
    starts_ptr = ks.data_ptr<int32_t>();
    start_stride = ks.stride(0);
  }

  const int64_t nblocks = cdiv_i64(width, block_size);
  auto flags = at::empty({rows, nblocks + 1},
                         logits.options().dtype(at::kByte));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  v41_candidate_flags_kernel<<<static_cast<unsigned>(rows), kThreads, 0,
                               stream>>>(
      candidate_blocks.data_ptr<int32_t>(), candidate_blocks.stride(0),
      candidate_blocks.stride(1), starts_ptr, start_stride,
      flags.data_ptr<uint8_t>(), static_cast<int>(width),
      static_cast<int>(nblocks), static_cast<int>(block_size),
      static_cast<int>(candidate_blocks.size(1)),
      static_cast<int>(row_repeat));

  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(cdiv_i64(width, kThreads)));
  VLLM_ILU_DISPATCH_LOGITS(
      logits.scalar_type(),
      v41_mask_candidates_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<scalar_t*>(logits.data_ptr()), logits.stride(0),
          logits.stride(1), starts_ptr, start_stride,
          row_ke.data_ptr<int32_t>(), row_ke.stride(0),
          flags.data_ptr<uint8_t>(), static_cast<int>(width),
          static_cast<int>(nblocks), static_cast<int>(block_size),
          static_cast<int>(row_repeat)));
}

void v41_apply_initialized_decode_mask(torch::Tensor& logits,
                                       torch::Tensor& row_ke,
                                       torch::Tensor& candidate_blocks) {
  check_logits(logits);
  check_bound(row_ke, "row_ke");
  const int64_t rows = logits.size(0);
  const int64_t width = logits.size(1);
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.stride(1) == 1,
              "initialized decode logits must be bf16 with contiguous columns");
  TORCH_CHECK(rows == 0 || rows == 6 || rows == 12 || rows == 18 || rows == 24,
              "initialized decode mask requires 6, 12, 18 or 24 query rows");
  TORCH_CHECK(candidate_blocks.is_cuda() && candidate_blocks.dim() == 2 &&
                  candidate_blocks.scalar_type() == at::kInt &&
                  candidate_blocks.size(0) >= rows,
              "candidate_blocks must hold all query rows as CUDA int32");
  TORCH_CHECK(row_ke.device() == logits.device() &&
                  candidate_blocks.device() == logits.device(),
              "initialized decode mask tensors must be on the same device");
  check_row_groups(rows, row_ke.numel(), 1, "row_ke");
  if (rows == 0 || width == 0) return;
  const int64_t nblocks = cdiv_i64(width, 8);
  auto flags = at::empty({rows, nblocks + 1}, logits.options().dtype(at::kByte));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  v41_initialized_decode_flags_kernel<<<static_cast<unsigned>(rows), kThreads,
                                        0, stream>>>(
      candidate_blocks.data_ptr<int32_t>(), row_ke.data_ptr<int32_t>(),
      flags.data_ptr<uint8_t>(), static_cast<int>(width),
      static_cast<int>(candidate_blocks.size(1)), row_ke.stride(0),
      candidate_blocks.stride(0), candidate_blocks.stride(1));
  v41_initialized_decode_mask_kernel<<<dim3(static_cast<unsigned>(rows), 32),
                                       kThreads, 0, stream>>>(
      reinterpret_cast<int16_t*>(logits.data_ptr()), row_ke.data_ptr<int32_t>(),
      flags.data_ptr<uint8_t>(), static_cast<int>(width), logits.stride(0),
      row_ke.stride(0));
}

}  // namespace vllm_iluvatar
