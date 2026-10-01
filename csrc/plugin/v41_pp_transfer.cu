// SPDX-License-Identifier: Apache-2.0
//
// v41_move_rows_to_paged_cache -- the receiving half of a DeepSeek-V4.1
// pipeline KV transfer.
//
// A KV or index-key source on an earlier pipeline rank wrote its records twice:
// once into its own paged cache and once into a dense ``[num_tokens, width]``
// buffer that rode the pipeline here. This places that buffer into the local
// paged cache using *this* rank's slot mapping, so the forwarder accumulates a
// full replica of the source's cache step by step.
//
// Nothing is recomputed and nothing is requantized. The bytes that land here
// are the bytes the source wrote, which is what makes the replica exact rather
// than merely close.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "ops.h"
#include "v41_common.cuh"

namespace vllm_iluvatar {
namespace {

using namespace v41;

constexpr int kMoveThreads = 128;

// Packed records keep two regions per page: every row's values, then every
// row's scales. A dense row is the two laid end to end, so the copy splits.
// Both region widths are multiples of four, and so is every offset into them,
// which is what lets this move 32 bits at a time.
__global__ void v41_move_packed_rows_kernel(
    const uint8_t* __restrict__ rows, int row_bytes,
    uint8_t* __restrict__ cache, int64_t page_stride, int rows_per_page,
    int64_t num_pages, const void* __restrict__ slot_mapping,
    bool slots_are_int64, const void* __restrict__ positions,
    bool positions_are_int64, int compress_ratio, int value_bytes,
    int scale_bytes) {
  const int64_t token = blockIdx.x;
  const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
  const int64_t pos = load_index(positions, positions_are_int64, token);
  // The sender applied the same two tests, so a row it skipped is stale here
  // and these guards are what keep it from being read.
  if (slot < 0 || (pos + 1) % compress_ratio != 0) {
    return;
  }
  const int64_t page = slot / rows_per_page;
  if (page >= num_pages) {
    return;
  }
  const int64_t row = slot % rows_per_page;

  const uint32_t* src =
      reinterpret_cast<const uint32_t*>(rows + token * row_bytes);
  uint8_t* const page_base = cache + page * page_stride;

  uint32_t* dst_values =
      reinterpret_cast<uint32_t*>(page_base + row * value_bytes);
  const int value_words = value_bytes / 4;
  for (int i = threadIdx.x; i < value_words; i += kMoveThreads) {
    dst_values[i] = src[i];
  }

  uint32_t* dst_scales = reinterpret_cast<uint32_t*>(
      page_base + static_cast<int64_t>(rows_per_page) * value_bytes +
      row * scale_bytes);
  const int scale_words = scale_bytes / 4;
  for (int i = threadIdx.x; i < scale_words; i += kMoveThreads) {
    dst_scales[i] = src[value_words + i];
  }
}

// Plain records are one contiguous row of the cache dtype, so the page's row
// stride is all the addressing needed.
__global__ void v41_move_plain_rows_kernel(
    const __nv_bfloat16* __restrict__ rows, int width,
    __nv_bfloat16* __restrict__ cache, int64_t page_stride, int64_t row_stride,
    int rows_per_page, int64_t num_pages,
    const void* __restrict__ slot_mapping, bool slots_are_int64,
    const void* __restrict__ positions, bool positions_are_int64,
    int compress_ratio) {
  const int64_t token = blockIdx.x;
  const int64_t slot = load_index(slot_mapping, slots_are_int64, token);
  const int64_t pos = load_index(positions, positions_are_int64, token);
  if (slot < 0 || (pos + 1) % compress_ratio != 0) {
    return;
  }
  const int64_t page = slot / rows_per_page;
  if (page >= num_pages) {
    return;
  }
  const int64_t row = slot % rows_per_page;

  const __nv_bfloat162* src =
      reinterpret_cast<const __nv_bfloat162*>(rows + token * width);
  __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
      cache + page * page_stride + row * row_stride);
  const int pairs = width / 2;
  for (int i = threadIdx.x; i < pairs; i += kMoveThreads) {
    dst[i] = src[i];
  }
}

// Value / scale split of a packed record, keyed by its width. These are the
// two the plugin writes; the wider MXFP8 and NVFP4 records have no reader here.
bool packed_regions(int64_t row_bytes, int& value_bytes, int& scale_bytes) {
  if (row_bytes == kFp8RecordBytes) {  // 584 = 576 + 8, compressed KV
    value_bytes = kFp8TokenDataBytes;
    scale_bytes = kFp8ScaleBytes;
    return true;
  }
  if (row_bytes == 132) {  // 128 e4m3 + one fp32 scale, indexer key
    value_bytes = 128;
    scale_bytes = 4;
    return true;
  }
  return false;
}

}  // namespace

void v41_move_rows_to_paged_cache(torch::Tensor& rows,
                                  torch::Tensor& kv_cache,
                                  torch::Tensor& slot_mapping,
                                  torch::Tensor& positions,
                                  int64_t compress_ratio) {
  TORCH_CHECK(compress_ratio == 1 || compress_ratio == 2,
              "compress_ratio must be 1 or 2, got ", compress_ratio);
  TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous(),
              "rows must be a 2D contiguous CUDA tensor");
  TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dim() == 3,
              "kv_cache must be a 3D CUDA tensor [pages, rows, width]");
  TORCH_CHECK(rows.scalar_type() == kv_cache.scalar_type(),
              "rows must share the cache dtype");
  TORCH_CHECK(rows.size(1) == kv_cache.size(2),
              "rows are ", rows.size(1), " wide but the cache record is ",
              kv_cache.size(2));
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.is_contiguous(),
              "slot_mapping must be a contiguous CUDA tensor");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong,
              "slot_mapping must be int64");
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous(),
              "positions must be a contiguous CUDA tensor");
  TORCH_CHECK(
      positions.scalar_type() == at::kLong ||
          positions.scalar_type() == at::kInt,
      "positions must be int32 or int64");

  const int64_t num_tokens = slot_mapping.numel();
  TORCH_CHECK(num_tokens <= rows.size(0) && num_tokens <= positions.numel(),
              "slot_mapping is longer than rows / positions");
  if (num_tokens == 0) {
    return;
  }

  const int rows_per_page = static_cast<int>(kv_cache.size(1));
  const int64_t num_pages = kv_cache.size(0);
  const bool slots_are_int64 = slot_mapping.scalar_type() == at::kLong;
  const bool positions_are_int64 = positions.scalar_type() == at::kLong;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned grid = static_cast<unsigned>(num_tokens);

  if (kv_cache.scalar_type() == at::kByte) {
    int value_bytes = 0;
    int scale_bytes = 0;
    TORCH_CHECK(packed_regions(kv_cache.size(2), value_bytes, scale_bytes),
                "unsupported packed record width ", kv_cache.size(2),
                "; expected 584 (compressed KV) or 132 (indexer key)");
    TORCH_CHECK(kv_cache.stride(2) == 1 &&
                    kv_cache.stride(1) == value_bytes + scale_bytes,
                "the packed page must be internally contiguous");
    v41_move_packed_rows_kernel<<<grid, kMoveThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(rows.data_ptr()),
        static_cast<int>(rows.size(1)),
        reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache.stride(0),
        rows_per_page, num_pages, slot_mapping.data_ptr(), slots_are_int64,
        positions.data_ptr(), positions_are_int64,
        static_cast<int>(compress_ratio), value_bytes, scale_bytes);
    return;
  }

  TORCH_CHECK(kv_cache.scalar_type() == at::kBFloat16,
              "a plain cache row is bfloat16; got ", kv_cache.scalar_type());
  TORCH_CHECK(kv_cache.stride(2) == 1 && kv_cache.size(2) % 2 == 0,
              "a plain cache row must be contiguous and of even width");
  v41_move_plain_rows_kernel<<<grid, kMoveThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(rows.data_ptr()),
      static_cast<int>(rows.size(1)),
      reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr()),
      kv_cache.stride(0), kv_cache.stride(1), rows_per_page, num_pages,
      slot_mapping.data_ptr(), slots_are_int64, positions.data_ptr(),
      positions_are_int64, static_cast<int>(compress_ratio));
}

}  // namespace vllm_iluvatar
