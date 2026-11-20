// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-to-host OpenMP scatter of dense FP8 wire rows into the sparse host pool.

#include <torch/extension.h>

#include <cstdint>
#include <cstring>

void h2h_scatter(torch::Tensor pool, torch::Tensor staging,
                 torch::Tensor block_ids, torch::Tensor dst_offsets,
                 torch::Tensor src_offsets, torch::Tensor lengths,
                 int64_t num_threads) {
  TORCH_CHECK(pool.scalar_type() == torch::kUInt8, "pool must be uint8");
  TORCH_CHECK(staging.scalar_type() == torch::kUInt8, "staging must be uint8");
  TORCH_CHECK(pool.dim() == 2 && staging.dim() == 2, "expected [rows, stride]");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt64, "block_ids must be int64");
  TORCH_CHECK(dst_offsets.numel() == src_offsets.numel() &&
                  dst_offsets.numel() == lengths.numel(),
              "span arrays must have equal length");

  const int64_t n_blocks = block_ids.numel();
  const int64_t n_spans = lengths.numel();
  if (n_blocks == 0 || n_spans == 0) {
    return;
  }

  auto* pool_base = pool.data_ptr<uint8_t>();
  const auto* staging_base = staging.data_ptr<uint8_t>();
  const int64_t pool_stride = pool.stride(0);
  const int64_t staging_stride = staging.stride(0);
  const int64_t pool_rows = pool.size(0);
  const int64_t staging_rows = staging.size(0);
  TORCH_CHECK(n_blocks <= staging_rows, "more block ids than staging rows");

  const auto* ids = block_ids.data_ptr<int64_t>();
  const auto* dst_off = dst_offsets.data_ptr<int64_t>();
  const auto* src_off = src_offsets.data_ptr<int64_t>();
  const auto* lens = lengths.data_ptr<int64_t>();

  for (int64_t i = 0; i < n_blocks; ++i) {
    TORCH_CHECK(ids[i] >= 0 && ids[i] < pool_rows, "block id out of range");
  }
  for (int64_t k = 0; k < n_spans; ++k) {
    TORCH_CHECK(dst_off[k] >= 0 && dst_off[k] + lens[k] <= pool_stride,
                "destination span outside the pool block");
    TORCH_CHECK(src_off[k] >= 0 && src_off[k] + lens[k] <= staging_stride,
                "source span outside the staging row");
  }

#pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t k = 0; k < n_spans; ++k) {
      std::memcpy(pool_base + ids[i] * pool_stride + dst_off[k],
                  staging_base + i * staging_stride + src_off[k], lens[k]);
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2h_scatter", &h2h_scatter, "Scatter dense FP8 wire rows into the host pool");
}
