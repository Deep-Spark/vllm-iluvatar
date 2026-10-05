// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 Engram kernels.
//
// engram_write_hash_cache  -- compressed ids into the slot cache
// engram_ngram_hash        -- paged lookback + n-gram hash (one launch)
// engram_lookup_fp8        -- e4m3 * ue8m0 gather into bf16, UVA-safe
// engram_register_host     -- cudaHostRegister a CPU tensor, no copy
//
// The hash matches EngramTokenHistory.lookback_paged + IluEngram.hash_ids:
// rolling starts at 0, each shift xors (token * multiplier), and shifts > 0
// emit rolling % prime + offset. A dead source (slot miss, block id <= 0,
// or a position before 0) sticks, and later shifts hash the pad id.
// Modulo is floor-mod so it matches torch's int64 `%` when the xor sets the
// sign bit. This file does not use ixformer.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <unistd.h>

#include "ops.h"

namespace vllm_iluvatar {
namespace {

constexpr int64_t kDeadId = -1;
constexpr int kThreads = 128;

__device__ __forceinline__ int64_t floor_mod(int64_t value, int64_t prime) {
  int64_t mod = value % prime;
  if (mod < 0) {
    mod += prime;
  }
  return mod;
}

// NVIDIA fp8 e4m3fn: bias 7, no infinity, 0x7F/0xFF are NaN, 0x7E is 448.
__device__ __forceinline__ float e4m3fn_to_fp32(uint8_t bits) {
  const int sign = bits >> 7;
  const int exp = (bits >> 3) & 0xF;
  const int mant = bits & 0x7;
  float value;
  if (exp == 0) {
    value = ldexpf(static_cast<float>(mant), -9);
  } else if (exp == 0xF && mant == 0x7) {
    value = __uint_as_float(0x7fc00000u);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  }
  return sign ? -value : value;
}

// ue8m0 byte is the fp32 exponent field. byte 0 is 2^-127, not 0.
__device__ __forceinline__ float ue8m0_to_fp32(uint8_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

__global__ void engram_write_hash_cache_kernel(
    const int64_t* __restrict__ input_ids, const int64_t* __restrict__ token_map,
    const int64_t* __restrict__ slot_mapping, int64_t* __restrict__ cache,
    int64_t ids_stride, int64_t slot_stride, int64_t nslots, int64_t vocab,
    int64_t num_tokens) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int64_t raw = input_ids[token * ids_stride];
  int64_t id = raw;
  if (id < 0) {
    id = 0;
  } else if (id >= vocab) {
    id = vocab - 1;
  }
  const int64_t value = token_map[id];
  const int64_t slot = slot_mapping[token * slot_stride];
  const int64_t dst = (slot >= 0 && slot < nslots) ? slot : nslots;
  cache[dst] = value;
}

__global__ void engram_ngram_hash_kernel(
    const int64_t* __restrict__ input_ids, const int64_t* __restrict__ positions,
    const int64_t* __restrict__ token_map, const int64_t* __restrict__ cache,
    const int32_t* __restrict__ block_table, const int64_t* __restrict__ token_to_req,
    const int32_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ multipliers, const int64_t* __restrict__ primes,
    const int64_t* __restrict__ offsets, int64_t* __restrict__ output,
    int64_t ids_stride, int64_t pos_stride, int64_t req_stride, int64_t bt_stride0,
    int64_t bt_stride1, int64_t qsl_stride, int max_ngram, int n_heads,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    int64_t vocab, int64_t num_tokens, int64_t bt_rows, int64_t bt_cols,
    int64_t qsl_n, int64_t cache_n, bool paged,
    const int32_t* __restrict__ lookback, int64_t lookback_stride,
    int64_t lookback_rows, int64_t lookback_depth,
    bool lookback_is_prompt_only) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int64_t position = positions[token * pos_stride];
  const int n_cols = (max_ngram - 1) * n_heads;

  int64_t chunk_start = 0;
  int64_t req = 0;
  if (paged) {
    const int64_t req_cap = num_reqs < qsl_n ? num_reqs : qsl_n;
    const int64_t req_lim = req_cap - 1;
    req = token_to_req[token * req_stride];
    if (req < 0) {
      req = 0;
    } else if (req > req_lim) {
      req = req_lim;
    }
    int64_t t0 = query_start_loc[req * qsl_stride];
    if (t0 < 0) {
      t0 = 0;
    } else if (t0 > num_tokens - 1) {
      t0 = num_tokens - 1;
    }
    chunk_start = positions[t0 * pos_stride];
  }

  int64_t rolling = 0;
  bool blocked = false;
  for (int shift = 0; shift < max_ngram; ++shift) {
    const int64_t look = position - static_cast<int64_t>(shift);
    int64_t source = kDeadId;
    if (paged) {
      const bool in_batch = look >= chunk_start;
      if (in_batch) {
        int64_t batch_idx = token - static_cast<int64_t>(shift);
        if (batch_idx < 0) {
          batch_idx = 0;
        }
        int64_t id = input_ids[batch_idx * ids_stride];
        if (id < 0) {
          id = 0;
        } else if (id >= vocab) {
          id = vocab - 1;
        }
        source = token_map[id];
      } else if (look >= 0) {
        bool read_cache = lookback == nullptr;
        if (lookback != nullptr) {
          const int64_t offset = chunk_start - look - 1;
          if (req < lookback_rows && offset < lookback_depth) {
            const int32_t id = lookback[req * lookback_stride + offset];
            if (id >= 0 && id < vocab) {
              source = token_map[id];
            }
            // V1 only supplies prompt history: -1 for generated positions
            // means consult the paged token cache. In V2 the window is
            // complete, so -1 remains missing history even if a page exists.
            read_cache = lookback_is_prompt_only && id == kDeadId;
          }
        }
        if (read_cache && bt_rows > 0 && bt_cols > 0 && block_size > 0) {
          const int64_t blk = look / block_size;
          const bool in_range = blk < bt_cols;
          int64_t row = req;
          if (row >= bt_rows) {
            row = bt_rows - 1;
          }
          int64_t col = blk;
          if (col < 0) {
            col = 0;
          } else if (col >= bt_cols) {
            col = bt_cols - 1;
          }
          const int32_t block =
              block_table[row * bt_stride0 + col * bt_stride1];
          const int64_t slot =
              static_cast<int64_t>(block) * block_size + (look % block_size);
          if (in_range && block > 0 && slot >= 0 && slot < nslots) {
            source = cache[slot];
          }
        }
      }
    } else if (look >= 0 && look < cache_n) {
      source = cache[look];
    }

    blocked = blocked || (position < static_cast<int64_t>(shift)) ||
              (source == kDeadId);
    const int64_t value = blocked ? pad_id : source;
    rolling ^= value * multipliers[shift];
    if (shift == 0) {
      continue;
    }
    const int group = shift - 1;
    for (int head = 0; head < n_heads; ++head) {
      const int col = group * n_heads + head;
      const int64_t prime = primes[col];
      const int64_t hashed = floor_mod(rolling, prime) + offsets[col];
      output[token * n_cols + col] = hashed;
    }
  }
}

__global__ void engram_lookup_fp8_kernel(
    const int64_t* __restrict__ indices, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale, __nv_bfloat16* __restrict__ out,
    int64_t idx_stride0, int64_t idx_stride1, int64_t vocab_start,
    int64_t vocab_end, int dim, int n_cols, int64_t num_tokens) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t num_rows = num_tokens * static_cast<int64_t>(n_cols);
  if (row >= num_rows) {
    return;
  }
  const int64_t token = row / n_cols;
  const int col = static_cast<int>(row - token * n_cols);
  const int64_t index = indices[token * idx_stride0 + col * idx_stride1];
  const bool owned = index >= vocab_start && index < vocab_end;
  const int64_t local = owned ? (index - vocab_start) : 0;
  const int scale_stride = dim >> 5;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += kThreads) {
    float value = 0.f;
    if (owned) {
      const uint8_t w = weight[local * dim + d];
      const uint8_t s = scale[local * scale_stride + (d >> 5)];
      value = e4m3fn_to_fp32(w) * ue8m0_to_fp32(s);
    }
    out[row * dim + d] = __float2bfloat16(value);
  }
}

torch::Tensor require_cuda_long(torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  if (tensor.scalar_type() == torch::kLong && tensor.is_contiguous()) {
    return tensor;
  }
  return tensor.to(torch::kLong).contiguous();
}

}  // namespace

void engram_write_hash_cache(torch::Tensor& input_ids, torch::Tensor& token_map,
                             torch::Tensor& slot_mapping, torch::Tensor& cache,
                             int64_t nslots) {
  TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
  TORCH_CHECK(token_map.is_cuda() && token_map.is_contiguous(),
              "token_map must be a contiguous CUDA tensor");
  TORCH_CHECK(token_map.scalar_type() == torch::kLong, "token_map must be int64");
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(),
              "cache must be a contiguous CUDA tensor");
  TORCH_CHECK(cache.scalar_type() == torch::kLong, "cache must be int64");
  TORCH_CHECK(nslots >= 0, "nslots must be >= 0");
  TORCH_CHECK(cache.numel() >= nslots + 1,
              "cache must hold nslots + 1 entries, got ", cache.numel());
  TORCH_CHECK(token_map.numel() > 0, "token_map is empty");

  const int64_t num_tokens = input_ids.numel() == 0 ? 0 : input_ids.size(0);
  if (num_tokens == 0) {
    return;
  }
  TORCH_CHECK(slot_mapping.numel() >= num_tokens,
              "slot_mapping is shorter than input_ids");
  auto ids = require_cuda_long(input_ids, "input_ids");
  auto slots = require_cuda_long(slot_mapping, "slot_mapping");

  const at::cuda::OptionalCUDAGuard guard(device_of(cache));
  const int64_t blocks = (num_tokens + kThreads - 1) / kThreads;
  engram_write_hash_cache_kernel<<<blocks, kThreads, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int64_t>(), token_map.data_ptr<int64_t>(),
      slots.data_ptr<int64_t>(), cache.data_ptr<int64_t>(), ids.stride(0),
      slots.stride(0), nslots, token_map.numel(), num_tokens);
  AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor engram_ngram_hash(
    torch::Tensor& input_ids, torch::Tensor& positions, torch::Tensor& token_map,
    torch::Tensor& cache, torch::Tensor& block_table,
    torch::Tensor& token_to_req, torch::Tensor& query_start_loc,
    torch::Tensor& multipliers, torch::Tensor& primes, torch::Tensor& offsets,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    bool paged, const std::optional<torch::Tensor>& lookback_token_ids,
    bool lookback_is_prompt_only) {
  TORCH_CHECK(multipliers.is_cuda() && multipliers.is_contiguous(),
              "multipliers must be a contiguous CUDA tensor");
  TORCH_CHECK(primes.is_cuda() && primes.is_contiguous(),
              "primes must be a contiguous CUDA tensor");
  TORCH_CHECK(offsets.is_cuda() && offsets.is_contiguous(),
              "offsets must be a contiguous CUDA tensor");
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(),
              "cache must be a contiguous CUDA tensor");
  TORCH_CHECK(token_map.is_cuda() && token_map.is_contiguous(),
              "token_map must be a contiguous CUDA tensor");
  TORCH_CHECK(multipliers.scalar_type() == torch::kLong &&
                  primes.scalar_type() == torch::kLong &&
                  offsets.scalar_type() == torch::kLong &&
                  cache.scalar_type() == torch::kLong &&
                  token_map.scalar_type() == torch::kLong,
              "engram hash tables must be int64");
  TORCH_CHECK(primes.dim() == 2, "primes must be rank 2");
  const int64_t max_ngram_64 = multipliers.numel();
  const int64_t n_heads_64 = primes.size(1);
  TORCH_CHECK(max_ngram_64 >= 1 && max_ngram_64 <= 16, "max_ngram out of range");
  TORCH_CHECK(n_heads_64 >= 1 && n_heads_64 <= 256, "n_heads out of range");
  TORCH_CHECK(primes.size(0) == max_ngram_64 - 1,
              "primes rows must be max_ngram - 1");
  const int64_t n_cols = (max_ngram_64 - 1) * n_heads_64;
  if (lookback_token_ids.has_value()) {
    const auto& lookback = *lookback_token_ids;
    TORCH_CHECK(paged, "lookback_token_ids requires paged request metadata");
    TORCH_CHECK(lookback.is_cuda() && lookback.device() == cache.device() &&
                    lookback.scalar_type() == torch::kInt && lookback.dim() == 2 &&
                    lookback.stride(1) == 1,
                "lookback_token_ids must be a rank-2 CUDA int32 tensor");
    TORCH_CHECK(lookback.size(0) >= num_reqs &&
                    lookback.size(1) >= max_ngram_64 - 1,
                "lookback_token_ids is shorter than request/ngram history");
  }
  TORCH_CHECK(offsets.numel() == n_cols, "offsets numel must be n_hash_cols");
  TORCH_CHECK(token_map.numel() > 0, "token_map is empty");
  if (paged) {
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_reqs >= 1, "num_reqs must be >= 1");
    TORCH_CHECK(block_table.is_cuda() && block_table.dim() == 2,
                "block_table must be a rank-2 CUDA tensor");
    TORCH_CHECK(query_start_loc.is_cuda() && query_start_loc.numel() >= 1,
                "query_start_loc must be a non-empty CUDA tensor");
  }

  const int64_t num_tokens = input_ids.numel() == 0 ? 0 : input_ids.size(0);
  auto out = torch::empty(
      {num_tokens, n_cols},
      torch::TensorOptions().dtype(torch::kLong).device(cache.device()));
  if (num_tokens == 0 || n_cols == 0) {
    return out;
  }
  TORCH_CHECK(positions.numel() >= num_tokens, "positions is shorter than input_ids");

  auto ids = require_cuda_long(input_ids, "input_ids");
  auto pos = require_cuda_long(positions, "positions");
  torch::Tensor table;
  torch::Tensor req;
  torch::Tensor qsl;
  int64_t bt_stride0 = 0;
  int64_t bt_stride1 = 0;
  int64_t bt_rows = 0;
  int64_t bt_cols = 0;
  int64_t qsl_stride = 0;
  int64_t qsl_n = 0;
  int64_t req_stride = 0;
  if (paged) {
    TORCH_CHECK(token_to_req.numel() >= num_tokens,
                "token_to_req is shorter than input_ids");
    table = block_table.scalar_type() == torch::kInt && block_table.is_contiguous()
                ? block_table
                : block_table.to(torch::kInt).contiguous();
    req = require_cuda_long(token_to_req, "token_to_req");
    qsl = query_start_loc.scalar_type() == torch::kInt ? query_start_loc
                                                       : query_start_loc.to(torch::kInt);
    if (!qsl.is_contiguous()) {
      qsl = qsl.contiguous();
    }
    bt_stride0 = table.stride(0);
    bt_stride1 = table.stride(1);
    bt_rows = table.size(0);
    bt_cols = table.size(1);
    qsl_stride = qsl.stride(0);
    qsl_n = qsl.numel();
    req_stride = req.stride(0);
    TORCH_CHECK(bt_rows > 0, "block_table has no rows");
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(cache));
  const int64_t blocks = (num_tokens + kThreads - 1) / kThreads;
  engram_ngram_hash_kernel<<<blocks, kThreads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int64_t>(), pos.data_ptr<int64_t>(),
      token_map.data_ptr<int64_t>(), cache.data_ptr<int64_t>(),
      paged ? table.data_ptr<int32_t>() : nullptr, paged ? req.data_ptr<int64_t>() : nullptr,
      paged ? qsl.data_ptr<int32_t>() : nullptr, multipliers.data_ptr<int64_t>(),
      primes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(), ids.stride(0), pos.stride(0), req_stride, bt_stride0,
      bt_stride1, qsl_stride, static_cast<int>(max_ngram_64),
      static_cast<int>(n_heads_64), pad_id, nslots, block_size, num_reqs,
      token_map.numel(), num_tokens, bt_rows, bt_cols, qsl_n, cache.numel(), paged,
      lookback_token_ids.has_value() ? lookback_token_ids->data_ptr<int32_t>() : nullptr,
      lookback_token_ids.has_value() ? lookback_token_ids->stride(0) : 0,
      lookback_token_ids.has_value() ? lookback_token_ids->size(0) : 0,
      lookback_token_ids.has_value() ? lookback_token_ids->size(1) : 0,
      lookback_is_prompt_only);
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor engram_ngram_hash_meta(
    torch::Tensor& input_ids, torch::Tensor& positions, torch::Tensor& token_map,
    torch::Tensor& cache, torch::Tensor& block_table,
    torch::Tensor& token_to_req, torch::Tensor& query_start_loc,
    torch::Tensor& multipliers, torch::Tensor& primes, torch::Tensor& offsets,
    int64_t pad_id, int64_t nslots, int64_t block_size, int64_t num_reqs,
    bool paged, const std::optional<torch::Tensor>& lookback_token_ids,
    bool lookback_is_prompt_only) {
  (void)positions;
  (void)token_map;
  (void)cache;
  (void)block_table;
  (void)token_to_req;
  (void)query_start_loc;
  (void)pad_id;
  (void)nslots;
  (void)block_size;
  (void)num_reqs;
  (void)paged;
  (void)lookback_token_ids;
  (void)lookback_is_prompt_only;
  const int64_t num_tokens = input_ids.numel() == 0 ? 0 : input_ids.size(0);
  const int64_t n_heads = primes.dim() == 2 ? primes.size(1) : 0;
  const int64_t groups = multipliers.numel() > 0 ? multipliers.numel() - 1 : 0;
  const int64_t n_cols = groups * n_heads;
  TORCH_CHECK(offsets.numel() == n_cols || n_cols == 0,
              "offsets numel must be n_hash_cols");
  return torch::empty({num_tokens, n_cols}, input_ids.options().dtype(torch::kLong));
}

torch::Tensor engram_lookup_fp8(torch::Tensor& indices, torch::Tensor& weight,
                                torch::Tensor& scale, int64_t vocab_start,
                                int64_t vocab_end) {
  TORCH_CHECK(indices.is_cuda() && indices.dim() == 2, "indices must be CUDA [T, C]");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
              "weight must be a contiguous CUDA tensor");
  TORCH_CHECK(scale.is_cuda() && scale.is_contiguous(),
              "scale must be a contiguous CUDA tensor");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kUInt8, "scale must be uint8");
  TORCH_CHECK(weight.dim() == 2 && scale.dim() == 2, "weight and scale must be rank 2");
  const int64_t dim64 = weight.size(1);
  TORCH_CHECK(dim64 > 0 && dim64 <= 8192 && (dim64 % 32) == 0,
              "engram dim must be a positive multiple of 32");
  TORCH_CHECK(scale.size(0) == weight.size(0), "scale rows must match weight");
  TORCH_CHECK(scale.size(1) == dim64 / 32, "scale cols must be dim / 32");
  TORCH_CHECK(vocab_end >= vocab_start, "vocab range is inverted");
  const int64_t part_rows = vocab_end - vocab_start;
  TORCH_CHECK(part_rows <= weight.size(0),
              "vocab shard is larger than the weight table");

  const int64_t num_tokens = indices.size(0);
  const int64_t n_cols = indices.size(1);
  auto out = torch::empty(
      {num_tokens, n_cols, dim64},
      torch::TensorOptions().dtype(torch::kBFloat16).device(indices.device()));
  if (num_tokens == 0 || n_cols == 0) {
    return out;
  }
  auto ids = indices.scalar_type() == torch::kLong ? indices : indices.to(torch::kLong);

  const at::cuda::OptionalCUDAGuard guard(device_of(indices));
  const int64_t rows = num_tokens * n_cols;
  engram_lookup_fp8_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int64_t>(), weight.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ids.stride(0), ids.stride(1),
      vocab_start, vocab_end, static_cast<int>(dim64), static_cast<int>(n_cols),
      num_tokens);
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor engram_lookup_fp8_meta(torch::Tensor& indices, torch::Tensor& weight,
                                     torch::Tensor& scale, int64_t vocab_start,
                                     int64_t vocab_end) {
  (void)scale;
  (void)vocab_start;
  (void)vocab_end;
  const int64_t num_tokens = indices.dim() >= 1 ? indices.size(0) : 0;
  const int64_t n_cols = indices.dim() >= 2 ? indices.size(1) : 0;
  const int64_t dim = weight.dim() >= 2 ? weight.size(1) : 0;
  return torch::empty({num_tokens, n_cols, dim},
                      indices.options().dtype(torch::kBFloat16));
}

torch::Tensor engram_register_host(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "engram_register_host expects a CPU tensor");
  TORCH_CHECK(cpu_tensor.is_contiguous(),
              "engram_register_host expects a contiguous tensor");
  if (cpu_tensor.numel() == 0) {
    return torch::empty(cpu_tensor.sizes(),
                        cpu_tensor.options().device(torch::kCUDA));
  }

  const long page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK(page_size > 0, "sysconf(_SC_PAGESIZE) failed");
  const uintptr_t page = static_cast<uintptr_t>(page_size);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(cpu_tensor.data_ptr());
  const uintptr_t base = addr & ~(page - 1);
  size_t span = static_cast<size_t>(addr - base) + cpu_tensor.nbytes();
  span = (span + static_cast<size_t>(page) - 1) & ~(static_cast<size_t>(page) - 1);

  void* host = reinterpret_cast<void*>(base);
  // cudaHostRegisterReadOnly is 0x08 (CUDA 11.1). The engram table is a
  // PROT_READ mmap, which the driver rejects without this flag.
  constexpr unsigned int kReadOnly = 0x08;
  bool registered_here = false;
  cudaError_t err = cudaHostRegister(host, span, cudaHostRegisterMapped | kReadOnly);
  if (err != cudaSuccess) {
    cudaGetLastError();
    err = cudaHostRegister(host, span, cudaHostRegisterMapped);
  }
  if (err != cudaSuccess) {
    cudaGetLastError();
    err = cudaHostRegister(host, span, cudaHostRegisterDefault);
  }
  if (err == cudaSuccess) {
    registered_here = true;
  } else if (err != cudaErrorHostMemoryAlreadyRegistered) {
    cudaGetLastError();
    TORCH_CHECK(false, "cudaHostRegister failed: ", cudaGetErrorString(err));
  } else {
    cudaGetLastError();
  }

  void* device = nullptr;
  err = cudaHostGetDevicePointer(&device, host, 0);
  if (err != cudaSuccess) {
    if (registered_here) {
      cudaHostUnregister(host);
    }
    TORCH_CHECK(false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }
  void* data = static_cast<char*>(device) + (addr - base);

  struct Holder {
    torch::Tensor keep;
    void* host = nullptr;
    bool unregister = false;
  };
  auto holder = std::make_shared<Holder>();
  holder->keep = cpu_tensor;
  holder->host = host;
  holder->unregister = registered_here;
  return torch::from_blob(
      data, cpu_tensor.sizes(), cpu_tensor.strides(),
      [holder](void*) {
        if (holder->unregister && holder->host != nullptr) {
          cudaHostUnregister(holder->host);
          holder->unregister = false;
        }
      },
      cpu_tensor.options().device(torch::kCUDA));
}

}  // namespace vllm_iluvatar
