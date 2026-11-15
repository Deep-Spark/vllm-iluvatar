// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA equivalents of the Triton sampling kernels used by DSpark.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <torch/library.h>
#include <c10/cuda/CUDAException.h>

namespace vllm_iluvatar {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxWarps = 32;

template <typename T>
__device__ __forceinline__ float load_float(const T* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t index,
                                               bool is_int64) {
  return is_int64 ? static_cast<const int64_t*>(ptr)[index]
                  : static_cast<const int32_t*>(ptr)[index];
}

template <typename acc_t>
struct MaxPair {
  acc_t value;
  int64_t index;
};

__device__ __forceinline__ int64_t shfl_down_i64(int64_t value, int offset) {
  const uint64_t bits = static_cast<uint64_t>(value);
  const uint32_t lo = __shfl_down(static_cast<uint32_t>(bits), offset);
  const uint32_t hi =
      __shfl_down(static_cast<uint32_t>(bits >> 32), offset);
  return static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo);
}

template <typename acc_t>
__device__ __forceinline__ MaxPair<acc_t> choose_max(MaxPair<acc_t> a,
                                                      MaxPair<acc_t> b) {
  if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
    return b;
  }
  return a;
}

template <typename acc_t>
__device__ __forceinline__ MaxPair<acc_t> block_max_pair(
    MaxPair<acc_t> value, acc_t* shared_values, int64_t* shared_indices) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    MaxPair<acc_t> other{
        __shfl_down(value.value, offset),
        shfl_down_i64(value.index, offset),
    };
    if (lane + offset < warpSize) {
      value = choose_max(value, other);
    }
  }
  if (lane == 0) {
    shared_values[warp] = value.value;
    shared_indices[warp] = value.index;
  }
  __syncthreads();
  if (warp == 0) {
    MaxPair<acc_t> first{
        lane < num_warps ? shared_values[lane]
                         : -std::numeric_limits<acc_t>::infinity(),
        lane < num_warps ? shared_indices[lane]
                         : std::numeric_limits<int64_t>::max(),
    };
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      MaxPair<acc_t> other{
          __shfl_down(first.value, offset),
          shfl_down_i64(first.index, offset),
      };
      if (lane + offset < warpSize) {
        first = choose_max(first, other);
      }
    }
    if (lane == 0) {
      shared_values[0] = first.value;
      shared_indices[0] = first.index;
    }
  }
  __syncthreads();
  return {shared_values[0], shared_indices[0]};
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = warp == 0 && lane < num_warps ? shared[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down(value, offset);
    }
    if (lane == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ uint32_t rotl32(uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

__device__ __forceinline__ uint32_t murmur_mix(uint32_t hash, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  hash ^= key;
  hash = rotl32(hash, 13);
  return hash * 5u + 0xE6546B64u;
}

__device__ __forceinline__ uint32_t murmur_fmix(uint32_t hash) {
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  return hash ^ (hash >> 16);
}

__device__ __forceinline__ uint32_t murmur_hash(
    int64_t seed, int64_t pos, uint32_t offset, uint32_t domain = 0) {
  uint32_t hash = domain;
  hash = murmur_mix(hash, static_cast<uint32_t>(seed));
  hash = murmur_mix(hash, static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32));
  hash = murmur_mix(hash, static_cast<uint32_t>(pos));
  hash = murmur_mix(hash, offset);
  return murmur_fmix(hash ^ 16u);
}

__device__ __forceinline__ float murmur_uniform32(int64_t seed, int64_t pos,
                                                   uint32_t offset) {
  const uint32_t random = murmur_hash(seed, pos, offset);
  const int32_t hi = static_cast<int32_t>(random >> 16);
  const int32_t lo = static_cast<int32_t>(random & 0xFFFFu);
  return static_cast<float>(hi) * 1.52587890625e-05f +
         (static_cast<float>(lo) + 0.5f) * 2.3283064365386963e-10f;
}

__device__ __forceinline__ double murmur_uniform64(int64_t seed, int64_t pos,
                                                    uint32_t offset) {
  const uint64_t lo = murmur_hash(seed, pos, offset);
  const uint64_t hi = murmur_hash(seed, pos, offset, 0x9E3779B9u);
  const uint64_t random53 = ((hi << 32) | lo) >> 11;
  double value = (static_cast<double>(random53) + 0.5) *
                 1.1102230246251565e-16;
  return value == 1.0 ? value - 1.1102230246251565e-16 : value;
}

__device__ __forceinline__ float log1p_neg_stable(float value) {
  float polynomial = 1.0f / 8.0f;
  polynomial = 1.0f / 7.0f + value * polynomial;
  polynomial = 1.0f / 6.0f + value * polynomial;
  polynomial = 1.0f / 5.0f + value * polynomial;
  polynomial = 1.0f / 4.0f + value * polynomial;
  polynomial = 1.0f / 3.0f + value * polynomial;
  polynomial = 1.0f / 2.0f + value * polynomial;
  polynomial = 1.0f + value * polynomial;
  const float series = -value * polynomial;
  const float direct = logf(fmaxf(1.0f - value, 5.960464477539063e-08f));
  return value < 0.25f ? series : direct;
}

template <typename acc_t>
__device__ __forceinline__ acc_t add_gumbel(acc_t logit, int64_t seed,
                                             int64_t pos, uint32_t token) {
  if constexpr (std::is_same_v<acc_t, double>) {
    const double uniform = murmur_uniform64(seed, pos, token);
    return logit - log(-log(uniform));
  } else {
    const float uniform = murmur_uniform32(seed, pos, token);
    return logit - logf(-log1p_neg_stable(uniform));
  }
}

__device__ __forceinline__ uint32_t mul_hi(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

__device__ __forceinline__ uint32_t philox_rand(int64_t seed,
                                                uint32_t offset) {
  uint32_t c0 = offset;
  uint32_t c1 = 0;
  uint32_t c2 = 0;
  uint32_t c3 = 0;
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t old_c0 = c0;
    const uint32_t old_c2 = c2;
    c0 = mul_hi(0xCD9E8D57u, old_c2) ^ c1 ^ k0;
    c2 = mul_hi(0xD2511F53u, old_c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * old_c2;
    c3 = 0xD2511F53u * old_c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

__device__ __forceinline__ float triton_uniform(int64_t seed,
                                                uint32_t offset) {
  const int32_t bits = static_cast<int32_t>(philox_rand(seed, offset));
  const int64_t magnitude =
      bits < 0 ? -static_cast<int64_t>(bits) - 1 : bits;
  return static_cast<float>(magnitude) * 4.6566127342e-10f;
}

template <typename scalar_t, typename acc_t>
__global__ void bias_gumbel_sample_kernel(
    int64_t* local_argmax, int64_t local_argmax_stride, acc_t* local_max,
    int64_t local_max_stride, scalar_t* logits_cache,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const int32_t* logits_cache_col, const scalar_t* logits,
    int64_t logits_stride, const scalar_t* bias, int64_t bias_stride,
    int64_t valid_vocab, const void* expanded_idx_mapping,
    const int64_t* seeds, const int64_t* positions, const float* temperatures,
    int vocab_size, int block_size, bool is_drafting, bool apply_temperature,
    bool per_token_col, bool mapping_int64, int num_tokens) {
  const int token_row = blockIdx.x;
  const int vocab_block = blockIdx.y;
  if (token_row >= num_tokens) {
    return;
  }
  const int state = load_index(expanded_idx_mapping, token_row, mapping_int64);
  const bool valid_state = state >= 0;
  const float temp = valid_state ? temperatures[state] : 0.0f;
  const int64_t seed = valid_state ? seeds[state] : 0;
  int64_t pos = positions[token_row];
  if (is_drafting) {
    pos += (1LL << 30);
  }
  int cache_col = 0;
  if (logits_cache != nullptr) {
    cache_col = per_token_col ? logits_cache_col[token_row] : logits_cache_col[0];
  }

  const int begin = vocab_block * block_size;
  const int end = min(begin + block_size, vocab_size);
  // Seed the index with this block's first token, never INT64_MAX. choose_max
  // never selects a NaN, so an all-NaN block would otherwise leave the
  // sentinel in place and emit an out-of-range token id. Triton computes
  // `block_idx * BLOCK_SIZE + idx` with idx=0 in that case, which is what this
  // matches. The id is read straight back by markov_embed and by the target's
  // embedding, and an out-of-range lookup does not fault here: it silently
  // reads out-of-bounds memory.
  MaxPair<acc_t> best{-std::numeric_limits<acc_t>::infinity(), begin};
  for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const int64_t row = token_row;
    // Preserve the separate add's dtype rounding before caching or sampling.
    const scalar_t raw = token < valid_vocab
        ? static_cast<scalar_t>(
              static_cast<float>(logits[row * logits_stride + token]) +
              static_cast<float>(bias[row * bias_stride + token]))
        : static_cast<scalar_t>(-INFINITY);
    if (logits_cache != nullptr && valid_state) {
      logits_cache[static_cast<int64_t>(state) * logits_cache_stride_0 +
                   static_cast<int64_t>(cache_col) * logits_cache_stride_1 +
                   token] = raw;
    }
    acc_t value = static_cast<acc_t>(static_cast<float>(raw));
    if (temp != 0.0f && apply_temperature) {
      value /= static_cast<acc_t>(temp);
    }
    if (temp != 0.0f) {
      // CoreX lowers Triton's vector RNG one value per hardware warp. Match
      // the baseline exactly: all 64 lanes use the warp's first token key.
      const uint32_t noise_key = token - token % warpSize;
      value = add_gumbel(value, seed, pos, noise_key);
    }
    best = choose_max(best, MaxPair<acc_t>{value, token});
  }
  __shared__ acc_t shared_values[kMaxWarps];
  __shared__ int64_t shared_indices[kMaxWarps];
  best = block_max_pair(best, shared_values, shared_indices);
  if (threadIdx.x == 0) {
    local_argmax[static_cast<int64_t>(token_row) * local_argmax_stride +
                 vocab_block] = best.index;
    local_max[static_cast<int64_t>(token_row) * local_max_stride + vocab_block] =
        best.value;
  }
}


}  // namespace

void ilu_fused_bias_gumbel_sample(
    torch::Tensor& local_argmax, int64_t local_argmax_stride,
    torch::Tensor& local_max, int64_t local_max_stride,
    const std::optional<torch::Tensor>& logits_cache,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const std::optional<torch::Tensor>& logits_cache_col,
    torch::Tensor& logits, int64_t logits_stride,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& seeds,
    torch::Tensor& positions, torch::Tensor& temperatures, int64_t vocab_size,
    int64_t block_size, bool is_drafting, bool apply_temperature,
    bool use_fp64, bool per_token_col, torch::Tensor& bias,
    int64_t bias_stride, int64_t num_vocab_padding) {
  TORCH_CHECK(logits.dim() == 2 && bias.sizes() == logits.sizes());
  TORCH_CHECK(logits.is_cuda() && bias.device() == logits.device());
  TORCH_CHECK(logits.scalar_type() == bias.scalar_type());
  TORCH_CHECK(logits.stride(1) == 1 && bias.stride(1) == 1);
  TORCH_CHECK(vocab_size == logits.size(1));
  TORCH_CHECK(num_vocab_padding >= 0 && num_vocab_padding <= vocab_size);
  TORCH_CHECK(!logits_cache || logits_cache->scalar_type() == logits.scalar_type());
  const int num_tokens = logits.size(0);
  const int num_blocks = (vocab_size + block_size - 1) / block_size;
  if (num_tokens == 0 || num_blocks == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_tokens, num_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "ilu_fused_bias_gumbel_sample", [&] {
        scalar_t* cache =
            logits_cache ? logits_cache->data_ptr<scalar_t>() : nullptr;
        const int32_t* cols =
            logits_cache_col ? logits_cache_col->data_ptr<int32_t>() : nullptr;
        if (use_fp64) {
          bias_gumbel_sample_kernel<scalar_t, double><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), local_argmax_stride,
              local_max.data_ptr<double>(), local_max_stride, cache,
              logits_cache_stride_0, logits_cache_stride_1, cols,
              logits.data_ptr<scalar_t>(), logits_stride,
              bias.data_ptr<scalar_t>(), bias_stride, vocab_size - num_vocab_padding,
              expanded_idx_mapping.data_ptr(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              temperatures.data_ptr<float>(), vocab_size, block_size,
              is_drafting, apply_temperature, per_token_col,
              expanded_idx_mapping.scalar_type() == at::kLong, num_tokens);
        } else {
          bias_gumbel_sample_kernel<scalar_t, float><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), local_argmax_stride,
              local_max.data_ptr<float>(), local_max_stride, cache,
              logits_cache_stride_0, logits_cache_stride_1, cols,
              logits.data_ptr<scalar_t>(), logits_stride,
              bias.data_ptr<scalar_t>(), bias_stride, vocab_size - num_vocab_padding,
              expanded_idx_mapping.data_ptr(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              temperatures.data_ptr<float>(), vocab_size, block_size,
              is_drafting, apply_temperature, per_token_col,
              expanded_idx_mapping.scalar_type() == at::kLong, num_tokens);
        }
      });
}


}  // namespace vllm_iluvatar

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("ilu_fused_bias_gumbel_sample("
      "Tensor! local_argmax, int local_argmax_stride, Tensor! local_max, "
      "int local_max_stride, Tensor(a!)? logits_cache, "
      "int logits_cache_stride_0, int logits_cache_stride_1, "
      "Tensor? logits_cache_col, Tensor logits, int logits_stride, "
      "Tensor expanded_idx_mapping, Tensor seeds, Tensor positions, "
      "Tensor temperatures, int vocab_size, int block_size, "
      "bool is_drafting, bool apply_temperature, bool use_fp64, "
      "bool per_token_col, Tensor bias, int bias_stride, int num_vocab_padding) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("ilu_fused_bias_gumbel_sample", &vllm_iluvatar::ilu_fused_bias_gumbel_sample);
}
