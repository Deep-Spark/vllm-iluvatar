// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA equivalents of the Triton sampling kernels used by DSpark.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "ops.h"

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
__global__ void gumbel_sample_kernel(
    int64_t* local_argmax, int64_t local_argmax_stride, acc_t* local_max,
    int64_t local_max_stride, scalar_t* logits_cache,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const int32_t* logits_cache_col, const scalar_t* logits,
    int64_t logits_stride, const void* expanded_idx_mapping,
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

  MaxPair<acc_t> best{-std::numeric_limits<acc_t>::infinity(),
                      std::numeric_limits<int64_t>::max()};
  const int begin = vocab_block * block_size;
  const int end = min(begin + block_size, vocab_size);
  for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const scalar_t raw = logits[static_cast<int64_t>(token_row) * logits_stride +
                                token];
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

template <typename scalar_t>
__global__ void compute_local_logits_stats_kernel(
    int64_t* target_local_argmax, int64_t target_argmax_stride,
    float* target_local_max, int64_t target_max_stride,
    float* target_local_sumexp, int64_t target_sumexp_stride,
    float* draft_local_max, int64_t draft_max_stride,
    float* draft_local_sumexp, int64_t draft_sumexp_stride,
    const scalar_t* target_logits, int64_t target_logits_stride,
    const scalar_t* draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, const void* expanded_idx_mapping,
    const int32_t* expanded_local_pos, const float* temperatures,
    int vocab_size, int num_speculative_steps, int vocab_block_size,
    bool has_draft_logits, bool mapping_int64, int num_logits) {
  const int logit = blockIdx.x;
  const int vocab_block = blockIdx.y;
  if (logit >= num_logits) {
    return;
  }
  const int draft_step = expanded_local_pos[logit];
  if (draft_step >= num_speculative_steps) {
    return;
  }
  const int state = load_index(expanded_idx_mapping, logit, mapping_int64);
  const float temp = temperatures[state];
  const int begin = vocab_block * vocab_block_size;
  const int end = min(begin + vocab_block_size, vocab_size);

  MaxPair<float> target_best{-INFINITY, INT64_MAX};
  for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const float value =
        load_float(target_logits,
                   static_cast<int64_t>(logit) * target_logits_stride + token);
    target_best = choose_max(target_best, MaxPair<float>{value, token});
  }
  __shared__ float max_shared[kMaxWarps];
  __shared__ int64_t index_shared[kMaxWarps];
  target_best = block_max_pair(target_best, max_shared, index_shared);
  if (temp == 0.0f) {
    if (threadIdx.x == 0) {
      target_local_argmax[static_cast<int64_t>(logit) *
                              target_argmax_stride +
                          vocab_block] = target_best.index;
      target_local_max[static_cast<int64_t>(logit) * target_max_stride +
                       vocab_block] = target_best.value;
    }
    return;
  }

  float partial = 0.0f;
  if (target_best.value > -INFINITY) {
    for (int token = begin + threadIdx.x; token < end;
         token += blockDim.x) {
      const float value =
          load_float(target_logits,
                     static_cast<int64_t>(logit) * target_logits_stride + token);
      partial += expf(value - target_best.value);
    }
  }
  __shared__ float sum_shared[kMaxWarps];
  const float target_sum = block_sum(partial, sum_shared);
  if (threadIdx.x == 0) {
    target_local_max[static_cast<int64_t>(logit) * target_max_stride +
                     vocab_block] = target_best.value;
    target_local_sumexp[static_cast<int64_t>(logit) * target_sumexp_stride +
                        vocab_block] = target_sum;
  }
  if (!has_draft_logits) {
    return;
  }

  MaxPair<float> draft_best{-INFINITY, INT64_MAX};
  for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const float value =
        load_float(draft_logits,
                   static_cast<int64_t>(state) * draft_stride_0 +
                       static_cast<int64_t>(draft_step) * draft_stride_1 +
                       token) /
        temp;
    draft_best = choose_max(draft_best, MaxPair<float>{value, token});
  }
  draft_best = block_max_pair(draft_best, max_shared, index_shared);
  partial = 0.0f;
  if (draft_best.value > -INFINITY) {
    for (int token = begin + threadIdx.x; token < end;
         token += blockDim.x) {
      const float value =
          load_float(draft_logits,
                     static_cast<int64_t>(state) * draft_stride_0 +
                         static_cast<int64_t>(draft_step) * draft_stride_1 +
                         token) /
          temp;
      partial += expf(value - draft_best.value);
    }
  }
  const float draft_sum = block_sum(partial, sum_shared);
  if (threadIdx.x == 0) {
    draft_local_max[static_cast<int64_t>(logit) * draft_max_stride +
                    vocab_block] = draft_best.value;
    draft_local_sumexp[static_cast<int64_t>(logit) * draft_sumexp_stride +
                       vocab_block] = draft_sum;
  }
}

__device__ __forceinline__ float global_lse(
    const float* local_max, int64_t max_stride, const float* local_sumexp,
    int64_t sum_stride, int row, int num_blocks) {
  float maximum = -INFINITY;
  for (int block = 0; block < num_blocks; ++block) {
    maximum = fmaxf(maximum, local_max[static_cast<int64_t>(row) * max_stride +
                                      block]);
  }
  float sum = 0.0f;
  for (int block = 0; block < num_blocks; ++block) {
    const float value =
        local_max[static_cast<int64_t>(row) * max_stride + block];
    sum += local_sumexp[static_cast<int64_t>(row) * sum_stride + block] *
           expf(value - maximum);
  }
  return maximum + logf(sum);
}

template <typename scalar_t>
__global__ void rejection_kernel(
    int64_t* sampled, int64_t sampled_stride, int32_t* rejected_steps,
    float* target_rejected_lse, float* draft_rejected_lse,
    const scalar_t* target_logits, int64_t target_logits_stride,
    const int64_t* target_local_argmax, int64_t target_argmax_stride,
    const float* target_local_max, int64_t target_max_stride,
    const float* target_local_sumexp, int64_t target_sumexp_stride,
    const void* draft_sampled, const scalar_t* draft_logits,
    int64_t draft_stride_0, int64_t draft_stride_1,
    const float* draft_local_max, int64_t draft_max_stride,
    const float* draft_local_sumexp, int64_t draft_sumexp_stride,
    const int32_t* cu_num_logits, const void* idx_mapping,
    const float* temperatures, const int64_t* seeds, const int64_t* positions,
    const float* synthetic_rates, int vocab_num_blocks, bool has_draft_logits,
    bool synthetic_mode, bool mapping_int64, bool draft_sampled_int64,
    int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs || threadIdx.x != 0) {
    return;
  }
  const int state = load_index(idx_mapping, req, mapping_int64);
  const int begin = cu_num_logits[req];
  const int end = cu_num_logits[req + 1];
  const int num_draft = end - begin - 1;
  const float temp = temperatures[state];
  bool verifying = true;
  int accepted_count = 0;
  float target_lse = 0.0f;
  float draft_lse = 0.0f;
  for (int step = 0; step < num_draft; ++step) {
    const int logit = begin + step;
    int64_t proposal =
        load_index(draft_sampled, logit + 1, draft_sampled_int64);
    const bool valid = proposal >= 0;
    proposal = proposal > 0 ? proposal : 0;
    if (temp != 0.0f) {
      verifying = verifying && valid;
    }
    if (!verifying) {
      continue;
    }
    const float uniform =
        fmaxf(triton_uniform(seeds[state], static_cast<uint32_t>(positions[logit])),
              4.6566127342e-10f);
    if (temp == 0.0f) {
      float best_value = -INFINITY;
      int best_block = 0;
      for (int block = 0; block < vocab_num_blocks; ++block) {
        float value =
            target_local_max[static_cast<int64_t>(logit) * target_max_stride +
                             block];
        if (isnan(value)) {
          value = -INFINITY;
        }
        if (value > best_value) {
          best_value = value;
          best_block = block;
        }
      }
      const int64_t target =
          target_local_argmax[static_cast<int64_t>(logit) *
                                  target_argmax_stride +
                              best_block];
      const bool accepted =
          valid && (synthetic_mode ? uniform < synthetic_rates[step]
                                   : target == proposal);
      verifying = accepted;
      accepted_count += accepted ? 1 : 0;
      sampled[static_cast<int64_t>(req) * sampled_stride + step] =
          accepted ? proposal : target;
    } else {
      target_lse =
          global_lse(target_local_max, target_max_stride, target_local_sumexp,
                     target_sumexp_stride, logit, vocab_num_blocks);
      const float target_logprob =
          load_float(target_logits,
                     static_cast<int64_t>(logit) * target_logits_stride +
                         proposal) -
          target_lse;
      float draft_logprob = 0.0f;
      draft_lse = 0.0f;
      if (has_draft_logits) {
        draft_lse =
            global_lse(draft_local_max, draft_max_stride, draft_local_sumexp,
                       draft_sumexp_stride, logit, vocab_num_blocks);
        draft_logprob =
            load_float(draft_logits,
                       static_cast<int64_t>(state) * draft_stride_0 +
                           static_cast<int64_t>(step) * draft_stride_1 +
                           proposal) /
                temp -
            draft_lse;
      }
      const bool accepted =
          synthetic_mode ? uniform < synthetic_rates[step]
                         : target_logprob > logf(uniform) + draft_logprob;
      verifying = accepted;
      accepted_count += accepted ? 1 : 0;
      sampled[static_cast<int64_t>(req) * sampled_stride + step] = proposal;
    }
  }
  rejected_steps[req] = accepted_count;
  target_rejected_lse[req] = target_lse;
  draft_rejected_lse[req] = draft_lse;
}

template <typename scalar_t, typename acc_t>
__global__ void resample_kernel(
    int64_t* local_argmax, int64_t argmax_stride, acc_t* local_max,
    int64_t max_stride, const scalar_t* target_logits,
    int64_t target_logits_stride, const float* target_rejected_lse,
    const scalar_t* draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, const float* draft_rejected_lse,
    const int32_t* rejected_step, const int32_t* cu_num_logits,
    const void* expanded_idx_mapping, const void* draft_sampled,
    const float* temperatures, const int64_t* seeds, const int64_t* positions,
    const float* cumulative_log_p, int vocab_size, int block_size,
    bool has_draft_logits, bool use_block_verification, bool mapping_int64,
    bool draft_sampled_int64, int num_reqs) {
  const int req = blockIdx.x;
  const int vocab_block = blockIdx.y;
  if (req >= num_reqs) {
    return;
  }
  const int rejected = rejected_step[req];
  const int begin = cu_num_logits[req];
  const int end = cu_num_logits[req + 1];
  const int resample_row = begin + rejected;
  const int state =
      load_index(expanded_idx_mapping, resample_row, mapping_int64);
  const float temp = temperatures[state];
  const bool bonus = resample_row == end - 1;
  if (temp == 0.0f && !bonus) {
    return;
  }
  const int64_t rejected_token =
      bonus ? 0
            : load_index(draft_sampled, resample_row + 1,
                         draft_sampled_int64);
  const bool valid_rejected = rejected_token >= 0;
  const int vocab_begin = vocab_block * block_size;
  const int vocab_end = min(vocab_begin + block_size, vocab_size);
  MaxPair<acc_t> best{-std::numeric_limits<acc_t>::infinity(),
                      std::numeric_limits<int64_t>::max()};
  for (int token = vocab_begin + threadIdx.x; token < vocab_end;
       token += blockDim.x) {
    const float target =
        load_float(target_logits,
                   static_cast<int64_t>(resample_row) * target_logits_stride +
                       token);
    float residual = target;
    if (!bonus && valid_rejected) {
      if (has_draft_logits) {
        float target_logprob = target - target_rejected_lse[req];
        if (use_block_verification && rejected > 0) {
          target_logprob += cumulative_log_p[resample_row - 1];
        }
        const float draft_logprob =
            load_float(draft_logits,
                       static_cast<int64_t>(state) * draft_stride_0 +
                           static_cast<int64_t>(rejected) * draft_stride_1 +
                           token) /
                temp -
            draft_rejected_lse[req];
        const float ratio = expf(draft_logprob - target_logprob);
        residual =
            ratio < 1.0f ? target_logprob + log1pf(-ratio) : -INFINITY;
      } else if (token == rejected_token) {
        residual = -INFINITY;
      }
    }
    acc_t value = static_cast<acc_t>(residual);
    if (temp != 0.0f) {
      const uint32_t noise_key = token - token % warpSize;
      value =
          add_gumbel(value, seeds[state], positions[resample_row], noise_key);
    }
    best = choose_max(best, MaxPair<acc_t>{value, token});
  }
  __shared__ acc_t shared_values[kMaxWarps];
  __shared__ int64_t shared_indices[kMaxWarps];
  best = block_max_pair(best, shared_values, shared_indices);
  if (threadIdx.x == 0) {
    local_argmax[static_cast<int64_t>(req) * argmax_stride + vocab_block] =
        best.index;
    local_max[static_cast<int64_t>(req) * max_stride + vocab_block] =
        best.value;
  }
}

template <typename acc_t>
__global__ void insert_resampled_kernel(
    int64_t* sampled, int64_t sampled_stride, int32_t* num_sampled,
    const int64_t* local_argmax, int64_t argmax_stride,
    const acc_t* local_max, int64_t max_stride, int resample_num_blocks,
    const int32_t* cu_num_logits, const void* expanded_idx_mapping,
    const float* temperatures, bool mapping_int64, int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs || threadIdx.x != 0) {
    return;
  }
  const int count = num_sampled[req];
  const int begin = cu_num_logits[req];
  const int end = cu_num_logits[req + 1];
  const int row = begin + count;
  const int state = load_index(expanded_idx_mapping, row, mapping_int64);
  num_sampled[req] = count + 1;
  if (temperatures[state] == 0.0f && row != end - 1) {
    return;
  }
  acc_t best = -std::numeric_limits<acc_t>::infinity();
  int best_block = 0;
  for (int block = 0; block < resample_num_blocks; ++block) {
    acc_t value = local_max[static_cast<int64_t>(req) * max_stride + block];
    if (isnan(static_cast<double>(value))) {
      value = -std::numeric_limits<acc_t>::infinity();
    }
    if (value > best) {
      best = value;
      best_block = block;
    }
  }
  sampled[static_cast<int64_t>(req) * sampled_stride + count] =
      local_argmax[static_cast<int64_t>(req) * argmax_stride + best_block];
}

template <typename scalar_t>
__global__ void greedy_argmax_partials_kernel(
    float* partial_max, int64_t* partial_argmax, const scalar_t* logits,
    int64_t logits_stride, int vocab_size, int vocab_block_size,
    int num_rows) {
  const int row = blockIdx.x;
  const int vocab_block = blockIdx.y;
  if (row >= num_rows) {
    return;
  }
  const int begin = vocab_block * vocab_block_size;
  const int end = min(begin + vocab_block_size, vocab_size);
  MaxPair<float> best{-INFINITY, INT64_MAX};
  for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
    float value =
        static_cast<float>(logits[static_cast<int64_t>(row) * logits_stride +
                                  token]);
    if (isnan(value)) {
      value = -INFINITY;
    }
    best = choose_max(best, MaxPair<float>{value, token});
  }
  __shared__ float shared_values[kMaxWarps];
  __shared__ int64_t shared_indices[kMaxWarps];
  best = block_max_pair(best, shared_values, shared_indices);
  if (threadIdx.x == 0) {
    const int64_t output =
        static_cast<int64_t>(row) * gridDim.y + vocab_block;
    partial_max[output] = best.value;
    partial_argmax[output] = best.index;
  }
}

__device__ __forceinline__ int64_t reduce_greedy_argmax(
    const float* partial_max, const int64_t* partial_argmax, int row,
    int vocab_num_blocks) {
  MaxPair<float> best{-INFINITY, INT64_MAX};
  const int64_t base = static_cast<int64_t>(row) * vocab_num_blocks;
  for (int block = 0; block < vocab_num_blocks; ++block) {
    float value = partial_max[base + block];
    if (isnan(value)) {
      value = -INFINITY;
    }
    best = choose_max(
        best, MaxPair<float>{value, partial_argmax[base + block]});
  }
  return best.index;
}

__global__ void greedy_rejection_finalize_kernel(
    int64_t* sampled, int64_t sampled_stride, int32_t* num_sampled,
    const float* partial_max, const int64_t* partial_argmax,
    const void* draft_sampled, const int32_t* cu_num_logits,
    int vocab_num_blocks, int max_speculative_steps,
    bool draft_sampled_int64, int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs || threadIdx.x != 0) {
    return;
  }
  const int begin = cu_num_logits[req];
  const int end = cu_num_logits[req + 1];
  const int num_draft = end - begin - 1;
  if (num_draft < 0 || num_draft > max_speculative_steps) {
    num_sampled[req] = 0;
    return;
  }
  for (int step = 0; step < num_draft; ++step) {
    const int row = begin + step;
    const int64_t target = reduce_greedy_argmax(
        partial_max, partial_argmax, row, vocab_num_blocks);
    const int64_t proposal =
        load_index(draft_sampled, row + 1, draft_sampled_int64);
    if (proposal >= 0 && proposal == target) {
      sampled[static_cast<int64_t>(req) * sampled_stride + step] =
          proposal;
      continue;
    }
    sampled[static_cast<int64_t>(req) * sampled_stride + step] = target;
    num_sampled[req] = step + 1;
    return;
  }
  const int64_t bonus = reduce_greedy_argmax(
      partial_max, partial_argmax, end - 1, vocab_num_blocks);
  sampled[static_cast<int64_t>(req) * sampled_stride + num_draft] = bonus;
  num_sampled[req] = num_draft + 1;
}

}  // namespace

void triton_gumbel_sample(
    torch::Tensor& local_argmax, int64_t local_argmax_stride,
    torch::Tensor& local_max, int64_t local_max_stride,
    const std::optional<torch::Tensor>& logits_cache,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const std::optional<torch::Tensor>& logits_cache_col,
    torch::Tensor& logits, int64_t logits_stride,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& seeds,
    torch::Tensor& positions, torch::Tensor& temperatures, int64_t vocab_size,
    int64_t block_size, bool is_drafting, bool apply_temperature,
    bool use_fp64, bool per_token_col) {
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
      "triton_gumbel_sample", [&] {
        scalar_t* cache =
            logits_cache ? logits_cache->data_ptr<scalar_t>() : nullptr;
        const int32_t* cols =
            logits_cache_col ? logits_cache_col->data_ptr<int32_t>() : nullptr;
        if (use_fp64) {
          gumbel_sample_kernel<scalar_t, double><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), local_argmax_stride,
              local_max.data_ptr<double>(), local_max_stride, cache,
              logits_cache_stride_0, logits_cache_stride_1, cols,
              logits.data_ptr<scalar_t>(), logits_stride,
              expanded_idx_mapping.data_ptr(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              temperatures.data_ptr<float>(), vocab_size, block_size,
              is_drafting, apply_temperature, per_token_col,
              expanded_idx_mapping.scalar_type() == at::kLong, num_tokens);
        } else {
          gumbel_sample_kernel<scalar_t, float><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), local_argmax_stride,
              local_max.data_ptr<float>(), local_max_stride, cache,
              logits_cache_stride_0, logits_cache_stride_1, cols,
              logits.data_ptr<scalar_t>(), logits_stride,
              expanded_idx_mapping.data_ptr(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              temperatures.data_ptr<float>(), vocab_size, block_size,
              is_drafting, apply_temperature, per_token_col,
              expanded_idx_mapping.scalar_type() == at::kLong, num_tokens);
        }
      });
}

void triton_compute_local_logits_stats(
    torch::Tensor& target_local_argmax, int64_t target_argmax_stride,
    torch::Tensor& target_local_max, int64_t target_max_stride,
    torch::Tensor& target_local_sumexp, int64_t target_sumexp_stride,
    torch::Tensor& draft_local_max, int64_t draft_max_stride,
    torch::Tensor& draft_local_sumexp, int64_t draft_sumexp_stride,
    torch::Tensor& target_logits, int64_t target_logits_stride,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& expanded_idx_mapping,
    torch::Tensor& expanded_local_pos, torch::Tensor& temperatures,
    int64_t vocab_size, int64_t num_speculative_steps,
    int64_t vocab_block_size, bool has_draft_logits) {
  const int num_logits = target_logits.size(0);
  const int num_blocks = (vocab_size + vocab_block_size - 1) / vocab_block_size;
  if (num_logits == 0 || num_blocks == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_logits, num_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      target_logits.scalar_type(), "triton_compute_local_logits_stats", [&] {
        compute_local_logits_stats_kernel<scalar_t>
            <<<grid, kThreads, 0, stream>>>(
                target_local_argmax.data_ptr<int64_t>(),
                target_argmax_stride, target_local_max.data_ptr<float>(),
                target_max_stride, target_local_sumexp.data_ptr<float>(),
                target_sumexp_stride, draft_local_max.data_ptr<float>(),
                draft_max_stride, draft_local_sumexp.data_ptr<float>(),
                draft_sumexp_stride, target_logits.data_ptr<scalar_t>(),
                target_logits_stride,
                draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr,
                draft_stride_0, draft_stride_1,
                expanded_idx_mapping.data_ptr(),
                expanded_local_pos.data_ptr<int32_t>(),
                temperatures.data_ptr<float>(), vocab_size,
                num_speculative_steps, vocab_block_size, has_draft_logits,
                expanded_idx_mapping.scalar_type() == at::kLong, num_logits);
      });
}

void triton_rejection(
    torch::Tensor& sampled, int64_t sampled_stride,
    torch::Tensor& rejected_steps, torch::Tensor& target_rejected_lse,
    torch::Tensor& draft_rejected_lse, torch::Tensor& target_logits,
    int64_t target_logits_stride, torch::Tensor& target_local_argmax,
    int64_t target_argmax_stride, torch::Tensor& target_local_max,
    int64_t target_max_stride, torch::Tensor& target_local_sumexp,
    int64_t target_sumexp_stride, torch::Tensor& draft_sampled,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& draft_local_max,
    int64_t draft_max_stride, torch::Tensor& draft_local_sumexp,
    int64_t draft_sumexp_stride, torch::Tensor& cu_num_logits,
    torch::Tensor& idx_mapping, torch::Tensor& temperatures,
    torch::Tensor& seeds, torch::Tensor& positions,
    const std::optional<torch::Tensor>& synthetic_rates,
    int64_t vocab_num_blocks, bool has_draft_logits, bool synthetic_mode,
    bool use_block_verification) {
  TORCH_CHECK(!use_block_verification,
              "native CUDA rejection does not support block verification");
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      target_logits.scalar_type(), "triton_rejection", [&] {
        rejection_kernel<scalar_t><<<num_reqs, 1, 0, stream>>>(
            sampled.data_ptr<int64_t>(), sampled_stride,
            rejected_steps.data_ptr<int32_t>(),
            target_rejected_lse.data_ptr<float>(),
            draft_rejected_lse.data_ptr<float>(),
            target_logits.data_ptr<scalar_t>(), target_logits_stride,
            target_local_argmax.data_ptr<int64_t>(), target_argmax_stride,
            target_local_max.data_ptr<float>(), target_max_stride,
            target_local_sumexp.data_ptr<float>(), target_sumexp_stride,
            draft_sampled.data_ptr(),
            draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr,
            draft_stride_0, draft_stride_1,
            draft_local_max.data_ptr<float>(), draft_max_stride,
            draft_local_sumexp.data_ptr<float>(), draft_sumexp_stride,
            cu_num_logits.data_ptr<int32_t>(),
            idx_mapping.data_ptr(), temperatures.data_ptr<float>(),
            seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
            synthetic_rates ? synthetic_rates->data_ptr<float>() : nullptr,
            vocab_num_blocks, has_draft_logits, synthetic_mode,
            idx_mapping.scalar_type() == at::kLong,
            draft_sampled.scalar_type() == at::kLong, num_reqs);
      });
}

void triton_resample(
    torch::Tensor& local_argmax, int64_t argmax_stride,
    torch::Tensor& local_max, int64_t max_stride,
    torch::Tensor& target_logits, int64_t target_logits_stride,
    torch::Tensor& target_rejected_lse,
    const std::optional<torch::Tensor>& draft_logits, int64_t draft_stride_0,
    int64_t draft_stride_1, torch::Tensor& draft_rejected_lse,
    torch::Tensor& rejected_step, torch::Tensor& cu_num_logits,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& draft_sampled,
    torch::Tensor& temperatures, torch::Tensor& seeds,
    torch::Tensor& positions,
    const std::optional<torch::Tensor>& cumulative_log_p, int64_t vocab_size,
    int64_t block_size, bool has_draft_logits, bool use_fp64,
    bool use_block_verification) {
  const int num_reqs = rejected_step.numel();
  const int num_blocks = (vocab_size + block_size - 1) / block_size;
  if (num_reqs == 0 || num_blocks == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_reqs, num_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      target_logits.scalar_type(), "triton_resample", [&] {
        const scalar_t* draft =
            draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr;
        const float* cumulative =
            cumulative_log_p ? cumulative_log_p->data_ptr<float>() : nullptr;
        if (use_fp64) {
          resample_kernel<scalar_t, double><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), argmax_stride,
              local_max.data_ptr<double>(), max_stride,
              target_logits.data_ptr<scalar_t>(), target_logits_stride,
              target_rejected_lse.data_ptr<float>(), draft, draft_stride_0,
              draft_stride_1, draft_rejected_lse.data_ptr<float>(),
              rejected_step.data_ptr<int32_t>(),
              cu_num_logits.data_ptr<int32_t>(),
              expanded_idx_mapping.data_ptr(), draft_sampled.data_ptr(),
              temperatures.data_ptr<float>(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              cumulative, vocab_size, block_size, has_draft_logits,
              use_block_verification,
              expanded_idx_mapping.scalar_type() == at::kLong,
              draft_sampled.scalar_type() == at::kLong, num_reqs);
        } else {
          resample_kernel<scalar_t, float><<<grid, kThreads, 0, stream>>>(
              local_argmax.data_ptr<int64_t>(), argmax_stride,
              local_max.data_ptr<float>(), max_stride,
              target_logits.data_ptr<scalar_t>(), target_logits_stride,
              target_rejected_lse.data_ptr<float>(), draft, draft_stride_0,
              draft_stride_1, draft_rejected_lse.data_ptr<float>(),
              rejected_step.data_ptr<int32_t>(),
              cu_num_logits.data_ptr<int32_t>(),
              expanded_idx_mapping.data_ptr(), draft_sampled.data_ptr(),
              temperatures.data_ptr<float>(),
              seeds.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
              cumulative, vocab_size, block_size, has_draft_logits,
              use_block_verification,
              expanded_idx_mapping.scalar_type() == at::kLong,
              draft_sampled.scalar_type() == at::kLong, num_reqs);
        }
      });
}

void triton_insert_resampled(
    torch::Tensor& sampled, int64_t sampled_stride,
    torch::Tensor& num_sampled, torch::Tensor& local_argmax,
    int64_t argmax_stride, torch::Tensor& local_max, int64_t max_stride,
    int64_t resample_num_blocks, torch::Tensor& cu_num_logits,
    torch::Tensor& expanded_idx_mapping, torch::Tensor& temperatures,
    bool use_fp64) {
  const int num_reqs = num_sampled.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(sampled));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (use_fp64) {
    insert_resampled_kernel<double><<<num_reqs, 1, 0, stream>>>(
        sampled.data_ptr<int64_t>(), sampled_stride,
        num_sampled.data_ptr<int32_t>(), local_argmax.data_ptr<int64_t>(),
        argmax_stride, local_max.data_ptr<double>(), max_stride,
        resample_num_blocks, cu_num_logits.data_ptr<int32_t>(),
        expanded_idx_mapping.data_ptr(), temperatures.data_ptr<float>(),
        expanded_idx_mapping.scalar_type() == at::kLong, num_reqs);
  } else {
    insert_resampled_kernel<float><<<num_reqs, 1, 0, stream>>>(
        sampled.data_ptr<int64_t>(), sampled_stride,
        num_sampled.data_ptr<int32_t>(), local_argmax.data_ptr<int64_t>(),
        argmax_stride, local_max.data_ptr<float>(), max_stride,
        resample_num_blocks, cu_num_logits.data_ptr<int32_t>(),
        expanded_idx_mapping.data_ptr(), temperatures.data_ptr<float>(),
        expanded_idx_mapping.scalar_type() == at::kLong, num_reqs);
  }
}

std::tuple<torch::Tensor, torch::Tensor> greedy_rejection_sample(
    torch::Tensor& target_logits, torch::Tensor& draft_sampled,
    torch::Tensor& cu_num_logits, int64_t max_speculative_steps) {
  TORCH_CHECK(target_logits.dim() == 2,
              "target_logits must be a 2D tensor");
  TORCH_CHECK(target_logits.stride(1) == 1,
              "target_logits must be contiguous in the vocabulary dimension");
  TORCH_CHECK(draft_sampled.scalar_type() == at::kInt ||
                  draft_sampled.scalar_type() == at::kLong,
              "draft_sampled must be int32 or int64");
  TORCH_CHECK(cu_num_logits.scalar_type() == at::kInt,
              "cu_num_logits must be int32");
  TORCH_CHECK(max_speculative_steps >= 0,
              "max_speculative_steps must be non-negative");

  const int num_reqs = cu_num_logits.numel() - 1;
  const int num_rows = target_logits.size(0);
  const int vocab_size = target_logits.size(1);
  constexpr int kVocabBlockSize = 8192;
  const int vocab_num_blocks =
      (vocab_size + kVocabBlockSize - 1) / kVocabBlockSize;
  auto sampled = torch::empty(
      {num_reqs, max_speculative_steps + 1},
      target_logits.options().dtype(at::kLong));
  auto num_sampled =
      torch::empty({num_reqs}, target_logits.options().dtype(at::kInt));
  if (num_reqs == 0) {
    return {sampled, num_sampled};
  }
  TORCH_CHECK(num_rows > 0 && vocab_num_blocks > 0,
              "greedy rejection requires non-empty logits");

  auto partial_max = torch::empty(
      {num_rows, vocab_num_blocks},
      target_logits.options().dtype(at::kFloat));
  auto partial_argmax = torch::empty(
      {num_rows, vocab_num_blocks},
      target_logits.options().dtype(at::kLong));
  const at::cuda::OptionalCUDAGuard guard(device_of(target_logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_rows, vocab_num_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      target_logits.scalar_type(), "greedy_rejection_argmax", [&] {
        greedy_argmax_partials_kernel<scalar_t>
            <<<grid, kThreads, 0, stream>>>(
                partial_max.data_ptr<float>(),
                partial_argmax.data_ptr<int64_t>(),
                target_logits.data_ptr<scalar_t>(), target_logits.stride(0),
                vocab_size, kVocabBlockSize, num_rows);
      });
  greedy_rejection_finalize_kernel<<<num_reqs, 1, 0, stream>>>(
      sampled.data_ptr<int64_t>(), sampled.stride(0),
      num_sampled.data_ptr<int32_t>(), partial_max.data_ptr<float>(),
      partial_argmax.data_ptr<int64_t>(), draft_sampled.data_ptr(),
      cu_num_logits.data_ptr<int32_t>(), vocab_num_blocks,
      max_speculative_steps, draft_sampled.scalar_type() == at::kLong,
      num_reqs);
  return {sampled, num_sampled};
}

}  // namespace vllm_iluvatar
