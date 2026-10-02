// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA replacements for the small Triton metadata kernels exercised by
// the V2 GPU runner and DSpark.  The Python side keeps the community launch
// syntax through tiny launcher objects; all device work is done here.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>

#include "ops.h"

namespace vllm_iluvatar {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int64_t local_cp_slot(
    int64_t position, int64_t block_number, int block_size, int cp_rank,
    int cp_size, int cp_interleave, int64_t pad_id) {
  const int64_t block_offset = position % (block_size * cp_size);
  if (cp_size == 1) {
    return block_number * block_size + block_offset;
  }
  const bool local =
      (block_offset / cp_interleave) % cp_size == cp_rank;
  const int64_t rounds = block_offset / (cp_interleave * cp_size);
  const int64_t remainder = block_offset % cp_interleave;
  const int64_t local_offset = rounds * cp_interleave + remainder;
  return local ? block_number * block_size + local_offset : pad_id;
}

__global__ void prepare_dflash_inputs_kernel(
    int32_t* out_input_ids, int64_t* out_query_positions,
    int32_t* out_query_start_loc, int32_t* out_seq_lens,
    int64_t* out_query_slot_mapping, int64_t* out_context_positions,
    int64_t* out_context_slot_mapping, int64_t* out_sample_indices,
    int64_t* out_sample_pos, int32_t* out_sample_idx_mapping,
    float* out_temperature, int64_t* out_seeds,
    const int64_t* target_positions, const int32_t* target_query_start_loc,
    const int64_t* idx_mapping, const int64_t* last_sampled,
    const int32_t* next_prefill_tokens, const int32_t* num_sampled,
    const int32_t* num_rejected, const float* temperature,
    const int64_t* seeds, const int32_t* block_table,
    int64_t block_table_stride, int parallel_drafting_token_id, int block_size,
    int num_query_per_req, int num_speculative_steps, int max_num_reqs,
    int max_num_tokens, int max_model_len, int cp_rank, int cp_size,
    int cp_interleave, bool sample_from_anchor, int64_t pad_slot_id,
    int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs) {
    return;
  }
  const int req_state = idx_mapping[req];
  const int ctx_start = target_query_start_loc[req];
  const int ctx_end = target_query_start_loc[req + 1];
  const int num_ctx = ctx_end - ctx_start;
  const int valid_ctx_end = ctx_end - num_rejected[req];
  const int num_valid_ctx = valid_ctx_end - ctx_start;
  const int query_base = req * num_query_per_req;
  const int64_t last_valid_pos = target_positions[valid_ctx_end - 1];
  const int32_t bonus_token =
      num_sampled[req] > 0 ? static_cast<int32_t>(last_sampled[req_state])
                           : next_prefill_tokens[req_state];

  for (int j = threadIdx.x; j < num_ctx; j += blockDim.x) {
    int64_t ctx_pos = 0;
    int64_t ctx_slot = pad_slot_id;
    if (j < num_valid_ctx) {
      ctx_pos = target_positions[ctx_start + j];
      int64_t block_num = ctx_pos / (block_size * cp_size);
      block_num = min(block_num, block_table_stride - 1);
      const int64_t block_id =
          block_table[static_cast<int64_t>(req) * block_table_stride + block_num];
      if (block_id != 0) {
        ctx_slot = local_cp_slot(ctx_pos, block_id, block_size, cp_rank,
                                 cp_size, cp_interleave, pad_slot_id);
      }
    }
    out_context_positions[ctx_start + j] = ctx_pos;
    out_context_slot_mapping[ctx_start + j] = ctx_slot;
  }

  for (int q = threadIdx.x; q < num_query_per_req; q += blockDim.x) {
    const int64_t query_pos = last_valid_pos + 1 + q;
    const int query_idx = query_base + q;
    out_input_ids[query_idx] =
        q == 0 ? bonus_token : parallel_drafting_token_id;
    out_query_positions[query_idx] =
        min(query_pos, static_cast<int64_t>(max_model_len - 1));

    int64_t block_num = query_pos / (block_size * cp_size);
    block_num = min(block_num, block_table_stride - 1);
    const int64_t block_id =
        block_table[static_cast<int64_t>(req) * block_table_stride + block_num];
    out_query_slot_mapping[query_idx] =
        block_id == 0
            ? pad_slot_id
            : local_cp_slot(query_pos, block_id, block_size, cp_rank, cp_size,
                            cp_interleave, pad_slot_id);

    const int sample_off = sample_from_anchor ? 0 : 1;
    if (q >= sample_off) {
      const int sample_idx =
          req * num_speculative_steps + (q - sample_off);
      out_sample_indices[sample_idx] = query_idx;
      out_sample_pos[sample_idx] =
          sample_from_anchor ? query_pos + 1 : query_pos;
      out_sample_idx_mapping[sample_idx] = req_state;
    }
  }

  if (threadIdx.x == 0) {
    out_query_start_loc[req] = query_base;
    out_seq_lens[req] = static_cast<int32_t>(
        min(last_valid_pos + 1 + num_query_per_req,
            static_cast<int64_t>(max_model_len)));
    out_temperature[req_state] = temperature[req_state];
    out_seeds[req_state] = seeds[req_state];
  }

  if (req != num_reqs - 1) {
    return;
  }
  const int last_query_end = num_reqs * num_query_per_req;
  for (int i = num_reqs + threadIdx.x; i < max_num_reqs + 1;
       i += blockDim.x) {
    out_query_start_loc[i] = last_query_end;
  }
  for (int i = num_reqs + threadIdx.x; i < max_num_reqs; i += blockDim.x) {
    out_seq_lens[i] = 0;
  }
  const int sample_begin = num_reqs * num_speculative_steps;
  const int sample_end = max_num_reqs * num_speculative_steps;
  for (int i = sample_begin + threadIdx.x; i < sample_end;
       i += blockDim.x) {
    out_sample_indices[i] = 0;
    out_sample_pos[i] = 0;
    out_sample_idx_mapping[i] = -1;
  }
  for (int i = last_query_end + threadIdx.x; i < max_num_tokens;
       i += blockDim.x) {
    out_query_slot_mapping[i] = pad_slot_id;
  }
}

__global__ void zero_kv_blocks_kernel(
    const uint64_t* seg_addrs, const int64_t* seg_block_strides,
    const int64_t* seg_page_sizes, const int64_t* block_ids, int num_segments) {
  const int block_index = blockIdx.x;
  const int segment = blockIdx.y;
  if (segment >= num_segments) {
    return;
  }
  int32_t* base = reinterpret_cast<int32_t*>(seg_addrs[segment]);
  const int64_t offset = block_ids[block_index] * seg_block_strides[segment];
  const int64_t size = seg_page_sizes[segment];
  for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
    base[offset + i] = 0;
  }
}

__global__ void prepare_uniform_decode_kernel(
    const int32_t* seq_lens, int32_t* decode_seq_lens,
    const int32_t* block_table, int64_t block_table_stride,
    int32_t* expanded_block_table, int64_t expanded_stride,
    int32_t* decode_lens, int max_decode_len, int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int req = token / max_decode_len;
  const int local = token % max_decode_len;
  if (threadIdx.x == 0) {
    decode_seq_lens[token] = seq_lens[req] - max_decode_len + local + 1;
    decode_lens[token] = 1;
  }
  for (int64_t col = threadIdx.x; col < expanded_stride; col += blockDim.x) {
    expanded_block_table[static_cast<int64_t>(token) * expanded_stride + col] =
        block_table[static_cast<int64_t>(req) * block_table_stride + col];
  }
}

__global__ void token_request_mapping_kernel(
    const int32_t* query_start_loc, int32_t* output, int num_reqs,
    int num_mapped, int num_tokens) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= num_tokens) {
    return;
  }
  int lo = 0;
  int hi = num_reqs;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (query_start_loc[mid + 1] <= token) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int req = min(lo, num_reqs - 1);
  output[token] = token < num_mapped ? req : 0;
}

__global__ void gather_block_tables_kernel(
    const int64_t* batch_to_req, const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs, const int64_t* strides,
    const int32_t* num_blocks, int64_t num_blocks_stride, int num_reqs,
    int num_reqs_padded) {
  const int group = blockIdx.x;
  const int batch = blockIdx.y;
  if (batch >= num_reqs_padded) {
    return;
  }
  const int64_t stride = strides[group];
  const int32_t* src = reinterpret_cast<const int32_t*>(src_ptrs[group]);
  int32_t* dst = reinterpret_cast<int32_t*>(dst_ptrs[group]);
  int count = static_cast<int>(stride);
  int req = 0;
  if (batch < num_reqs) {
    req = batch_to_req[batch];
    count = num_blocks[static_cast<int64_t>(group) * num_blocks_stride + req];
  }
  for (int col = threadIdx.x; col < count; col += blockDim.x) {
    dst[static_cast<int64_t>(batch) * stride + col] =
        batch < num_reqs ? src[static_cast<int64_t>(req) * stride + col] : 0;
  }
}

__global__ void compute_slot_mappings_kernel(
    int max_num_tokens, const int64_t* idx_mapping,
    const int32_t* query_start_loc, const int64_t* positions,
    const uint64_t* block_table_ptrs, const int64_t* block_table_strides,
    const int32_t* block_sizes, const int32_t* kernel_block_sizes,
    const bool* mapping_enabled, int64_t* slot_mappings,
    int64_t slot_mappings_stride, int cp_rank, int cp_size, int cp_interleave,
    int64_t pad_id, int num_reqs) {
  const int group = blockIdx.x;
  const int batch = blockIdx.y;
  int64_t* output =
      slot_mappings + static_cast<int64_t>(group) * slot_mappings_stride;
  if (batch == num_reqs) {
    const int actual_tokens = query_start_loc[num_reqs];
    for (int i = actual_tokens + threadIdx.x; i < max_num_tokens;
         i += blockDim.x) {
      output[i] = pad_id;
    }
    return;
  }
  if (batch > num_reqs) {
    return;
  }
  const int req_state = idx_mapping[batch];
  const int start = query_start_loc[batch];
  const int end = query_start_loc[batch + 1];
  const int32_t* table =
      reinterpret_cast<const int32_t*>(block_table_ptrs[group]);
  const int64_t table_stride = block_table_strides[group];
  const int kv_block_size = block_sizes[group];
  const int kernel_block_size = kernel_block_sizes[group];
  const bool enabled = mapping_enabled[group];
  for (int token = start + threadIdx.x; token < end; token += blockDim.x) {
    const int64_t pos = positions[token];
    int64_t local_pos = pos;
    bool local = true;
    if (cp_size != 1) {
      const int64_t virtual_size = kv_block_size * cp_size;
      const int64_t virtual_index = pos / virtual_size;
      const int64_t virtual_offset = pos % virtual_size;
      local = (virtual_offset / cp_interleave) % cp_size == cp_rank;
      const int64_t rounds = virtual_offset / (cp_interleave * cp_size);
      const int64_t remainder = virtual_offset % cp_interleave;
      local_pos =
          virtual_index * kv_block_size + rounds * cp_interleave + remainder;
    }
    int64_t slot = pad_id;
    if (enabled && req_state >= 0 && local) {
      const int64_t block_index = local_pos / kernel_block_size;
      const int64_t block_offset = local_pos % kernel_block_size;
      const int64_t block_number =
          table[static_cast<int64_t>(req_state) * table_stride + block_index];
      slot = block_number * kernel_block_size + block_offset;
    }
    output[token] = slot;
  }
}

template <typename scalar_t>
__global__ void apply_write_kernel(
    scalar_t* output, const uint64_t* output_ptrs, int64_t output_stride,
    const int64_t* output_strides, const int32_t* write_indices,
    const int32_t* write_starts, const scalar_t* contents,
    const int32_t* cu_lens, const int32_t* group_ids, bool multi_group,
    int num_writes) {
  const int write = blockIdx.x;
  if (write >= num_writes) {
    return;
  }
  scalar_t* base = output;
  int64_t stride = output_stride;
  if (multi_group) {
    const int group = group_ids[write];
    base = reinterpret_cast<scalar_t*>(output_ptrs[group]);
    stride = output_strides[group];
  }
  const int begin = write == 0 ? 0 : cu_lens[write - 1];
  const int end = cu_lens[write];
  scalar_t* row =
      base + static_cast<int64_t>(write_indices[write]) * stride +
      write_starts[write];
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    row[i - begin] = contents[i];
  }
}

__global__ void prepare_prefill_inputs_kernel(
    int32_t* input_ids, int32_t* next_prefill_tokens,
    int64_t next_prefill_stride, int num_lookahead,
    const int64_t* idx_mapping, const int32_t* query_start_loc,
    const int32_t* all_token_ids, int64_t all_token_ids_stride,
    const int32_t* prefill_lens, const int32_t* num_computed_tokens,
    int num_reqs) {
  const int batch = blockIdx.x;
  if (batch >= num_reqs) {
    return;
  }
  const int state = idx_mapping[batch];
  const int prefill_len = prefill_lens[state];
  const int computed = num_computed_tokens[state];
  if (computed >= prefill_len) {
    return;
  }
  const int start = query_start_loc[batch];
  const int query_len = query_start_loc[batch + 1] - start;
  const int32_t* request =
      all_token_ids + static_cast<int64_t>(state) * all_token_ids_stride;
  for (int i = threadIdx.x; i < query_len; i += blockDim.x) {
    input_ids[start + i] = request[computed + i];
  }
  for (int i = threadIdx.x; i < num_lookahead; i += blockDim.x) {
    const int pos = computed + query_len + i;
    next_prefill_tokens[static_cast<int64_t>(i) * next_prefill_stride + state] =
        pos < prefill_len ? request[pos] : 0;
  }
}

__global__ void prepare_pos_seq_lens_kernel(
    int64_t* positions, int32_t* seq_lens, const int64_t* idx_mapping,
    const int32_t* query_start_loc, const int32_t* num_computed_tokens,
    int num_reqs, int max_num_reqs) {
  const int req = blockIdx.x;
  if (req == num_reqs) {
    for (int i = num_reqs + threadIdx.x; i < max_num_reqs;
         i += blockDim.x) {
      seq_lens[i] = 0;
    }
    return;
  }
  if (req > num_reqs) {
    return;
  }
  const int computed = num_computed_tokens[idx_mapping[req]];
  const int start = query_start_loc[req];
  const int end = query_start_loc[req + 1];
  if (threadIdx.x == 0) {
    seq_lens[req] = computed + end - start;
  }
  for (int token = start + threadIdx.x; token < end; token += blockDim.x) {
    positions[token] = computed + token - start;
  }
}

__global__ void combine_sampled_and_draft_tokens_kernel(
    int32_t* input_ids, const int64_t* idx_mapping,
    const int64_t* last_sampled, const int32_t* query_start_loc,
    const int32_t* seq_lens, const int32_t* prefill_lens,
    const int64_t* draft_tokens, int64_t draft_stride,
    const int32_t* cu_num_logits, int64_t* logits_indices,
    int num_new_sampled_tokens, int num_reqs) {
  const int batch = blockIdx.x;
  if (batch >= num_reqs) {
    return;
  }
  const int state = idx_mapping[batch];
  const int logits_begin = cu_num_logits[batch];
  const int logits_end_cu = cu_num_logits[batch + 1];
  const int num_logits = logits_end_cu - logits_begin;
  const int num_draft = num_logits - num_new_sampled_tokens;
  const int query_end = query_start_loc[batch + 1];
  const int logits_start = query_end - num_logits;
  for (int i = threadIdx.x; i < num_logits; i += blockDim.x) {
    logits_indices[logits_begin + i] = logits_start + i;
  }
  if (seq_lens[batch] <= prefill_lens[state]) {
    return;
  }
  const int first_logit_seq_pos = seq_lens[batch] - num_logits;
  if (threadIdx.x == 0 && num_new_sampled_tokens > 0 &&
      first_logit_seq_pos >= prefill_lens[state]) {
    input_ids[logits_start] = static_cast<int32_t>(last_sampled[state]);
  }
  for (int i = threadIdx.x; i < num_draft; i += blockDim.x) {
    input_ids[query_end - num_draft + i] = static_cast<int32_t>(
        draft_tokens[static_cast<int64_t>(state) * draft_stride + i]);
  }
}

__global__ void get_num_sampled_and_rejected_kernel(
    int32_t* num_sampled, int32_t* num_rejected, const int32_t* seq_lens,
    const int32_t* cu_num_logits, const int64_t* idx_mapping,
    const int32_t* prefill_lens, int num_reqs) {
  const int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= num_reqs) {
    return;
  }
  const bool chunked = seq_lens[batch] < prefill_lens[idx_mapping[batch]];
  const int sampled = chunked ? 0 : num_sampled[batch];
  num_sampled[batch] = sampled;
  const int num_logits = cu_num_logits[batch + 1] - cu_num_logits[batch];
  num_rejected[batch] = chunked ? 0 : num_logits - sampled;
}

__global__ void post_update_kernel(
    const int64_t* idx_mapping, int32_t* num_computed_tokens,
    int64_t* last_sampled, int32_t* output_bin_counts,
    int64_t output_bin_stride, const int64_t* sampled_tokens,
    int64_t sampled_stride, const int32_t* num_sampled,
    const int32_t* num_rejected, const int32_t* query_start_loc,
    int32_t* all_token_ids, int64_t all_token_ids_stride, int32_t* total_len,
    int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs || threadIdx.x != 0) {
    return;
  }
  const int state = idx_mapping[req];
  if (state < 0) {
    return;
  }
  const int old_total = total_len[state];
  const int sampled_count = num_sampled[req];
  if (sampled_count > 0) {
    last_sampled[state] =
        sampled_tokens[static_cast<int64_t>(req) * sampled_stride +
                       sampled_count - 1];
    total_len[state] = old_total + sampled_count;
  }
  for (int i = 0; i < sampled_count; ++i) {
    const int token = static_cast<int>(
        sampled_tokens[static_cast<int64_t>(req) * sampled_stride + i]);
    all_token_ids[static_cast<int64_t>(state) * all_token_ids_stride +
                  old_total + i] = token;
    if (output_bin_counts != nullptr) {
      ++output_bin_counts[static_cast<int64_t>(state) * output_bin_stride +
                          token];
    }
  }
  int query_len = 0;
  if (query_start_loc != nullptr) {
    query_len = query_start_loc[req + 1] - query_start_loc[req];
  }
  const int delta = query_len - num_rejected[req];
  if (delta != 0) {
    num_computed_tokens[state] += delta;
  }
}

__global__ void expand_idx_mapping_kernel(
    const int64_t* idx_mapping, int64_t* expanded_idx_mapping,
    int32_t* expanded_local_pos, const int32_t* cu_num_logits,
    int num_reqs) {
  const int req = blockIdx.x;
  if (req >= num_reqs) {
    return;
  }
  const int begin = cu_num_logits[req];
  const int end = cu_num_logits[req + 1];
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    expanded_idx_mapping[i] = idx_mapping[req];
    expanded_local_pos[i] = i - begin;
  }
}

}  // namespace

void triton_prepare_dflash_inputs(
    torch::Tensor& out_input_ids, torch::Tensor& out_query_positions,
    torch::Tensor& out_query_start_loc, torch::Tensor& out_seq_lens,
    torch::Tensor& out_query_slot_mapping,
    torch::Tensor& out_context_positions,
    torch::Tensor& out_context_slot_mapping,
    torch::Tensor& out_sample_indices, torch::Tensor& out_sample_pos,
    torch::Tensor& out_sample_idx_mapping, torch::Tensor& out_temperature,
    torch::Tensor& out_seeds, torch::Tensor& target_positions,
    torch::Tensor& target_query_start_loc, torch::Tensor& idx_mapping,
    torch::Tensor& last_sampled, torch::Tensor& next_prefill_tokens,
    torch::Tensor& num_sampled, torch::Tensor& num_rejected,
    torch::Tensor& temperature, torch::Tensor& seeds,
    torch::Tensor& block_table, int64_t block_table_stride,
    int64_t parallel_drafting_token_id, int64_t block_size,
    int64_t num_query_per_req, int64_t num_speculative_steps,
    int64_t max_num_reqs, int64_t max_num_tokens, int64_t max_model_len,
    int64_t cp_rank, bool sample_from_anchor, int64_t pad_slot_id,
    int64_t cp_size, int64_t cp_interleave) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(out_input_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prepare_dflash_inputs_kernel<<<num_reqs, kThreads, 0, stream>>>(
      out_input_ids.data_ptr<int32_t>(),
      out_query_positions.data_ptr<int64_t>(),
      out_query_start_loc.data_ptr<int32_t>(), out_seq_lens.data_ptr<int32_t>(),
      out_query_slot_mapping.data_ptr<int64_t>(),
      out_context_positions.data_ptr<int64_t>(),
      out_context_slot_mapping.data_ptr<int64_t>(),
      out_sample_indices.data_ptr<int64_t>(), out_sample_pos.data_ptr<int64_t>(),
      out_sample_idx_mapping.data_ptr<int32_t>(),
      out_temperature.data_ptr<float>(), out_seeds.data_ptr<int64_t>(),
      target_positions.data_ptr<int64_t>(),
      target_query_start_loc.data_ptr<int32_t>(),
      idx_mapping.data_ptr<int64_t>(), last_sampled.data_ptr<int64_t>(),
      next_prefill_tokens.data_ptr<int32_t>(), num_sampled.data_ptr<int32_t>(),
      num_rejected.data_ptr<int32_t>(), temperature.data_ptr<float>(),
      seeds.data_ptr<int64_t>(), block_table.data_ptr<int32_t>(),
      block_table_stride, parallel_drafting_token_id, block_size,
      num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
      max_model_len, cp_rank, cp_size, cp_interleave, sample_from_anchor,
      pad_slot_id, num_reqs);
}

void triton_zero_kv_blocks(torch::Tensor& seg_addrs,
                           torch::Tensor& seg_block_strides,
                           torch::Tensor& seg_page_sizes,
                           torch::Tensor& block_ids) {
  if (block_ids.numel() == 0 || seg_addrs.numel() == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(seg_addrs));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(block_ids.numel(), seg_addrs.numel());
  zero_kv_blocks_kernel<<<grid, kThreads, 0, stream>>>(
      seg_addrs.data_ptr<uint64_t>(), seg_block_strides.data_ptr<int64_t>(),
      seg_page_sizes.data_ptr<int64_t>(), block_ids.data_ptr<int64_t>(),
      seg_addrs.numel());
}

void triton_prepare_uniform_decode(
    torch::Tensor& seq_lens, torch::Tensor& decode_seq_lens,
    torch::Tensor& block_table, int64_t block_table_stride,
    torch::Tensor& expanded_block_table, int64_t expanded_stride,
    torch::Tensor& decode_lens, int64_t max_decode_len,
    int64_t num_tokens) {
  if (num_tokens == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(seq_lens));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prepare_uniform_decode_kernel<<<num_tokens, kThreads, 0, stream>>>(
      seq_lens.data_ptr<int32_t>(), decode_seq_lens.data_ptr<int32_t>(),
      block_table.data_ptr<int32_t>(), block_table_stride,
      expanded_block_table.data_ptr<int32_t>(), expanded_stride,
      decode_lens.data_ptr<int32_t>(), max_decode_len, num_tokens);
}

void triton_token_request_mapping(torch::Tensor& query_start_loc,
                                  torch::Tensor& output, int64_t num_reqs,
                                  int64_t num_mapped, int64_t num_tokens) {
  if (num_tokens == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (num_tokens + kThreads - 1) / kThreads;
  token_request_mapping_kernel<<<blocks, kThreads, 0, stream>>>(
      query_start_loc.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
      num_reqs, num_mapped, num_tokens);
}

void triton_gather_block_tables(
    torch::Tensor& batch_to_req, torch::Tensor& src_ptrs,
    torch::Tensor& dst_ptrs, torch::Tensor& strides, torch::Tensor& num_blocks,
    int64_t num_blocks_stride, int64_t num_reqs,
    int64_t num_reqs_padded) {
  if (num_reqs_padded == 0 || src_ptrs.numel() == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(batch_to_req));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(src_ptrs.numel(), num_reqs_padded);
  gather_block_tables_kernel<<<grid, kThreads, 0, stream>>>(
      batch_to_req.data_ptr<int64_t>(), src_ptrs.data_ptr<uint64_t>(),
      dst_ptrs.data_ptr<uint64_t>(), strides.data_ptr<int64_t>(),
      num_blocks.data_ptr<int32_t>(), num_blocks_stride, num_reqs,
      num_reqs_padded);
}

void triton_compute_slot_mappings(
    int64_t max_num_tokens, torch::Tensor& idx_mapping,
    torch::Tensor& query_start_loc, torch::Tensor& positions,
    torch::Tensor& block_table_ptrs, torch::Tensor& block_table_strides,
    torch::Tensor& block_sizes, torch::Tensor& kernel_block_sizes,
    torch::Tensor& mapping_enabled, torch::Tensor& slot_mappings,
    int64_t slot_mappings_stride, int64_t cp_rank, int64_t cp_size,
    int64_t cp_interleave, int64_t pad_id) {
  const int groups = block_table_ptrs.numel();
  const int num_reqs = idx_mapping.numel();
  if (groups == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(slot_mappings));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(groups, num_reqs + 1);
  compute_slot_mappings_kernel<<<grid, kThreads, 0, stream>>>(
      max_num_tokens, idx_mapping.data_ptr<int64_t>(),
      query_start_loc.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
      block_table_ptrs.data_ptr<uint64_t>(),
      block_table_strides.data_ptr<int64_t>(), block_sizes.data_ptr<int32_t>(),
      kernel_block_sizes.data_ptr<int32_t>(),
      mapping_enabled.data_ptr<bool>(), slot_mappings.data_ptr<int64_t>(),
      slot_mappings_stride, cp_rank, cp_size, cp_interleave, pad_id,
      num_reqs);
}

void triton_apply_write(
    torch::Tensor& output, int64_t output_stride,
    torch::Tensor& write_indices, torch::Tensor& write_starts,
    torch::Tensor& contents, torch::Tensor& cu_lens,
    const std::optional<torch::Tensor>& group_ids,
    const std::optional<torch::Tensor>& output_strides, bool multi_group) {
  const int num_writes = write_indices.numel();
  if (num_writes == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(contents));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(contents.scalar_type() == at::kInt ||
                  contents.scalar_type() == at::kFloat,
              "triton_apply_write supports int32 and float32");
  if (contents.scalar_type() == at::kInt) {
    apply_write_kernel<int32_t><<<num_writes, kThreads, 0, stream>>>(
        multi_group ? nullptr : output.data_ptr<int32_t>(),
        multi_group ? output.data_ptr<uint64_t>() : nullptr, output_stride,
        multi_group ? output_strides->data_ptr<int64_t>() : nullptr,
        write_indices.data_ptr<int32_t>(), write_starts.data_ptr<int32_t>(),
        contents.data_ptr<int32_t>(), cu_lens.data_ptr<int32_t>(),
        multi_group ? group_ids->data_ptr<int32_t>() : nullptr, multi_group,
        num_writes);
  } else {
    TORCH_CHECK(!multi_group, "multi-group float writes are not supported");
    apply_write_kernel<float><<<num_writes, kThreads, 0, stream>>>(
        output.data_ptr<float>(), nullptr, output_stride, nullptr,
        write_indices.data_ptr<int32_t>(), write_starts.data_ptr<int32_t>(),
        contents.data_ptr<float>(), cu_lens.data_ptr<int32_t>(), nullptr, false,
        num_writes);
  }
}

void triton_prepare_prefill_inputs(
    torch::Tensor& input_ids, torch::Tensor& next_prefill_tokens,
    int64_t next_prefill_stride, int64_t num_lookahead,
    torch::Tensor& idx_mapping, torch::Tensor& query_start_loc,
    torch::Tensor& all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor& prefill_lens, torch::Tensor& num_computed_tokens) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(input_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prepare_prefill_inputs_kernel<<<num_reqs, kThreads, 0, stream>>>(
      input_ids.data_ptr<int32_t>(), next_prefill_tokens.data_ptr<int32_t>(),
      next_prefill_stride, num_lookahead, idx_mapping.data_ptr<int64_t>(),
      query_start_loc.data_ptr<int32_t>(), all_token_ids.data_ptr<int32_t>(),
      all_token_ids_stride, prefill_lens.data_ptr<int32_t>(),
      num_computed_tokens.data_ptr<int32_t>(), num_reqs);
}

void triton_prepare_pos_seq_lens(
    torch::Tensor& positions, torch::Tensor& seq_lens,
    torch::Tensor& idx_mapping, torch::Tensor& query_start_loc,
    torch::Tensor& num_computed_tokens, int64_t max_num_reqs) {
  const int num_reqs = idx_mapping.numel();
  const at::cuda::OptionalCUDAGuard guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prepare_pos_seq_lens_kernel<<<num_reqs + 1, kThreads, 0, stream>>>(
      positions.data_ptr<int64_t>(), seq_lens.data_ptr<int32_t>(),
      idx_mapping.data_ptr<int64_t>(), query_start_loc.data_ptr<int32_t>(),
      num_computed_tokens.data_ptr<int32_t>(), num_reqs, max_num_reqs);
}

void triton_combine_sampled_and_draft_tokens(
    torch::Tensor& input_ids, torch::Tensor& idx_mapping,
    torch::Tensor& last_sampled, torch::Tensor& query_start_loc,
    torch::Tensor& seq_lens, torch::Tensor& prefill_lens,
    torch::Tensor& draft_tokens, int64_t draft_stride,
    torch::Tensor& cu_num_logits, torch::Tensor& logits_indices,
    int64_t num_new_sampled_tokens) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(input_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  combine_sampled_and_draft_tokens_kernel<<<num_reqs, kThreads, 0, stream>>>(
      input_ids.data_ptr<int32_t>(), idx_mapping.data_ptr<int64_t>(),
      last_sampled.data_ptr<int64_t>(), query_start_loc.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), prefill_lens.data_ptr<int32_t>(),
      draft_tokens.data_ptr<int64_t>(), draft_stride,
      cu_num_logits.data_ptr<int32_t>(), logits_indices.data_ptr<int64_t>(),
      num_new_sampled_tokens, num_reqs);
}

void triton_get_num_sampled_and_rejected(
    torch::Tensor& num_sampled, torch::Tensor& num_rejected,
    torch::Tensor& seq_lens, torch::Tensor& cu_num_logits,
    torch::Tensor& idx_mapping, torch::Tensor& prefill_lens) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(num_sampled));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (num_reqs + kThreads - 1) / kThreads;
  get_num_sampled_and_rejected_kernel<<<blocks, kThreads, 0, stream>>>(
      num_sampled.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), cu_num_logits.data_ptr<int32_t>(),
      idx_mapping.data_ptr<int64_t>(), prefill_lens.data_ptr<int32_t>(),
      num_reqs);
}

void triton_post_update(
    torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    torch::Tensor& last_sampled,
    const std::optional<torch::Tensor>& output_bin_counts,
    int64_t output_bin_stride, torch::Tensor& sampled_tokens,
    int64_t sampled_stride, torch::Tensor& num_sampled,
    torch::Tensor& num_rejected,
    const std::optional<torch::Tensor>& query_start_loc,
    torch::Tensor& all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor& total_len) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  post_update_kernel<<<num_reqs, 1, 0, stream>>>(
      idx_mapping.data_ptr<int64_t>(),
      num_computed_tokens.data_ptr<int32_t>(),
      last_sampled.data_ptr<int64_t>(),
      output_bin_counts ? output_bin_counts->data_ptr<int32_t>() : nullptr,
      output_bin_stride, sampled_tokens.data_ptr<int64_t>(), sampled_stride,
      num_sampled.data_ptr<int32_t>(), num_rejected.data_ptr<int32_t>(),
      query_start_loc ? query_start_loc->data_ptr<int32_t>() : nullptr,
      all_token_ids.data_ptr<int32_t>(), all_token_ids_stride,
      total_len.data_ptr<int32_t>(), num_reqs);
}

void triton_expand_idx_mapping(torch::Tensor& idx_mapping,
                               torch::Tensor& expanded_idx_mapping,
                               torch::Tensor& expanded_local_pos,
                               torch::Tensor& cu_num_logits) {
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(idx_mapping));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  expand_idx_mapping_kernel<<<num_reqs, kThreads, 0, stream>>>(
      idx_mapping.data_ptr<int64_t>(),
      expanded_idx_mapping.data_ptr<int64_t>(),
      expanded_local_pos.data_ptr<int32_t>(),
      cu_num_logits.data_ptr<int32_t>(), num_reqs);
}

}  // namespace vllm_iluvatar
