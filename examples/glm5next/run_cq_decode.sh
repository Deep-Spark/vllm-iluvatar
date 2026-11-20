#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Set DEVICES in the order required by the machine's peer topology.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: run_cq_decode.sh [--print-command|--serve]
Required: MODEL_PATH, DEVICES. CQ defaults: TP2/DP4, MTP3, FP8 attention and a 278528-token context.
Overrides: HOST, PORT, QUANTIZATION, MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS,
  MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, SPECULATIVE_CONFIG,
  USE_REPLAYSSM, MAMBA_SSM_CACHE_DTYPE, ASYNC_SCHEDULING, DP_RPC_PORT.
  BLOCK_SIZE=auto selects a physical page compatible with the hybrid cache.
The default mode prints the command. --serve runs vLLM in the foreground.
HELP
  exit 0
fi

mode=${1:---print-command}
case "$mode" in
  --print-command|--serve) shift || true ;;
  *) echo "usage: $0 [--print-command|--serve]" >&2; exit 2 ;;
esac
if (($#)); then echo "Unexpected arguments" >&2; exit 2; fi

: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.3-Flash checkpoint}"
: "${DEVICES:?Set DEVICES to the available CQ device IDs}"
export CUDA_VISIBLE_DEVICES="$DEVICES"
export CUDA_HOME="${CUDA_HOME:-/usr/local/corex}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_PLUGINS=iluvatar VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_GRAPH_REGISTER=0
export VLLM_ILU_GLM5_FP8_ATTN="${VLLM_ILU_GLM5_FP8_ATTN:-1}"
export VLLM_SSM_CONV_STATE_LAYOUT=SD
export VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8="${VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8:-1}"
export VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8="${VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8:-1}"
export VLLM_ILU_INDEXER_KV_DTYPE="${VLLM_ILU_INDEXER_KV_DTYPE:-bf16}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
export VLLM_ILU_MOE_AG_SHARED_OVERLAP="${VLLM_ILU_MOE_AG_SHARED_OVERLAP:-0}"
unset VLLM_ILU_GLM5NEXT_MLA_BACKEND VLLM_ILU_GDN_ATTN_BACKEND
unset CUDA_LAUNCH_BLOCKING UMD_WAITAFTERLAUNCH UMD_ENABLELOGGING
unset UMD_DUMPKERNELINFO UMD_LOGSUBSYS UMD_FILEOUTPUTPATH LD_PRELOAD

case "${ASYNC_SCHEDULING:-0}" in
  0) scheduling_flag=--no-async-scheduling ;;
  1) scheduling_flag=--async-scheduling ;;
  *) echo "ASYNC_SCHEDULING must be 0 or 1" >&2; exit 2 ;;
esac
compilation_config='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
extra_args=()
if [[ "${ENFORCE_EAGER:-0}" == 1 ]]; then
  extra_args+=(--enforce-eager)
  compilation_config='{"mode":0,"cudagraph_mode":"NONE"}'
fi
speculative_config='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"index_share_for_mtp_iteration":true}'
cmd=(vllm serve "$MODEL_PATH"
  --served-model-name GLM-5-Next
  --tensor-parallel-size "${TP_SIZE:-2}"
  --data-parallel-size "${DP_SIZE:-4}" --data-parallel-size-local "${DP_SIZE:-4}"
  --pipeline-parallel-size 1 --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --dtype bfloat16
  --kv-cache-dtype bfloat16 --mamba-cache-dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN:-278528}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --max-num-seqs "${MAX_NUM_SEQS:-32}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --safetensors-load-strategy prefetch --enable-prefix-caching "$scheduling_flag"
  --compilation-config "${COMPILATION_CONFIG:-$compilation_config}"
  --speculative-config "${SPECULATIVE_CONFIG:-$speculative_config}"
  --reasoning-parser glm45 --limit-mm-per-prompt '{"image":2,"video":0}'
  --data-parallel-rpc-port "${DP_RPC_PORT:-29751}"
  --host "${HOST:-0.0.0.0}" --port "${PORT:-12347}" "${extra_args[@]}")
if [[ -n "${QUANTIZATION:-}" ]]; then
  cmd+=(--quantization "$QUANTIZATION")
fi
if [[ "${BLOCK_SIZE:-auto}" != auto ]]; then
  cmd+=(--block-size "$BLOCK_SIZE")
fi
case "${USE_REPLAYSSM:-1}" in
  0) ;;
  1) cmd+=(--use-replayssm) ;;
  *) echo "USE_REPLAYSSM must be 0 or 1" >&2; exit 2 ;;
esac
cmd+=(--mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE:-bfloat16}")
case "${MAMBA_CACHE_MODE:-align}" in
  align) ;;
  none) cmd+=(--no-enable-prefix-caching --mamba-cache-mode none) ;;
  *) echo "MAMBA_CACHE_MODE must be align or none" >&2; exit 2 ;;
esac
if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
  cmd+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi
if [[ -n "${PROFILER_CONFIG:-}" ]]; then
  cmd+=(--profiler-config "$PROFILER_CONFIG")
fi
case "$mode" in
  --print-command) printf '%q ' "${cmd[@]}"; printf '\n' ;;
  --serve) exec "${cmd[@]}" ;;
esac
