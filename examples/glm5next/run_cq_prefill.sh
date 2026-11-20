#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
platform=cq
default_pp=8
default_partition=7,6,6,6,6,5,5,4
default_budget=2304

# One PP4 replica per four chips for throughput; eight chips run a single PP8.
if [[ ${TP_SIZE:-1} == 1 && ${DP_SIZE:-1} == 1 ]]; then
  IFS=, read -r -a cq_devices <<< "${DEVICES:-}"
  if [[ ${PP_SIZE:-${#cq_devices[@]}} == 4 ]]; then
    default_pp=4
    default_partition=13,12,11,9
  fi
fi
default_block=128
default_prefix_unit=128
default_quantization=""
export VLLM_ILU_GLM5_FP8_ATTN="${VLLM_ILU_GLM5_FP8_ATTN:-1}"
export VLLM_SSM_CONV_STATE_LAYOUT=SD VLLM_ILU_INDEXER_KV_DTYPE=bf16
export NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_GRAPH_REGISTER=0
unset VLLM_ILU_GLM5NEXT_MLA_BACKEND VLLM_ILU_GDN_ATTN_BACKEND
source "$(dirname "${BASH_SOURCE[0]}")/_prefill.sh" "$@"
