#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
platform=mr
default_pp=8
default_partition=7,6,6,6,5,6,5,4
default_budget=2048
default_block=auto
default_prefix_unit=128
default_quantization=compressed-tensors
export VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_ILU_INDEXER_KV_DTYPE=int8
export VLLM_ILU_GDN_ATTN_BACKEND=1 VLLM_FORCE_NCCL_COMM=1
export NCCL_P2P_LEVEL=SYS NCCL_GRAPH_REGISTER=0
source "$(dirname "${BASH_SOURCE[0]}")/_prefill.sh" "$@"
