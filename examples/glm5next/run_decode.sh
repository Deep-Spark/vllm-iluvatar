#!/usr/bin/env bash
set -euo pipefail

# Run from an environment containing the matching IxFormer/plugin builds.
export PATH="${VENV_BIN:-/opt/glm53/.venv/bin}:/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/usr/local/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export VLLM_PLUGINS=iluvatar
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ILU_GDN_ATTN_BACKEND=1
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_ILU_INDEXER_KV_DTYPE=int8
export VLLM_ILU_W4A8_DECODE_IXMM=1
export IXMM_W4A8_GRAPH_SMALL_M_THRESHOLD=192
export VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8=1
export VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8=1
export VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK=1
export VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8=1
export VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE=1
export VLLM_ILU_TOP_P_RADIX=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=36
export VLLM_FORCE_NCCL_COMM=0
export NCCL_P2P_LEVEL=SYS
export NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_BLOCKING_WAIT=1
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GID_INDEX=3
export UCX_TLS=rc_mlx5,ud_mlx5,self,cuda_copy
export UCX_IB_ROCE_REACHABILITY_MODE=local_subnet
export VLLM_ILU_SKIP_CUDAGRAPH_MEMORY_PROFILE=1
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

exec vllm serve "${MODEL:?Set MODEL to the NN V1 checkpoint}" \
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port "${PORT:-12354}" \
  --tensor-parallel-size 2 --data-parallel-size 8 --pipeline-parallel-size 1 \
  --enable-expert-parallel --quantization ilu_deepseek_v4_w4a8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --cudagraph-metrics \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --block-size 256 --max-model-len 131072 --max-num-seqs 64 \
  --gpu-memory-utilization 0.88 --max-num-batched-tokens 2048 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode align \
  --safetensors-load-strategy prefetch --enable-prefix-caching \
  --enable-prompt-tokens-details --no-async-scheduling \
  --limit-mm-per-prompt '{"image":0,"video":0}'
