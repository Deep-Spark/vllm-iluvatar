#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shared argument handling; source from run_{mr,cq}_prefill.sh.
mode=${1:---print-command}
default_ssm_dtype=bfloat16
default_mtp_tokens=1
case "$mode" in
  --help|-h)
    cat <<HELP
Usage: run_${platform}_prefill.sh [--print-command|--serve]
Required: MODEL_PATH, DEVICES (distinct GPU IDs, in peer-topology order).
Defaults: TP1/PP$default_pp/DP1, eager, async, MTP$default_mtp_tokens,
  recurrent state=$default_ssm_dtype, BF16 KV/Conv.
Environment overrides:
  TP_SIZE=1 PP_SIZE=$default_pp DP_SIZE=1 ENABLE_EXPERT_PARALLEL=1
  HOST=127.0.0.1 PORT=12715 SERVED_MODEL_NAME=GLM-5-Next
  MAX_MODEL_LEN=131072 MAX_NUM_SEQS=64 GPU_MEMORY_UTILIZATION=0.90
  MAX_NUM_BATCHED_TOKENS=$default_budget BLOCK_SIZE=$default_block
  PREFIX_MATCH_UNIT=$default_prefix_unit VLLM_PP_LAYER_PARTITION=$default_partition
  MAMBA_SSM_CACHE_DTYPE=$default_ssm_dtype MTP_TOKENS=$default_mtp_tokens (0 disables MTP)
  SPECULATIVE_CONFIG=<explicit JSON override, or none to disable>
  BLOCK_SIZE=auto lets vLLM resolve hybrid cache pages for the selected TP/dtype.
  QUANTIZATION=<optional override; auto-detected from MODEL_PATH/config.json>
  PYTHON_BIN=python3
DEVICES must contain TP_SIZE * PP_SIZE * DP_SIZE IDs. For PP_SIZE other than $default_pp,
set VLLM_PP_LAYER_PARTITION explicitly (one positive count per stage, sum 45).
Each stage containing KDA must also contain a target MLA layer; MTP does not
count toward this constraint in the current hybrid cache grouping.
The service runs in the foreground. Stop the vLLM parent normally after testing.
HELP
    if [[ "$platform" == cq ]]; then
      echo 'CQ TP1/DP1 selects PP4 for four chips and PP8 for eight. For two PP4 replicas,'
      echo 'start one process per four-chip group with distinct PORT and VLLM_PORT values.'
      echo 'Each process serves one endpoint. The client or load balancer routes requests.'
    fi
    exit 0 ;;
  --print-command|--serve) shift || true ;;
  *) echo "Use --help, --print-command or --serve" >&2; exit 2 ;;
esac
if (($#)); then echo 'Unexpected arguments' >&2; exit 2; fi
: "${MODEL_PATH:?Set MODEL_PATH to the platform P checkpoint}"
: "${DEVICES:?Set DEVICES to the available device IDs}"
export CUDA_VISIBLE_DEVICES="$DEVICES"
tp=${TP_SIZE:-1}
pp=${PP_SIZE:-$default_pp}
dp=${DP_SIZE:-1}
if [[ "$pp" != "$default_pp" && -z "${VLLM_PP_LAYER_PARTITION:-}" ]]; then
  echo 'Set VLLM_PP_LAYER_PARTITION for this PP_SIZE' >&2
  exit 2
fi
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-$default_partition}"
"${PYTHON_BIN:-python3}" - "$DEVICES" "$VLLM_PP_LAYER_PARTITION" "$tp" "$pp" "$dp" <<'PY'
import sys

try:
    devices = [int(v) for v in sys.argv[1].split(',')]
    layers = [int(v) for v in sys.argv[2].split(',')]
    tp, pp, dp = map(int, sys.argv[3:])
except ValueError:
    sys.exit('Device IDs, stage sizes, and TP/PP/DP sizes must be integers')
if min(tp, pp, dp) < 1 or 64 % tp:
    sys.exit('TP, PP and DP must be positive and TP must divide 64 attention heads')
if len(devices) != tp * pp * dp or len(set(devices)) != len(devices) or min(devices) < 0:
    sys.exit('DEVICES must contain TP_SIZE * PP_SIZE * DP_SIZE distinct, nonnegative IDs')
if len(layers) != pp or min(layers) < 1 or sum(layers) != 45:
    sys.exit('VLLM_PP_LAYER_PARTITION must give one positive size per stage totaling 45')

mla_layers = set(range(3, 45, 4))
start = 0
for stage, count in enumerate(layers):
    end = start + count
    if mla_layers.isdisjoint(range(start, end)):
        sys.exit(f'PP stage {stage} [{start}, {end}) has no target MLA layer for the hybrid cache')
    start = end
PY
export CUDA_HOME="${CUDA_HOME:-/usr/local/corex}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_PLUGINS=iluvatar VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8=1 VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8=1
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
unset CUDA_LAUNCH_BLOCKING UMD_WAITAFTERLAUNCH UMD_ENABLELOGGING
unset UMD_DUMPKERNELINFO UMD_LOGSUBSYS UMD_FILEOUTPUTPATH LD_PRELOAD
mtp_tokens=${MTP_TOKENS:-$default_mtp_tokens}
[[ "$mtp_tokens" =~ ^[0-9]+$ ]] || { echo 'MTP_TOKENS must be nonnegative' >&2; exit 2; }
speculative_config=${SPECULATIVE_CONFIG:-}
if [[ -z "$speculative_config" && "$mtp_tokens" != 0 ]]; then
  speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens,\"use_local_argmax_reduction\":true,\"index_share_for_mtp_iteration\":true}"
fi
quantization="${QUANTIZATION:-$default_quantization}"
if [[ "$platform" == mr && -z "${QUANTIZATION:-}" && -r "$MODEL_PATH/config.json" ]]; then
  quantization=$("${PYTHON_BIN:-python3}" - "$MODEL_PATH/config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    method = json.load(source)['quantization_config']['quant_method']
if method not in ('compressed-tensors', 'ilu_deepseek_v4_w4a8'):
    sys.exit(f'Unsupported MR checkpoint quantization method: {method}')
print(method)
PY
  )
fi
cmd=(vllm serve "$MODEL_PATH"
  --served-model-name "${SERVED_MODEL_NAME:-GLM-5-Next}"
  --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" --data-parallel-size "$dp"
  --distributed-executor-backend mp
  --all2all-backend "${ALL2ALL_BACKEND:-allgather_reducescatter}"
  --dtype bfloat16
  --kv-cache-dtype bfloat16 --mamba-cache-dtype bfloat16
  --mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE:-$default_ssm_dtype}"
  --prefix-match-unit "${PREFIX_MATCH_UNIT:-$default_prefix_unit}"
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-$default_budget}"
  --max-num-seqs "${MAX_NUM_SEQS:-64}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --safetensors-load-strategy prefetch
  --enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details
  --enable-chunked-prefill --long-prefill-token-threshold 0
  --async-scheduling --enforce-eager
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --host "${HOST:-127.0.0.1}" --port "${PORT:-12715}")
if [[ -n "$quantization" ]]; then
  cmd+=(--quantization "$quantization")
fi
block_size=${BLOCK_SIZE:-$default_block}
if [[ "$block_size" != auto ]]; then
  cmd+=(--block-size "$block_size")
fi
case "${ENABLE_EXPERT_PARALLEL:-1}" in
  1) cmd+=(--enable-expert-parallel) ;;
  0) ;;
  *) echo 'ENABLE_EXPERT_PARALLEL must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -n "$speculative_config" && "$speculative_config" != none ]]; then
  cmd+=(--speculative-config "$speculative_config" --enable-mamba-shared-prefix-checkpoint)
fi
case "$mode" in
  --print-command) printf '%q ' "${cmd[@]}"; printf '\n' ;;
  --serve)
    test -r "$MODEL_PATH/config.json"
    printf 'Platform=%s DEVICES=%s PP_LAYERS=%s\n' "$platform" "$DEVICES" "$VLLM_PP_LAYER_PARTITION"
    printf '%q ' "${cmd[@]}"; printf '\n'
    exec "${cmd[@]}" ;;
esac
