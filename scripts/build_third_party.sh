#!/usr/bin/env bash
# 开发用脚本：按 submodule 锁定的 commit 编译并安装 ixformer 与 vllm。
# ixformer 的底层依赖由其自带的 scripts/build_dev_env.sh 负责,本脚本不涉及。
# 仅用于开发环境,不参与 vllm-iluvatar 打包与 IXCI 构建。
#
# 用法：
#   git submodule update --init --recursive
#   bash scripts/build_third_party.sh
#
# 目录结构：
#   third_party/ixformer
#   third_party/vllm
#
# 常用环境变量：
#   COREX_ARCH / CUDA_ARCH   目标架构,默认 ivcore11
#   COREX_HOME               corex 工具链路径,默认 /usr/local/corex
#   SW_HOME                  传递给 ixformer 构建脚本
#   MAX_JOBS                 并行任务数,默认 nproc,再按容器内存上限收敛
#   SKIP_IXFORMER=1          跳过 ixformer(含其依赖)
#   SKIP_VLLM=1              跳过 vllm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IXFORMER="${ROOT}/third_party/ixformer"
VLLM="${ROOT}/third_party/vllm"
CUDA_ARCH="${CUDA_ARCH:-${COREX_ARCH:-ivcore11}}"
COREX_HOME="${COREX_HOME:-/usr/local/corex}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# 单个 ivcore CUDA 编译单元峰值约 5GiB,IXCI 构建容器是 128 核 / 32GiB,
# 不按内存上限收敛并行度会在 ninja 起满 job 后被 OOMKill。
mem_max="$(cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | head -1 || true)"
if [[ "${mem_max}" =~ ^[0-9]+$ ]] && (( mem_max < 1 << 62 )); then
  mem_jobs=$(( mem_max / (5 * 1024 ** 3) ))
  if (( mem_jobs < 1 )); then
    mem_jobs=1
  fi
  if (( MAX_JOBS > mem_jobs )); then
    MAX_JOBS="${mem_jobs}"
  fi
fi
echo "==> MAX_JOBS=${MAX_JOBS}"

init_submodules() {
  echo "==> checkout submodule SHAs"
  git -C "${ROOT}" submodule update --init --recursive -- \
    third_party/ixformer third_party/vllm
  if [[ ! -f "${IXFORMER}/scripts/build_dev_env.sh" || ! -f "${VLLM}/build_vllm.sh" ]]; then
    echo "ERROR: submodules missing; run: git submodule update --init --recursive" >&2
    exit 1
  fi
  echo "    ixformer $(git -C "${IXFORMER}" rev-parse --short HEAD)"
  echo "    vllm     $(git -C "${VLLM}" rev-parse --short HEAD)"
}

build_ixformer() {
  if [[ "${SKIP_IXFORMER:-0}" == "1" ]]; then
    echo "==> skip ixformer"
    return
  fi
  echo "==> build + install ixformer (${CUDA_ARCH})"
  CUDA_ARCH="${CUDA_ARCH}" COREX_HOME="${COREX_HOME}" MAX_JOBS="${MAX_JOBS}" \
    bash "${IXFORMER}/scripts/build_dev_env.sh"
}

build_vllm() {
  if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
    echo "==> skip vllm"
    return
  fi
  echo "==> build + install vllm"
  export MAX_JOBS
  bash "${VLLM}/build_vllm.sh"
  bash "${VLLM}/install_vllm.sh"
}

init_submodules
build_ixformer
build_vllm
echo "==> done"
