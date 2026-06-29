#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_PATH="${PYTHON_PATH:-$(command -v python3)}"
MAX_JOBS="${VLLM_ILUVATAR_MAX_JOBS:-${MAX_JOBS:-$(nproc --all)}}"

export VLLM_ILUVATAR_MAX_JOBS="${MAX_JOBS}"
export MAX_JOBS="${MAX_JOBS}"
export VLLM_ILUVATAR_CUDA_ARCHITECTURES="${VLLM_ILUVATAR_CUDA_ARCHITECTURES:-ivcore11}"
export VLLM_ILUVATAR_ENABLE_CUINFER="${VLLM_ILUVATAR_ENABLE_CUINFER:-1}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export VLLM_ILUVATAR_BUILD_OUTPUT="${VLLM_ILUVATAR_BUILD_OUTPUT:-stdout}"
export NINJA_STATUS="${NINJA_STATUS:-[running=%r started=%s finished=%f total=%t elapsed=%e] }"

echo "INFO: python=${PYTHON_PATH}"
echo "INFO: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
echo "INFO: VLLM_ILUVATAR_CUDA_ARCHITECTURES=${VLLM_ILUVATAR_CUDA_ARCHITECTURES}"
echo "INFO: VLLM_ILUVATAR_ENABLE_CUINFER=${VLLM_ILUVATAR_ENABLE_CUINFER}"
echo "INFO: MAX_JOBS=${MAX_JOBS}"

ensure_build_deps() {
  if [[ "${VLLM_ILUVATAR_SKIP_BUILD_DEPS:-0}" == "1" ]]; then
    return
  fi

  if "${PYTHON_PATH}" - <<'PY' >/dev/null 2>&1
import setuptools  # noqa: F401
import wheel  # noqa: F401
from packaging.version import Version
assert Version(setuptools.__version__) >= Version("77.0.3")
PY
  then
    return
  fi

  echo "INFO: Installing Python build dependencies: setuptools>=77.0.3 wheel"
  "${PYTHON_PATH}" -m pip install -U 'setuptools>=77.0.3,<81' wheel
}

ensure_build_deps

mkdir -p build_pip

echo "INFO: Building wheel into build_pip/"
"${PYTHON_PATH}" setup.py bdist_wheel -d build_pip 2>&1 | tee -a compile.log
status=${PIPESTATUS[0]}
if [[ "${status}" != "0" ]]; then
  exit "${status}"
fi

exit 0