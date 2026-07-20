#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_PATH="${PYTHON_PATH:-$(command -v python3)}"
TARGET_DIR="${TARGET_DIR:-}"
PKG_DIR="${PKG_DIR:-build_pip}"
PKG_NAME="vllm_iluvatar"

ensure_build_deps() {
  if [[ "${VLLM_ILUVATAR_SKIP_BUILD_DEPS:-0}" == "1" ]]; then
    return
  fi

  if "${PYTHON_PATH}" - <<'PY' >/dev/null 2>&1
import setuptools  # noqa: F401
import wheel  # noqa: F401
PY
  then
    return
  fi

  echo "INFO: Installing Python build dependencies: setuptools wheel"
  "${PYTHON_PATH}" -m pip install -U setuptools wheel
}


# 这里是为了日常的打包和CI测试，在安装vllm的时候不会安装任何依赖，统一在安装plugin的时候进行安装

PIP_INSTALL_FLAGS=()

install_runtime_deps() {
  if [[ "${VLLM_ILUVATAR_INSTALL_RUNTIME_DEPS:-0}" != "1" ]]; then
    echo "INFO: VLLM_ILUVATAR_INSTALL_RUNTIME_DEPS!=1, skipping runtime deps install."
    return
  fi

  echo "INFO: Installing runtime deps (numpy, --no-deps quartet, requirements.txt)"
  "${PYTHON_PATH}" -m pip install --upgrade pip

  # ``--only-binary=:all:`` forbids sdist fallback so pip fails loudly (with a
  # clear "no matching distribution" message pointing at the offending package)
  # instead of trying to compile a Rust / C extension in a container that has
  # no toolchain. On by default in CI / packaging (where the toolchain gap is
  # the norm); developer hosts that legitimately need to build a sdist can opt
  # out with ``VLLM_ILUVATAR_PIP_BINARY_ONLY=0``.
  BINARY_ONLY_FLAG=()
  if [[ "${VLLM_ILUVATAR_PIP_BINARY_ONLY:-1}" == "1" ]]; then
    BINARY_ONLY_FLAG=(--only-binary=:all:)
  fi

  # numpy must be pinned to a NumPy 1.x release to match CoreX torch's C ABI.
  # ``.ixci/build.json`` also pins it before the build step; this line is the
  # fallback for hosts that run install_vllm_iluvatar.sh standalone. Idempotent
  # (skips if numpy 1.26.x is already installed).
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" 'numpy~=1.26.0'

  # T
  # gets its own explicit pip line here.
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" --no-deps 'opencv-python-headless==4.11.0.86'
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" --no-deps 'compressed-tensors==0.15.0.1'
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" --no-deps 'mistral_common==1.11.5'
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" --no-deps 'xgrammar>=0.2.1,<1.0.0'
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" --no-deps 'llguidance>=1.7.0,<1.8.0'

  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" "${BINARY_ONLY_FLAG[@]}" -r "${ROOT_DIR}/requirements.txt"
}

if [[ "${EDITABLE:-0}" == "1" ]]; then
  # Editable / developer install: assume the CoreX SDK Python bundle already
  # provides every runtime dep. install_runtime_deps is off by default anyway
  # (see the guard inside the function); this branch just skips it explicitly.
  ensure_build_deps
  "${PYTHON_PATH}" -m pip install --no-build-isolation -e .
  exit 0
fi

if [[ ! -d "${PKG_DIR}" ]] || ! ls "${PKG_DIR}"/${PKG_NAME}-*.whl >/dev/null 2>&1; then
  echo "INFO: wheel not found in ${PKG_DIR}; building one first."
  ./build_vllm_iluvatar.sh
fi

latest_pkg="$(ls -t "${PKG_DIR}"/${PKG_NAME}-*.whl | head -1)"
if [[ -z "${latest_pkg}" ]]; then
  echo "ERROR: Cannot find ${PKG_NAME} wheel in ${PKG_DIR}"
  exit 1
fi

echo "INFO: Installing ${latest_pkg}"

if [[ -n "${TARGET_DIR}" ]]; then
  python_dist_path="${TARGET_DIR}/lib/python3/dist-packages"
  mkdir -p "${python_dist_path}"


  python_dist_path_lib64="${TARGET_DIR}/lib64/python3/dist-packages"
  export PYTHONPATH="${python_dist_path}:${python_dist_path_lib64}${PYTHONPATH:+:${PYTHONPATH}}"


  PIP_INSTALL_FLAGS=(-t "${python_dist_path}")
  "${PYTHON_PATH}" -m pip install "${PIP_INSTALL_FLAGS[@]}" --upgrade "${latest_pkg}"
  echo "INFO: ${PKG_NAME} installed in ${python_dist_path}; please add it to PYTHONPATH."
else
  "${PYTHON_PATH}" -m pip uninstall "${PKG_NAME}" -y || true
  "${PYTHON_PATH}" -m pip install "${latest_pkg}"
fi

install_runtime_deps