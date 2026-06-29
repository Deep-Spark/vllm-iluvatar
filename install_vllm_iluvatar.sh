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

if [[ "${EDITABLE:-0}" == "1" ]]; then
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
  "${PYTHON_PATH}" -m pip install --upgrade -t "${python_dist_path}" "${latest_pkg}"
  echo "INFO: ${PKG_NAME} installed in ${python_dist_path}; please add it to PYTHONPATH."
else
  "${PYTHON_PATH}" -m pip uninstall "${PKG_NAME}" -y || true
  "${PYTHON_PATH}" -m pip install "${latest_pkg}"
fi