#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "INFO: Cleaning vllm_iluvatar build artifacts in ${ROOT_DIR}"

echo "INFO: Removing build directories"
rm -rf build
rm -rf build_pip
rm -rf dist

echo "INFO: Removing Python package metadata"
rm -rf ./*.egg-info
rm -rf ./.eggs

echo "INFO: Removing generated logs and compile database"
rm -rf compile.log
rm -f compile_commands.json

echo "INFO: Removing in-place native extensions"
rm -f vllm_iluvatar/_C*.so

echo "INFO: Clean complete"