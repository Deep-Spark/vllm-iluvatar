#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.
#
# Fetch the SWAPP/vllm source at the ref required by this plugin PR and run
# the vllm-side build/install scripts. Invoked by .ixci/build.json:steps.config
# so that `vllm` is importable before build_vllm_iluvatar.sh starts.
#
# CI-provided env (see .ixci/build.json):
#   VLLM_SOURCE_REPO   git URL of SWAPP/vllm
#   VLLM_SOURCE_REF    branch / tag / sha to checkout (default: iluvatar_main)
#   VLLM_SOURCE_DIR    workdir (default: ../vllm-source, relative to repo root)
#
# For PRs that also modify vllm source, the CI platform is expected to
# override VLLM_SOURCE_REF with the modified-branch commit declared in the
# PR description (convention TBD with the platform team; see .ixci notes).

set -euo pipefail



REPO="${VLLM_SOURCE_REPO:?VLLM_SOURCE_REPO is required}"
REF="${VLLM_SOURCE_REF:-4b38b39a75e}"
DEST="${VLLM_SOURCE_DIR:-../vllm-source}"

echo "INFO: prepare vllm source"
echo "INFO:   repo=${REPO}"
echo "INFO:   ref=${REF}"
echo "INFO:   dest=${DEST}"

# Force-wipe any leftover ${DEST}. Packaging hosts sometimes leave files with
# no write bit from a previous run, which makes a plain `rm -rf` silently
# no-op and the follow-up `git clone` blow up with "already exists". Reset
# permissions first, then delete, and bail out if anything survives.
if [ -e "${DEST}" ]; then
    chmod -R u+w "${DEST}" 2>/dev/null || true
    rm -rf "${DEST}"
fi
if [ -e "${DEST}" ]; then
    echo "ERROR: ${DEST} still exists after rm -rf (permission / mount issue?)" >&2
    ls -la "${DEST}" >&2 || true
    exit 1
fi

git clone --no-single-branch "${REPO}" "${DEST}"

pushd "${DEST}" >/dev/null

git checkout "${REF}"

echo "INFO: checked out $(git rev-parse HEAD)"

echo "INFO: build vllm from source"
bash build_vllm.sh

echo "INFO: install vllm from source"
bash install_vllm.sh

popd >/dev/null

echo "INFO: vllm source prepared at ${DEST}"
