#!/bin/bash
# Editable (in-place) install of the vllm_iluvatar plugin into the current env.
# Pure-Python package (no csrc / no compilation): we reuse the environment's
# vendored corex torch + ixformer and do NOT pull build isolation or deps.
#
# Install vllm (天数 vllm) FIRST, then run this script to layer the plugin on top.
SCRIPT_DIR=$(dirname $(realpath "$0"))
cd "${SCRIPT_DIR}"
PYTHON_PATH=$(which python3)

# vcs/setuptools may shell out to git; running as root rejects user dirs
# (dubious ownership). Allow all paths for root's git config only.
git config --global --add safe.directory '*'
echo "INFO: git safe.directory set to '*'"

# --no-build-isolation: build with the env's setuptools (no isolated env).
# --no-deps: torch / ixformer come from the corex environment, not PyPI.
# The env-gated startup hook .pth (vllm_iluvatar_startup_patch.pth) is emitted
# by setup.py's editable_wheel cmdclass, so no manual copy is needed here.
echo "Python cmd: ${PYTHON_PATH} -m pip install -e . --no-build-isolation --no-deps -v"
${PYTHON_PATH} -m pip install -e . --no-build-isolation --no-deps -v || exit

echo "INFO: vllm_iluvatar editable install complete."
exit 0
