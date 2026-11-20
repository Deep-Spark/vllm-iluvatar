#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Existing Tencent wave protocol with actual response cache accounting.
set -euo pipefail
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: bench_prefill.sh RUN_NAME [--dry-run]
Required: TENCENT_CLIENT_DIR (contains measure_prefill.py and prefill_throughput.py).
Overrides: BASE_URL=http://127.0.0.1:12715 MODEL=GLM-5-Next
  CC_LIST=1,8,16,32 ROUNDS=3 PREFIX_RATIOS=0.93 SEED=24357209
  RESULTS_ROOT=<workspace>/logs INPUT_TOKENS=112640 PYTHON_BIN=python3
  CACHE_ROUNDING_TOLERANCE=<maximum token deficit from the requested prefix>
Cold-prefix reference: PREFIX_RATIOS=0 CC_LIST=1,8.
Use a new RUN_NAME and seed range for additional waves on the same service.
Output is never overwritten. --dry-run does not contact the service.
HELP
    exit 0 ;;
esac
[[ $# -ge 1 && $# -le 2 && "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Use RUN_NAME [--dry-run]; see --help.' >&2; exit 2; }
[[ $# == 1 || "$2" == --dry-run ]] || { echo 'Unknown argument' >&2; exit 2; }
: "${TENCENT_CLIENT_DIR:?Set TENCENT_CLIENT_DIR to the existing Tencent client directory}"
test -r "$TENCENT_CLIENT_DIR/measure_prefill.py"
test -r "$TENCENT_CLIENT_DIR/prefill_throughput.py"
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
out="${RESULTS_ROOT:-$workspace/logs}/$1"
cmd=("${PYTHON_BIN:-python3}" -u "$TENCENT_CLIENT_DIR/measure_prefill.py"
  --base-url "${BASE_URL:-http://127.0.0.1:12715}" --model "${MODEL:-GLM-5-Next}"
  --input-tokens "${INPUT_TOKENS:-112640}" --prefix-ratios "${PREFIX_RATIOS:-0.93}"
  --cc "${CC_LIST:-1,8,16,32}" --rounds "${ROUNDS:-3}"
  --seed "${SEED:-24357209}" --ttft-slo-s 5 --output "$out")
if [[ -n "${CACHE_ROUNDING_TOLERANCE:-}" ]]; then
  cmd+=(--cache-rounding-tolerance "$CACHE_ROUNDING_TOLERANCE")
fi
printf '%q ' "${cmd[@]}"; printf '\n'
[[ "${2:-}" != --dry-run ]] || exit 0
test ! -e "$out"
export PYTHONDONTWRITEBYTECODE=1
exec "${cmd[@]}"
