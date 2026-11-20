#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility entry; maintained parameters live in run_mr_decode.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_mr_decode.sh" "$@"
