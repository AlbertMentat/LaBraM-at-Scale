#!/bin/bash
# Quark recipe; configure data paths and distributed settings in legacy_25k/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/quark_paths.sh"
quark_prepare_paths "$@"
cd "$QUARK_REPO_ROOT/legacy_25k"
bash scripts/run_vqnsp.sh "${QUARK_ARGS[@]}"
