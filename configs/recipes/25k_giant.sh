#!/bin/bash
# Quark Giant on an existing ROCm/PyTorch installation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/quark_paths.sh"
quark_prepare_paths "$@"
cd "$QUARK_REPO_ROOT"
bash legacy_25k/giant/run_pretrain_giant_fsdp_mi300x.sh "${QUARK_ARGS[@]}"
