#!/usr/bin/env bash
# Continue Molecule Large from a completed epoch with the continuation contract.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the existing Molecule Large run}"
EPOCH_CKPT=$(python scripts/select_completed_epoch_checkpoint.py "$OUTPUT_DIR")
RESUME="$EPOCH_CKPT" bash configs/recipes/79k_large.sh --resume "$EPOCH_CKPT" --no_auto_resume "$@"
