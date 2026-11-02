#!/bin/bash
# Atom tokenizer recipe; install the pinned requirements before launching.
set -euo pipefail
cd "$(dirname "$0")/../.."
export USE_WDS=1
export WDS_ROOT="${WDS_ROOT:-${WDS_ROOT:-/data/atom-wds}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR:-/data/outputs/atom-tokenizer}}"
bash scripts/run_vqnsp_legacy_wds_49k.sh "$@"
