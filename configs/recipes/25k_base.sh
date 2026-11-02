#!/usr/bin/env bash
# Quark Base: 32 GPUs, global batch 2048 in the reported run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/quark_paths.sh"
quark_prepare_paths "$@"
cd "$QUARK_REPO_ROOT/legacy_25k"
export MODEL=labram_base_patch200_1600_8k_vocab
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export CLIP_GRAD="${CLIP_GRAD:-3.0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/quark_base}"
bash scripts/run_pretrain.sh "${QUARK_ARGS[@]}"
