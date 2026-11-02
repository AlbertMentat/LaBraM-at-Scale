#!/usr/bin/env bash
# Quark Large: 32 GPUs, global batch 512 in the reported run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/quark_paths.sh"
quark_prepare_paths "$@"
cd "$QUARK_REPO_ROOT/legacy_25k"
export MODEL=labram_large_patch200_1600_8k_vocab
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export CLIP_GRAD="${CLIP_GRAD:-3.0}"
bash scripts/run_pretrain.sh "${QUARK_ARGS[@]}"
