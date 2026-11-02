#!/usr/bin/env bash
# VQ-NSP tokenizer on the WDS-549 full-R1 release. Set OUTPUT_DIR and LOG_DIR to your output locations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$HERE"
cd "$REPO"

RELEASE_ROOT="${RELEASE_ROOT:-/data/wds549}"
CATALOG_ROOT="${CATALOG_ROOT:-$RELEASE_ROOT/catalogs/phase5-global-20260817}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/checkpoints/vqnsp_wds549}"
LOG_DIR="${LOG_DIR:-$HERE/logs/vqnsp_wds549}"
NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count() or 1)')}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-5}"
SAVE_CKPT_STEPS="${SAVE_CKPT_STEPS:-200}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "release  : $RELEASE_ROOT"
echo "catalog  : $CATALOG_ROOT"
echo "outputs  : $OUTPUT_DIR"
echo "logs     : $LOG_DIR"

torchrun --nproc_per_node="$NPROC" train_vqnsp_wds549.py \
  --wds549_root "$RELEASE_ROOT" \
  --wds549_catalog_root "$CATALOG_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --save_ckpt_freq 1 \
  --save_ckpt_steps "$SAVE_CKPT_STEPS" \
  --num_workers 8 \
  --pin_mem \
  --auto_resume \
  "$@"

