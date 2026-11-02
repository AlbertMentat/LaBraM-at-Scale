#!/usr/bin/env bash
# LaBraM pre-training (stage 2) on the WDS-549 full-R1 release.
# Usage: SIZE=base|large|huge|giant|giant_m bash pretrain_wds549.sh
# giant/giant_m default to FSDP; everything else uses DDP.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$HERE"
cd "$REPO"

SIZE="${SIZE:-base}"
case "$SIZE" in
  base)    MODEL=labram_base_patch200_1600_8k_vocab;    DEF_BS=64; DEF_LR=5e-4; FSDP=0 ;;
  large)   MODEL=labram_large_patch200_1600_8k_vocab;   DEF_BS=32; DEF_LR=3e-4; FSDP=0 ;;
  huge)    MODEL=labram_huge_patch200_1600_8k_vocab;    DEF_BS=16; DEF_LR=1e-4; FSDP=1 ;;
  giant)   MODEL=labram_giant_patch200_1600_8k_vocab;   DEF_BS=8;  DEF_LR=1e-4; FSDP=1 ;;
  giant_m) MODEL=labram_giant_m_patch200_1600_8k_vocab; DEF_BS=8;  DEF_LR=1e-4; FSDP=1 ;;
  *) echo "unknown SIZE: $SIZE (base|large|huge|giant|giant_m)" >&2; exit 2 ;;
esac

RELEASE_ROOT="${RELEASE_ROOT:-/data/wds549}"
CATALOG_ROOT="${CATALOG_ROOT:-$RELEASE_ROOT/catalogs/phase5-global-20260817}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/data/outputs/molecule-tokenizer}"
TOKENIZER_WEIGHT="${TOKENIZER_WEIGHT:-$TOKENIZER_ROOT/molecule-tokenizer.pth}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-vqnsp_encoder_base_decoder_3x200x12}"

OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/molecule_${SIZE}_wds549}"
LOG_DIR="${LOG_DIR:-/data/outputs/molecule_${SIZE}_wds549/logs}"
BATCH_SIZE="${BATCH_SIZE:-$DEF_BS}"
LR="${LR:-$DEF_LR}"
if [[ "$SIZE" == "large" ]]; then DEF_EPOCHS=30; else DEF_EPOCHS=50; fi
EPOCHS="${EPOCHS:-$DEF_EPOCHS}"
SAVE_CKPT_STEPS="${SAVE_CKPT_STEPS:-200}"
USE_FSDP="${USE_FSDP:-$FSDP}"
NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count() or 1)')}"

[[ -f "$TOKENIZER_WEIGHT" ]] || { echo "tokenizer weight not found: $TOKENIZER_WEIGHT" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ENTRY=run_labram_pretraining.py
[[ "$USE_FSDP" == "1" ]] && ENTRY=run_labram_pretraining_fsdp.py

GLOBAL_BATCH=$(( BATCH_SIZE * NPROC * ${WORLD_SIZE:-1} ))
echo "size=$SIZE model=$MODEL entry=$ENTRY bs=$BATCH_SIZE lr=$LR"
echo "global_batch=$GLOBAL_BATCH (bs x $NPROC gpu x ${WORLD_SIZE:-1} node); lr is NOT auto-scaled"
echo "outputs=$OUTPUT_DIR"

NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
echo "dist: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"

torchrun --nproc_per_node="$NPROC" \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  "$ENTRY" \
  --model "$MODEL" \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --tokenizer_weight "$TOKENIZER_WEIGHT" \
  --use_wds --wds_format wds-549 \
  --wds_root "$RELEASE_ROOT" \
  --wds_catalog_root "$CATALOG_ROOT" \
  --input_scale 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --save_ckpt_freq 1 \
  --save_ckpt_steps "$SAVE_CKPT_STEPS" \
  --num_workers 8 \
  --pin_mem \
  --auto_resume \
  "$@"

