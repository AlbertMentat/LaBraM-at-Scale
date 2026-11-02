#!/bin/bash
# Training recipe. Set data paths and distributed launch variables for your hardware.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
set -euo pipefail
export OMP_NUM_THREADS=8
RELEASE=/data/wds549

# Singularity exports WORLD_SIZE/RANK per node; single-node jobs leave them unset
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
echo "dist: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"

python -m torch.distributed.run \
  --nproc_per_node 8 \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  train_vqnsp_wds549.py \
  --wds549_root       "$RELEASE" \
  --wds549_catalog_root "$RELEASE/catalogs/phase5-global-20260817" \
  --wds549_policy     configs/wds_549_full_rebuild_policy_20260815.json \
  --batch_size 192 \
  --epochs 40 \
  --warmup_epochs 15 \
  --lr 5e-6 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --clip_grad 1.0 \
  --save_ckpt_freq 1 \
  --save_ckpt_steps 500 \
  --num_workers 10 \
  --output_dir /data/outputs/molecule-tokenizer \
  --log_dir    /data/outputs/molecule-tokenizer/logs


