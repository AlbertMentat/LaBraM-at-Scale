#!/bin/bash
# =============================================================================
# Stage 1: VQ-NSP Tokenizer Training — MI300X edition
# Cloned from scripts/run_vqnsp.sh with MI300X-specific adjustments:
#   - GPU detection via rocm-smi (with nvidia-smi fallback)
#   - per-GPU batch size scaled for 192GB HBM3 to match A100 global batch
#   - Output dir suffixed _mi300x to keep A100 results intact
# =============================================================================

set -e

# Create MIOpen cache dir on local /scratch (writable, supports SQLite locks)
mkdir -p /scratch/miopen_cache

# ---------- GPU / Distributed ----------
if [[ -n "${HIP_VISIBLE_DEVICES}" ]]; then
    GPUS=$(echo "$HIP_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v rocm-smi >/dev/null 2>&1; then
    GPUS=$(rocm-smi -i 2>/dev/null | grep -c 'GPU\[')
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L | wc -l)
else
    echo "ERROR: cannot detect GPU count" >&2; exit 1
fi

if [[ -z "${WORLD_SIZE}" ]]; then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS"
else
    if (($WORLD_SIZE == 1)); then
        DISTRIBUTED_ARGS="--nproc_per_node $GPUS --master_port $MASTER_PORT"
    else
        DISTRIBUTED_ARGS="--nproc_per_node $GPUS --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    fi
fi
echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"
echo "Detected GPUS=${GPUS}  WORLD_SIZE=${WORLD_SIZE:-1}"
command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname || true

# ---------- Dataset / Model ----------
# 默认: 含下游评测集train (536集). 不含评测集版: dataset_configs_tok256_noeval.json (516集)
DATASET_CONFIGS="${DATASET_CONFIGS:-configs/dataset_configs_tok256_final.json}"
MODEL="vqnsp_encoder_base_decoder_3x200x12"
CODEBOOK_N_EMD=8192
CODEBOOK_EMD_DIM=32

# ---------- Hyperparameters ----------
# MI300X 2xG8 (16 GPU) x bs=128 = global 2048 (matches A100 4x40G8 x bs=64).
BATCH_SIZE=256  # 4节点32卡=global 8192(lr代码自动×global/128缩放)
EPOCHS=40
WARMUP_EPOCHS=${WARMUP_EPOCHS:-15}   # collapse-fix: longer warmup (was 10)
LEARNING_RATE=${LEARNING_RATE:-1.5e-5}   # base lr x(global/128); lowered to avoid codebook collapse (was 5e-5 -> peak 8e-4)
MIN_LR=${MIN_LR:-1e-5}   # original base (not scaled)
WEIGHT_DECAY=1e-4
OPT="adamw"
OPT_BETAS="0.9 0.99"
CLIP_GRAD=${CLIP_GRAD:-1.0}   # tighter grad clip (was 3.0; grad exploded >1e4)

# ---------- IO ----------
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/atom-tokenizer${USE_WDS:+_wds}}"  # 4节点32卡正式训练; USE_WDS时输出_wds隔离
SAVE_CKPT_FREQ=1
NUM_WORKERS=10  # 单节点8卡×10=80并发(参考giant working), 远低于64卡场景
SEED=0
RESUME="${RESUME:-}"
# WebDataset(.tar shard顺序读, 根治blobfuse百万小pkl IO瓶颈). USE_WDS=1启用
USE_WDS="${USE_WDS:-}"
WDS_ROOT="${WDS_ROOT:-/data/atom-wds}"

CMD="torchrun ${DISTRIBUTED_ARGS} run_vqnsp_training.py"
if [ -n "$USE_WDS" ]; then
  CMD="$CMD --use_wds --wds_root $WDS_ROOT --num_workers $NUM_WORKERS"
else
  CMD="$CMD --dataset_configs $DATASET_CONFIGS --num_workers $NUM_WORKERS"
fi
CMD="$CMD --model $MODEL --codebook_n_emd $CODEBOOK_N_EMD --codebook_emd_dim $CODEBOOK_EMD_DIM --quantize_kmeans_init"
CMD="$CMD --opt $OPT --opt_betas $OPT_BETAS"
CMD="$CMD --batch_size $BATCH_SIZE --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS"
CMD="$CMD --lr $LEARNING_RATE --min_lr $MIN_LR --weight_decay $WEIGHT_DECAY --clip_grad $CLIP_GRAD"
CMD="$CMD --output_dir $OUTPUT_DIR --save_ckpt_freq $SAVE_CKPT_FREQ --seed $SEED"
CMD="$CMD --save_ckpt_every_steps ${SAVE_CKPT_EVERY_STEPS:-500}"  # epoch内每500步存checkpoint.pth(抗抢断+续训)
# MI300X fast-path: bf16 autocast (no GradScaler), fused AdamW, disable eval each epoch
CMD="$CMD --use_bf16 --fused_optim --disable_eval"
[ -n "$RESUME" ] && CMD="$CMD --resume $RESUME"

echo "==== VQ-NSP (MI300X) ===="
echo "  bs=$BATCH_SIZE per GPU x $GPUS GPUs x ${WORLD_SIZE:-1} nodes  global=$(( BATCH_SIZE * GPUS * ${WORLD_SIZE:-1} ))"
echo "  epochs=$EPOCHS  lr=$LEARNING_RATE  output=$OUTPUT_DIR"
echo "  CMD: $CMD"
mkdir -p "$OUTPUT_DIR"
$CMD
echo "VQ-NSP MI300X training done -> $OUTPUT_DIR"

