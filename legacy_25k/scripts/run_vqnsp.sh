#!/bin/bash
# =============================================================================
# Stage 1: VQ-NSP Training — trains the discrete EEG tokenizer
#
# Trains a VQ-VAE to tokenize EEG patches into discrete codes.
# Supports single-node and multi-node distributed training.
# =============================================================================

set -e

# =============================================================================
# GPU / Distributed Configuration
# =============================================================================
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    GPUS=$(nvidia-smi -L | wc -l)
fi

if [[ -z "${WORLD_SIZE}" ]]; then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS"
else
    if (($WORLD_SIZE == 1)); then
        DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
            --master_port $MASTER_PORT"
    else
        DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
            --nnodes $WORLD_SIZE \
            --node_rank $RANK \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT"
    fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_CONFIGS="${DATASET_CONFIGS:-configs/dataset_configs_example.json}"

# =============================================================================
# Model
# =============================================================================
MODEL="${MODEL:-vqnsp_encoder_base_decoder_3x200x12}"
CODEBOOK_N_EMD=8192
CODEBOOK_EMD_DIM=32

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE=64
EPOCHS=100
WARMUP_EPOCHS=10
LEARNING_RATE=5e-5
MIN_LR=1e-5
WEIGHT_DECAY=1e-4
OPT="adamw"
OPT_BETAS="0.9 0.99"
CLIP_GRAD=3.0

# =============================================================================
# Logging & Checkpoints
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/quark-tokenizer}"
SAVE_CKPT_FREQ=20
NUM_WORKERS=10
SEED=0

# Resume from checkpoint (leave empty to train from scratch)
RESUME="${RESUME:-}"

# =============================================================================
# Build Command
# =============================================================================
read -r -a DISTRIBUTED_OPTIONS <<< "$DISTRIBUTED_ARGS"
read -r -a OPT_BETA_OPTIONS <<< "$OPT_BETAS"
CMD=(torchrun "${DISTRIBUTED_OPTIONS[@]}" run_vqnsp_training.py)

# Dataset
CMD+=(--dataset_configs "$DATASET_CONFIGS")
CMD+=(--num_workers "$NUM_WORKERS")

# Model
CMD+=(--model "$MODEL")
CMD+=(--codebook_n_emd "$CODEBOOK_N_EMD")
CMD+=(--codebook_emd_dim "$CODEBOOK_EMD_DIM")
CMD+=(--quantize_kmeans_init)

# Optimizer & Schedule
CMD+=(--opt "$OPT")
CMD+=(--opt_betas "${OPT_BETA_OPTIONS[@]}")
CMD+=(--batch_size "$BATCH_SIZE")
CMD+=(--epochs "$EPOCHS")
CMD+=(--warmup_epochs "$WARMUP_EPOCHS")
CMD+=(--lr "$LEARNING_RATE")
CMD+=(--min_lr "$MIN_LR")
CMD+=(--weight_decay "$WEIGHT_DECAY")
CMD+=(--clip_grad "$CLIP_GRAD")

# Logging & Checkpoints
CMD+=(--output_dir "$OUTPUT_DIR")
CMD+=(--save_ckpt_freq "$SAVE_CKPT_FREQ")
CMD+=(--seed "$SEED")

if [ -n "$RESUME" ]; then
    CMD+=(--resume "$RESUME")
fi

# =============================================================================
# Run
# =============================================================================
echo "=========================================================="
echo "  Stage 1: VQ-NSP Training"
echo "=========================================================="
echo ""
echo "  Config         : $DATASET_CONFIGS"
echo "  Model          : $MODEL"
echo "  Codebook       : $CODEBOOK_N_EMD entries x $CODEBOOK_EMD_DIM dim"
echo "  Batch Size     : $BATCH_SIZE per GPU x $GPUS GPUs x ${WORLD_SIZE:-1} nodes"
echo "  Epochs         : $EPOCHS (warmup: $WARMUP_EPOCHS)"
echo "  LR             : $LEARNING_RATE -> $MIN_LR (cosine)"
echo "  Weight Decay   : $WEIGHT_DECAY"
echo "  GPUs per node  : $GPUS"
echo "  Nodes          : ${WORLD_SIZE:-1}"
echo "  Output         : $OUTPUT_DIR"
echo "=========================================================="
echo ""
echo "Command:"
echo "  ${CMD[*]}"
echo ""

mkdir -p "$OUTPUT_DIR"

"${CMD[@]}" "$@"

echo ""
echo "=========================================================="
echo "  VQ-NSP training complete!"
echo "  Results saved to: $OUTPUT_DIR"
echo "=========================================================="

