#!/bin/bash
# =============================================================================
# Stage 2: LaBraM Pre-training — Masked EEG Modeling
#
# Pre-trains LaBraM with masked patch prediction using VQ-NSP tokens as targets.
# Supports single-node and multi-node distributed training.
# =============================================================================

set -e

# =============================================================================
# GPU / Distributed Configuration
# =============================================================================
if [[ -n "${NPROC:-}" ]]; then
    GPUS="$NPROC"
elif [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
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
MODEL="${MODEL:-labram_huge_patch200_1600_8k_vocab}"

# Tokenizer (from Stage 1)
TOKENIZER_MODEL="vqnsp_encoder_base_decoder_3x200x12"
TOKENIZER_WEIGHT="${TOKENIZER_WEIGHT:-/data/checkpoints/quark-tokenizer.pth}"

# Codebook
CODEBOOK_SIZE=8192
CODEBOOK_DIM=32

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-6}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
OPT_BETAS="0.9 0.98"

# =============================================================================
# Logging & Checkpoints
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/quark_huge}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-0}"

# Resume from checkpoint (leave empty to train from scratch)
RESUME="${RESUME:-}"

# =============================================================================
# Build Command
# =============================================================================
read -r -a DISTRIBUTED_OPTIONS <<< "$DISTRIBUTED_ARGS"
read -r -a OPT_BETA_OPTIONS <<< "$OPT_BETAS"
CMD=(torchrun "${DISTRIBUTED_OPTIONS[@]}" run_labram_pretraining.py)

# Dataset
CMD+=(--dataset_configs "$DATASET_CONFIGS")
CMD+=(--num_workers "$NUM_WORKERS")

# Model
CMD+=(--model "$MODEL")
CMD+=(--tokenizer_model "$TOKENIZER_MODEL")
CMD+=(--tokenizer_weight "$TOKENIZER_WEIGHT")
CMD+=(--codebook_size "$CODEBOOK_SIZE")
CMD+=(--codebook_dim "$CODEBOOK_DIM")

# Optimizer & Schedule
CMD+=(--batch_size "$BATCH_SIZE")
CMD+=(--epochs "$EPOCHS")
CMD+=(--warmup_epochs "$WARMUP_EPOCHS")
CMD+=(--lr "$LEARNING_RATE")
CMD+=(--min_lr "$MIN_LR")
CMD+=(--weight_decay "$WEIGHT_DECAY")
CMD+=(--clip_grad "$CLIP_GRAD")
CMD+=(--opt_betas "${OPT_BETA_OPTIONS[@]}")

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
echo "  Stage 2: LaBraM Pre-training (Masked EEG Modeling)"
echo "=========================================================="
echo ""
echo "  Config         : $DATASET_CONFIGS"
echo "  Model          : $MODEL"
echo "  Tokenizer      : $TOKENIZER_MODEL"
echo "  Tokenizer Ckpt : $TOKENIZER_WEIGHT"
echo "  Codebook       : $CODEBOOK_SIZE entries x $CODEBOOK_DIM dim"
echo "  Batch Size     : $BATCH_SIZE per GPU x $GPUS GPUs x ${WORLD_SIZE:-1} nodes"
echo "  Epochs         : $EPOCHS (warmup: $WARMUP_EPOCHS)"
echo "  LR             : $LEARNING_RATE -> $MIN_LR (cosine)"
echo "  Clip Grad      : $CLIP_GRAD"
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
echo "  LaBraM pre-training complete!"
echo "  Results saved to: $OUTPUT_DIR"
echo "=========================================================="

