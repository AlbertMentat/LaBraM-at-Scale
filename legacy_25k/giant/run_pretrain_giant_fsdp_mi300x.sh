#!/bin/bash
# =============================================================================
# Stage 2: LaBraM Pre-training -- Giant (~2B) with FSDP, MI300X edition
#
# Differences vs run_pretrain_giant_fsdp.sh (A100-40G):
#   * GPU count detected via rocm-smi (with nvidia-smi fallback for CUDA test).
#   * BATCH_SIZE bumped 4 -> 16. MI300X has 192GB HBM3 vs A100 40GB, so per-GPU
#     batch can be 4-8x larger even with FSDP+GC. Profile and push higher
#     (32, 64) if memory allows.
#   * Gradient checkpointing kept on by default; on MI300X you can usually
#     drop --gradient_checkpointing to gain ~20-30% step throughput.
#   * Output dir suffixed _mi300x to avoid clobbering A100 runs.
# =============================================================================

set -e

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
    echo "ERROR: cannot detect GPU count (no rocm-smi / nvidia-smi found)" >&2
    exit 1
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
echo "Detected GPUS=${GPUS}  WORLD_SIZE=${WORLD_SIZE:-1}"

if command -v rocm-smi >/dev/null 2>&1; then
    echo "---- rocm-smi ----"
    rocm-smi --showproductname --showmeminfo vram | head -40 || true
    echo "------------------"
fi

# ---------- Dataset ----------
DATASET_CONFIGS="${DATASET_CONFIGS:-legacy_25k/configs/dataset_configs_example.json}"

# ---------- Model ----------
MODEL="${MODEL:-labram_giant_patch200_1600_8k_vocab}"
TOKENIZER_MODEL="vqnsp_encoder_base_decoder_3x200x12"
TOKENIZER_WEIGHT="${TOKENIZER_WEIGHT:-/data/checkpoints/quark-tokenizer.pth}"
CODEBOOK_SIZE=8192
CODEBOOK_DIM=32

# ---------- Hyperparameters ----------
BATCH_SIZE=16          # was 4 on A100-40G
EPOCHS=30
WARMUP_EPOCHS=8
LEARNING_RATE=1e-4
MIN_LR=1e-5
WEIGHT_DECAY=0.05
CLIP_GRAD=0.5
OPT_BETAS="0.9 0.98"

# ---------- Logging ----------
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/quark_giant_mi300x}"
SAVE_CKPT_FREQ=1   # Plan B: save every epoch (Basic SLA = frequent preemption)
NUM_WORKERS=10
SEED=0
RESUME="${RESUME:-}"

# ---------- Build command ----------
read -r -a DISTRIBUTED_OPTIONS <<< "$DISTRIBUTED_ARGS"
read -r -a OPT_BETA_OPTIONS <<< "$OPT_BETAS"
CMD=(torchrun "${DISTRIBUTED_OPTIONS[@]}" --module legacy_25k.giant.run_labram_pretraining_fsdp)
CMD+=(--dataset_configs "$DATASET_CONFIGS")
CMD+=(--num_workers "$NUM_WORKERS")
CMD+=(--model "$MODEL")
CMD+=(--tokenizer_model "$TOKENIZER_MODEL")
CMD+=(--tokenizer_weight "$TOKENIZER_WEIGHT")
CMD+=(--codebook_size "$CODEBOOK_SIZE")
CMD+=(--codebook_dim "$CODEBOOK_DIM")
CMD+=(--batch_size "$BATCH_SIZE")
CMD+=(--epochs "$EPOCHS")
CMD+=(--warmup_epochs "$WARMUP_EPOCHS")
CMD+=(--lr "$LEARNING_RATE")
CMD+=(--min_lr "$MIN_LR")
CMD+=(--weight_decay "$WEIGHT_DECAY")
CMD+=(--clip_grad "$CLIP_GRAD")
CMD+=(--opt_betas "${OPT_BETA_OPTIONS[@]}")
CMD+=(--gradient_checkpointing)   # drop this on MI300X for more throughput
CMD+=(--output_dir "$OUTPUT_DIR")
CMD+=(--save_ckpt_freq "$SAVE_CKPT_FREQ")
CMD+=(--seed "$SEED")

if [ -n "$RESUME" ]; then
    CMD+=(--resume "$RESUME")
fi

echo "=========================================================="
echo "  Stage 2: LaBraM Pre-training Giant (~2B, FSDP) -- MI300X"
echo "=========================================================="
echo "  Config         : $DATASET_CONFIGS"
echo "  Model          : $MODEL"
echo "  Tokenizer      : $TOKENIZER_MODEL"
echo "  Tokenizer Ckpt : $TOKENIZER_WEIGHT"
echo "  Codebook       : $CODEBOOK_SIZE entries x $CODEBOOK_DIM dim"
echo "  Batch Size     : $BATCH_SIZE per GPU x $GPUS GPUs x ${WORLD_SIZE:-1} nodes"
echo "  Epochs         : $EPOCHS (warmup: $WARMUP_EPOCHS)"
echo "  LR             : $LEARNING_RATE -> $MIN_LR (cosine)"
echo "  FSDP           : enabled (FULL_SHARD)"
echo "  Grad Checkpoint: enabled"
echo "  GPUs per node  : $GPUS"
echo "  Nodes          : ${WORLD_SIZE:-1}"
echo "  Output         : $OUTPUT_DIR"
echo "=========================================================="
echo "Command:"
echo "  ${CMD[*]}"
echo ""

mkdir -p "$OUTPUT_DIR"

"${CMD[@]}" "$@"

echo "=========================================================="
echo "  LaBraM Giant (FSDP, MI300X) pre-training complete!"
echo "  Results saved to: $OUTPUT_DIR"
echo "=========================================================="

