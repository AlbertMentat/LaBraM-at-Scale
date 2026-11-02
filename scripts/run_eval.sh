#!/bin/bash
# One downstream fit of a pretrained LaBraM checkpoint: full fine-tuning or linear probing.
#
# Required:
#   MODE        ft | lp_pool | lp_nopool
#   FAMILY      25k | 49k | 79k      (selects the input normalisation used in pretraining)
#   MODEL       labram_base_patch200_200 | labram_large_patch200_200 | labram_huge_patch200_200
#   CHECKPOINT  pretrained checkpoint (.safetensors or .pth)
#   DATA_DIR    task directory: train/ val/ test/ (*.pkl with X, y, ch_names) + label_map.json
#   OUTPUT_DIR  where results go
# Optional hyperparameters (defaults in brackets):
#   SEED [42]
#   FT:  LR [3e-4] LAYER_DECAY [0.75] WEIGHT_DECAY [0.05] EPOCHS [50] WARMUP_EPOCHS [3]
#        BATCH_SIZE [64] UPDATE_FREQ [1] NPROC [number of GPUs]
#   LP:  LR [5e-4] WEIGHT_DECAY [0.05]
#
# Example:
#   MODE=lp_pool FAMILY=79k MODEL=labram_base_patch200_200 CHECKPOINT=ckpt/molecule_base.pth \
#   DATA_DIR=data/TUAB OUTPUT_DIR=out/tuab_lp bash scripts/run_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODE:?}" "${FAMILY:?}" "${MODEL:?}" "${CHECKPOINT:?}" "${DATA_DIR:?}" "${OUTPUT_DIR:?}"
SEED="${SEED:-42}"
mkdir -p "$OUTPUT_DIR"

case "$FAMILY" in
    79k) LP_NORM=zscore;   FT_NORM_ARGS=(--input_scale 1.0) ;;                # per-sample z-score inside the model input
    25k|49k) LP_NORM=scale100; FT_NORM_ARGS=(--input_scale 0.01 --skip_norm) ;;  # signal / 100, no z-score
    *) echo "FAMILY must be 25k, 49k or 79k" >&2; exit 2 ;;
esac

case "$MODE" in
    ft)
        CONFIG="$OUTPUT_DIR/dataset_config.json"
        printf '[{"type": "downstream", "name": "%s", "data_dir": "%s", "source_dataset_type": "downstream"}]\n' \
            "$(basename "$DATA_DIR")" "$(realpath "$DATA_DIR")" > "$CONFIG"
        NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
        torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" run_unified_finetuning.py \
            --model "$MODEL" --finetune "$CHECKPOINT" --dataset_config "$CONFIG" \
            --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR/logs" --seed "$SEED" \
            --batch_size "${BATCH_SIZE:-64}" --update_freq "${UPDATE_FREQ:-1}" \
            --epochs "${EPOCHS:-50}" --warmup_epochs "${WARMUP_EPOCHS:-3}" \
            --lr "${LR:-3e-4}" --layer_decay "${LAYER_DECAY:-0.75}" --weight_decay "${WEIGHT_DECAY:-0.05}" \
            "${FT_NORM_ARGS[@]}" --model_selection_metric balanced_accuracy --use_class_weight \
            --save_best_analysis_checkpoint --save_ckpt --save_ckpt_freq 5 --max_numbered_checkpoints 1 --auto_resume \
            --num_workers 8 --pin_mem
        ;;
    lp_pool|lp_nopool)
        READOUT=pooled; [[ "$MODE" == lp_nopool ]] && READOUT=no_pool
        python run_linear_probe.py --model "$MODEL" --checkpoint "$CHECKPOINT" --data_dir "$DATA_DIR" \
            --readout "$READOUT" --input_norm "$LP_NORM" --lr "${LR:-5e-4}" \
            --weight_decay "${WEIGHT_DECAY:-0.05}" --seed "$SEED" --output_dir "$OUTPUT_DIR"
        ;;
    *) echo "MODE must be ft, lp_pool or lp_nopool" >&2; exit 2 ;;
esac

