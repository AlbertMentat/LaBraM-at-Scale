#!/bin/bash
# Training recipe. Set data paths and distributed launch variables for your hardware.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
set -euo pipefail
export OMP_NUM_THREADS=8
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/molecule_base_wds549_any}"
export EPOCHS="${EPOCHS:-50}"
SIZE=base bash scripts/pretrain_wds549.sh "$@"


