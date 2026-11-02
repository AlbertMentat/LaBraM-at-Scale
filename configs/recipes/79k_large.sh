#!/bin/bash
# Training recipe. Set data paths and distributed launch variables for your hardware.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
set -euo pipefail
export OMP_NUM_THREADS=8
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/molecule_large_wds549}"
export LOG_DIR="${LOG_DIR:-/data/outputs/molecule_large_wds549/logs}"
export EPOCHS="${EPOCHS:-30}"
SIZE=large bash scripts/pretrain_wds549.sh "$@"
