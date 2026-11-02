#!/bin/bash
# Training recipe. Set data paths and distributed launch variables for your hardware.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
set -euo pipefail
python -c "import torch, timm, einops; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'devices=', torch.cuda.device_count(), 'timm=', timm.__version__, 'einops=', einops.__version__); assert torch.version.hip is None and torch.cuda.is_available(); assert torch.cuda.device_count() == 8; assert all('A100' in torch.cuda.get_device_name(index) for index in range(8))"
export OMP_NUM_THREADS=8
export NPROC=8
export RELEASE_ROOT="${RELEASE_ROOT:-/data/wds549}"
export CATALOG_ROOT="${CATALOG_ROOT:-$RELEASE_ROOT/catalogs/phase5-global-20260817}"
export TOKENIZER_WEIGHT="${TOKENIZER_WEIGHT:-/data/checkpoints/molecule-tokenizer.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/molecule_huge_wds549_a100_4node_fresh_0913}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
if [[ -e "$OUTPUT_DIR/checkpoint.pth" || -e "$OUTPUT_DIR/checkpoint_alt.pth" ]] || compgen -G "$OUTPUT_DIR/checkpoint-*.pth" > /dev/null; then
  echo "Refusing to overwrite existing Huge checkpoints in the fresh output directory" >&2
  exit 1
fi
export EPOCHS=50
export BATCH_SIZE=16
export LR=1e-4
export SAVE_CKPT_STEPS=200
export USE_FSDP=1
SIZE=huge bash scripts/pretrain_wds549.sh --no_auto_resume --resume "" --init_from "" --start_epoch 0 --start_step_in_epoch 0


