#!/bin/bash
# Training recipe. Set data paths and distributed launch variables for your hardware.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZER_WEIGHT="${TOKENIZER_WEIGHT:-/data/checkpoints/atom-tokenizer.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/atom_large_a100_wds}"
export WDS_ROOT="${WDS_ROOT:-/data/atom-wds}"
export WDS_CACHE_DIR=/scratch/wds_cache
set -e
echo "===== [A100] CUDA sanity + numpy2.x ====="
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())"
nvidia-smi -L || true
python -c "import numpy; assert numpy.__version__.split('.')[0]=='2'; print('numpy', numpy.__version__)"
python -c "import scipy; from scipy.signal import resample; from sklearn.metrics import r2_score; import pandas; print('deps OK')"
echo "===== [A100] launching Stage-2 pretrain (inlined) ====="
GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | awk -F',' '{print NF}')
if [ -z "${WORLD_SIZE}" ] || [ "${WORLD_SIZE}" = "1" ]; then DARGS="--nproc_per_node $GPUS --master_port 6105"; else DARGS="--nproc_per_node $GPUS --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port 6105"; fi
echo "DARGS=$DARGS GPUS=$GPUS WORLD_SIZE=${WORLD_SIZE:-1}"
torchrun $DARGS run_labram_pretraining.py \
  --use_wds --wds_root "${WDS_ROOT}" --wds_cache_dir "${WDS_CACHE_DIR}" --num_workers 10 \
  --model labram_large_patch200_1600_8k_vocab \
  --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 --tokenizer_weight "${TOKENIZER_WEIGHT}" \
  --codebook_size 8192 --codebook_dim 32 \
  --batch_size 64 --epochs 50 --warmup_epochs 10 \
  --lr 1.5e-4 --min_lr 1e-5 --weight_decay 0.05 --clip_grad 1.0 --opt_betas 0.9 0.98 \
  --output_dir "${OUTPUT_DIR}" --save_ckpt_freq 5 --save_ckpt_steps 500 --no_mid_epoch_resume --seed 0


