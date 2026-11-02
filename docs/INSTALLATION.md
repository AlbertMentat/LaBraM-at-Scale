# Installation and first run

Run the commands below from the repository root on Linux, using Python 3.10.
Pretraining and the GPU evaluation recipes use PyTorch 2.8.0 with torchvision
0.23.0. NVIDIA GPUs require a driver compatible with the selected PyTorch CUDA
wheel. Multi-GPU launches use `torchrun`, which is installed with PyTorch.

## Create an environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For an NVIDIA machine, this example selects the CUDA 12.8 wheels:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

Choose a different supported PyTorch 2.8 wheel if your driver or accelerator
requires it; see the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280).
The wheel supplies its runtime libraries. Compiling optional CUDA extensions
requires a compatible local toolkit as well.

For CPU-only data tools and command-line checks, use this alternative in a
separate environment:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

The root requirements cover tokenizer training, EEG pretraining, FT, both LP
readouts and the supplied WDS tools. The WDS readers use Python's `tarfile`;
they do not require the external `webdataset` package. The default AdamW/DDP/FSDP
recipes do not require Apex or Horovod. Those packages are optional for code
paths that explicitly select their optimizers or distributed loss backend.

For multitask EEG + Qwen3, additionally install:

```bash
python -m pip install -r multitask/requirements.txt
python -m pip check
```

Its shared dependency versions agree with the root requirements, and it adds
the pinned Transformers, PEFT and Accelerate stack. Continue with
[the multitask guide](../multitask/README.md) for feature extraction, joint
training, automatic resume and saved-model evaluation.

## Check the installation

```bash
python -c "import torch; print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())"
python run_vqnsp_training.py --help
python train_vqnsp_wds549.py --help
python run_labram_pretraining.py --help
python run_labram_pretraining_fsdp.py --help
python run_unified_finetuning.py --help
python run_linear_probe.py --help
python tools/pack_webdataset.py --help
python tools/build_wds_index.py --help
python -m tools.finalize_wds_549_full_release --help
```

All help commands exit without loading datasets or checkpoints. On an NVIDIA
training machine, also check GPU visibility:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU is visible'; print(torch.cuda.get_device_name(0))"
```

For the 25k entrypoints, use:

```bash
python legacy_25k/run_vqnsp_training.py --help
python legacy_25k/run_labram_pretraining.py --help
python -m legacy_25k.giant.run_labram_pretraining_fsdp --help
```

Run the `-m` commands from the repository root: the WDS finalizer and legacy
Giant module import shared packages from that root. Use the same module form
when supplying their actual arguments (`torchrun --module ...` for a
distributed legacy Giant launch).

## Supply data and checkpoints

The release consumes prepared data. The `/data/...` paths are placeholders to
replace with your own locations; datasets and EEG weights are supplied
separately. Paths can be absolute. Paths passed to the shell launchers below
are resolved from the repository root.

| Workflow | Inputs to configure |
|---|---|
| 79k WDS-549 tokenizer | `RELEASE_ROOT`: prepared WDS-549 release, including its release marker; `CATALOG_ROOT`: matching global catalogs |
| 79k EEG pretraining | The same WDS paths plus `TOKENIZER_WEIGHT`: tokenizer checkpoint |
| 49k legacy WDS | `--wds_root`: prepared legacy WDS shards/index; pretraining also takes `--tokenizer_weight` |
| 25k legacy data | Adapt `legacy_25k/configs/dataset_configs_example.json`; use the corresponding legacy entrypoint |
| FT / LP | `CHECKPOINT`: pretrained EEG backbone; `DATA_DIR`: prepared task splits; `OUTPUT_DIR`: output directory |
| Multitask | EEG checkpoint, task indices/split manifests, feature-cache paths and Qwen3 snapshot/configuration; see the multitask guide |

FT/LP expects `<task>/{train,val,test}/*.pkl` and `<task>/label_map.json` as
specified in the [usage guide](USAGE.md#downstream-evaluation). Use a tokenizer
checkpoint for masked EEG pretraining and a pretrained EEG backbone checkpoint
for downstream evaluation. Their model names must match the weight files.

## Launch a workflow

For example, train the WDS-549 tokenizer on a prepared release:

```bash
NPROC=1 RELEASE_ROOT=/data/wds549 \
CATALOG_ROOT=/data/wds549/catalogs \
OUTPUT_DIR=outputs/tokenizer LOG_DIR=outputs/tokenizer/logs \
bash scripts/train_tokenizer_wds549.sh
```

Then launch Base EEG pretraining with the corresponding tokenizer weight:

```bash
SIZE=base NPROC=1 RELEASE_ROOT=/data/wds549 \
CATALOG_ROOT=/data/wds549/catalogs \
TOKENIZER_WEIGHT=/data/checkpoints/tokenizer.pth \
OUTPUT_DIR=outputs/pretrain LOG_DIR=outputs/pretrain/logs \
bash scripts/pretrain_wds549.sh
```

Use `configs/recipes/` for each model's global batch, learning rate, epoch count
and GPU layout. Update the data paths and distributed-launch settings for your
cluster. The tokenizer example above runs on one node; use the corresponding
recipe to configure multiple nodes.

Download an encoder before downstream evaluation:

```bash
python scripts/download_checkpoint.py --model molecule-base
```

For one downstream fit:

```bash
# Full fine-tuning, with explicit hyperparameters.
MODE=ft FAMILY=79k MODEL=labram_base_patch200_200 NPROC=1 \
CHECKPOINT=checkpoints/molecule-base.safetensors DATA_DIR=/data/tasks/TUAB \
OUTPUT_DIR=outputs/tuab_ft LR=3e-4 LAYER_DECAY=0.75 SEED=42 \
bash scripts/run_eval.sh

# Linear probing; change MODE to lp_nopool for the other readout.
MODE=lp_pool FAMILY=79k MODEL=labram_base_patch200_200 \
CHECKPOINT=checkpoints/molecule-base.safetensors DATA_DIR=/data/tasks/TUAB \
OUTPUT_DIR=outputs/tuab_lp LR=5e-4 WEIGHT_DECAY=0.05 SEED=42 \
bash scripts/run_eval.sh
```

For 25k/49k weights, set `FAMILY=25k` or `FAMILY=49k` to select the appropriate
input normalization. Each invocation runs one fit with the supplied
hyperparameters. See [downstream evaluation](USAGE.md#defaults-and-outputs)
for outputs and the report's hyperparameter values.

