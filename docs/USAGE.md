# Training and evaluation guide

Start with [Installation and first run](INSTALLATION.md). Run the commands below from the repository root. Paths under `/data/...` are examples to replace with the locations of your prepared data and downloaded checkpoints.

## Downstream evaluation

Each task uses this layout:

```text
<task>/
  train/*.pkl
  val/*.pkl
  test/*.pkl
  label_map.json
```

For the reported benchmark, prepare each training split with at most **40 windows per subject–class pair**, using the same capped splits for FT, both LP readouts and multitask evaluation. The launchers consume these prepared splits; they do not apply the cap. Validation and test splits retain their full samples. Use the [split snapshots](../data_manifests/README.md#downstream-split-snapshots) with their matching sample indices and original sample order.

Each `.pkl` contains `{"X": float32 [channels, time], "y": int, "ch_names": [...]}` at 200 Hz. `label_map.json` maps class names to the integers used in `y`, for example `{"class_0": 0, "class_1": 1}`.

### Download a checkpoint

The [Hugging Face repository](https://huggingface.co/Pan-Junyu/LaBraM-at-Scale/tree/main) contains ten default pretrained models and ten Full-FT-selected files under `best-full-ft/`. Download the matching model size and family:

```bash
python scripts/download_checkpoint.py --model molecule-base
# Optional: select the best evaluated Full-FT version instead.
python scripts/download_checkpoint.py --model atom-large --variant best-full-ft
```

The downloader reads `config.json` to select the requested model or tokenizer and checks its SHA-256 before saving it. Use `--repo-id` to select another HF repository and `--revision` to pin a commit. FT, LP and multitask entrypoints accept `.safetensors` directly. Released files contain the full pretrained model state; loaders select the encoder tensors. Fine-tuned task heads and EEG–Qwen3 adapters are separate.

| Model size | `MODEL` | File suffix |
|:---|:---|:---|
| Base | `labram_base_patch200_200` | `base.safetensors` |
| Large | `labram_large_patch200_200` | `large.safetensors` |
| Huge | `labram_huge_patch200_200` | `huge.safetensors` |
| Giant (Quark only) | `labram_giant_patch200_200` | `giant.safetensors` |

File paths follow `<family>/<family>-<size>.safetensors`, for example `atom/atom-large.safetensors` or `quark/quark-giant.safetensors`. The [model configuration](https://huggingface.co/Pan-Junyu/LaBraM-at-Scale/blob/main/config.json) records both weight collections, tokenizers, architectures, input normalization and SHA-256 checksums.

### Run one fit

Set `MODEL` to the matching architecture and supply the downloaded `.safetensors` file as `CHECKPOINT`. These three examples use the Molecule-B checkpoint downloaded above.

```bash
# Full fine-tuning of a Molecule Base encoder.
MODE=ft FAMILY=79k MODEL=labram_base_patch200_200 NPROC=1 \
CHECKPOINT=checkpoints/molecule-base.safetensors DATA_DIR=/data/tasks/TUAB \
OUTPUT_DIR=outputs/tuab_ft LR=3e-4 LAYER_DECAY=0.75 SEED=42 \
bash scripts/run_eval.sh

# Linear probing of the same encoder with pooled features.
MODE=lp_pool FAMILY=79k MODEL=labram_base_patch200_200 \
CHECKPOINT=checkpoints/molecule-base.safetensors DATA_DIR=/data/tasks/TUAB \
OUTPUT_DIR=outputs/tuab_lp_pool LR=5e-4 WEIGHT_DECAY=0.05 SEED=42 \
bash scripts/run_eval.sh

# The same encoder with the no-pool readout.
MODE=lp_nopool FAMILY=79k MODEL=labram_base_patch200_200 \
CHECKPOINT=checkpoints/molecule-base.safetensors DATA_DIR=/data/tasks/TUAB \
OUTPUT_DIR=outputs/tuab_lp_nopool LR=5e-4 WEIGHT_DECAY=0.05 SEED=42 \
bash scripts/run_eval.sh
```

### Input normalization

`FAMILY` selects the normalization used in pretraining:

| Family | Setting | Input normalization |
|:---|:---|:---|
| Quark | `FAMILY=25k` | Signal divided by 100; FT also sets `--skip_norm` |
| Atom | `FAMILY=49k` | Signal divided by 100; FT also sets `--skip_norm` |
| Molecule | `FAMILY=79k` | Per-channel z-score, clipped at ±15 |

### Defaults and outputs

| Setting | Full fine-tuning | Linear probing |
|:---|:---|:---|
| Epochs | 50, including 3 warm-up epochs | 30 |
| Learning rate | `3e-4` | `5e-4` |
| Weight decay | `0.05` | `0.05` |
| Batch size | 64 per process | 128 |
| Additional settings | Layer decay `0.75`; class-weighted loss | AdamW; gradient clipping at `0.8` |
| Validation | Every epoch | Every 5 epochs |
| Outputs | `epoch_records/`, `log.txt`, `results.json`, `checkpoint-best-analysis.pth` | `results.json`, `probe-best.pth` |

Both entrypoints report validation and test metrics at the epoch with the best validation balanced accuracy. The FT launcher saves that model and its selection metrics as `checkpoint-best-analysis.pth`. Periodic checkpoints retain the optimizer state for training resume. Resuming in the same output directory restores validation selection from completed epoch records before the resumed checkpoint. Use a separate `OUTPUT_DIR` for each fit. The FT launcher also accepts `BATCH_SIZE`, `UPDATE_FREQ`, `EPOCHS`, `WARMUP_EPOCHS`, `WEIGHT_DECAY` and `NPROC`.

LP saves the selected head with its encoder SHA-256, channel layout, normalization and label map. Evaluate it later using the same encoder file:

```bash
python evaluate_linear_probe.py \
  --probe outputs/tuab_lp_pool/probe-best.pth \
  --checkpoint checkpoints/molecule-base.safetensors \
  --data_dir /data/tasks/TUAB --split test --output outputs/tuab_lp_pool/reloaded_test.json
```

The evaluator checks the encoder identity and task label map. No-pool inputs must match the saved window length; their channel alignment uses the saved layout.

The report evaluates seeds **42, 10 and 5**. Its hyperparameter grids are:

- Full-FT: learning rate `{1e-4, 3e-4, 1e-3}` × layer decay `{0.75, 1.0}`.
- LP: learning rate `{1e-4, 5e-4, 1e-3}` × weight decay `{0, 0.05, 0.1}`.

Each launcher invocation performs one fit. The report uses seed42 test performance to select hyperparameters before reusing them for seeds 10 and 5; its main LP comparison selects the higher available pooled or no-pool three-seed mean for each model and task. See the [technical report](../site/assets/report.pdf) for checkpoint selection, task splits and the full evaluation protocol.

## Pretraining pipelines

The [recipe directory](../configs/recipes/) provides the encoder recipes below. Set data paths, tokenizer weights and distributed settings for your machines. `WORLD_SIZE` in these shell recipes denotes nodes; the total GPU count is processes per node × nodes. Keep that total and the per-GPU batch size to match the reported global batch. Learning rates are not automatically scaled.

| Encoder | Recipe | Total GPUs | Batch / GPU | Global batch | Epochs | LR | Warm-up | Gradient clip |
|:---|:---|---:|---:|---:|---:|:---|---:|:---|
| Quark-B | `25k_base.sh` | 32 | 64 | 2,048 | 50 | 5e-4 | 5 | 3.0 |
| Quark-L | `25k_large.sh` | 32 | 16 | 512 | 50 | 3e-4 | 10 | 3.0 |
| Quark-H | `25k_huge.sh` | 24 | 4 | 96 | 50 | 1.5e-4 | 6 | 1.0 |
| Quark-G | `25k_giant.sh` | 64 | 16 | 1,024 | 30 | 1e-4 | 8 | 0.5 |
| Atom-B | `49k_base.sh` | 8 | 128 | 1,024 | 50 | 1.5e-4 | 10 | 1.0 |
| Atom-L | `49k_large.sh` | 16 | 64 | 1,024 | 50 | 1.5e-4 | 10 | 1.0 |
| Atom-H | `49k_huge.sh` | 64 | 16 | 1,024 | 50 | 1.5e-4 | 10 | 1.0 |
| Molecule-B | `79k_base.sh` | 32 | 64 | 2,048 | 50 | 5e-4 | 5 | — |
| Molecule-L | `79k_large.sh` | 32 | 32 | 1,024 | 30 | 3e-4 | 5 | — |
| Molecule-H | `79k_huge.sh` | 32 | 16 | 512 | 50 | 1e-4 | 5 | — |

For example, on four 8-GPU nodes, run `NPROC=8 WORLD_SIZE=4 RANK=<0..3> MASTER_ADDR=<node-0> MASTER_PORT=29500 TOKENIZER_WEIGHT=<path> DATASET_CONFIGS=<path> bash configs/recipes/25k_base.sh` on each node. `25k_base_large.sh` remains a compatibility entrypoint and selects the Base or Large recipe from `MODEL`.

### Tokenizer weights

```bash
python scripts/download_checkpoint.py --tokenizer molecule
```

Use the downloaded `checkpoints/molecule-tokenizer.safetensors` as `TOKENIZER_WEIGHT` for the matching pretraining recipe. Replace `molecule` with `quark` or `atom` for those collections.

### Quark · 25k

Base, Large and Huge use the [25k implementation](../legacy_25k/), with its dataset configuration and tokenizer entrypoint. Giant uses the [25k FSDP implementation](../legacy_25k/giant/). The encoder dimensions match the shared `models/labram_pretrain.py` definitions. Paths supplied to `configs/recipes/25k_*.sh` through environment variables or CLI options resolve relative to the repository root. Paths containing spaces are supported; quote them in the shell.

- Data configuration: `legacy_25k/configs/dataset_configs_example.json`.
- Tokenizer: `legacy_25k/run_vqnsp_training.py`.
- EEG pretraining: `legacy_25k/run_labram_pretraining.py`.
- Giant: `python -m legacy_25k.giant.run_labram_pretraining_fsdp` from the repository root.
- Recipes: `configs/recipes/25k_*.sh`.

### Atom · 49k

Atom uses the shared top-level code with legacy WDS data (`--use_wds`). Write the windowed samples into tar shards with `tools/pack_webdataset.py`, then create the training index with `tools/build_wds_index.py`.

- Tokenizer: `run_vqnsp_training.py --use_wds`.
- EEG pretraining: `run_labram_pretraining.py` or `run_labram_pretraining_fsdp.py`.
- Recipes: `configs/recipes/49k_*.sh`.

### Molecule · 79k

Molecule uses WDS-549 with `--use_wds --wds_format wds-549 --input_scale 1.0`. The data pipeline builds and checks a release before the loader consumes it:

1. `tools/wds_549_full_core.py` and `tools/wds_549_full_io.py` prepare recordings and write signature-checked shards.
2. `tools/build_wds_549_global_catalogs.py` creates global catalogs.
3. `tools/audit_wds_549_build_states.py` checks build states.
4. `tools/finalize_wds_549_full_release.py` writes the release marker required by the loader.

Use an empty output directory to start `79k_large.sh`, or reuse a run's output directory to resume it automatically. To select a completed epoch for resume, use `OUTPUT_DIR=<existing-run> bash configs/recipes/79k_large_resume.sh`.

The writer and loader share `configs/wds_549_full_rebuild_policy_20260815.json`. Continue with the [installation guide's WDS-549 examples](INSTALLATION.md#launch-a-workflow), and use `configs/recipes/79k_*.sh` for the training settings of a particular configuration.

## Multitask EEG–Qwen3

The [multitask guide](../multitask/README.md) covers feature extraction, task indices and caches, joint projector/LoRA training, resume, saved-model evaluation and three-seed aggregation.

```bash
python -m pip install -r multitask/requirements.txt
python -m pip check
```

The directory provides 66 configurations across 11 EEG weight choices and six Qwen3 sizes. `configs/atom_l_reference/` uses the Full-FT-selected Atom-L reference; `configs/atom_l/` uses the default release. The registry maps each configuration to its HF file and checksum.

## Repository map

The code retains a flat module layout so imports from `models`, `eeg_data` and `tools` work from the repository root.

| Path | Purpose |
|:---|:---|
| `models/` | Base/Large/Huge/Giant encoders, pretraining wrapper, VQ-NSP tokenizer and EMA quantizer |
| `labram_utils.py`, `optim_factory.py`, `loss.py` | Checkpointing, optimization, schedules, logging and losses |
| `run_vqnsp_training.py` | Tokenizer training, including the legacy-WDS mode |
| `train_vqnsp_wds549.py`, `wds549_data.py` | WDS-549 tokenizer training |
| `run_labram_pretraining.py`, `run_labram_pretraining_fsdp.py` | DDP/FSDP EEG pretraining |
| `pretraining_continuation.py` | Continuation from completed epochs |
| `eeg_data/wds_dataset.py` | Legacy-WDS reader |
| `eeg_data/wds_549_training_dataset.py`, `eeg_data/wds_549_dataset.py` | WDS-549 readers |
| `eeg_data/wds_cache.py`, `eeg_data/wds_sharding.py` | Shared caching and shard assignment |
| `eeg_data/unified_dataset.py` | Downstream sample loading |
| `tools/` | Shard writing, indexing, catalogs, release checks and finalization |
| `configs/recipes/` | Tokenizer and EEG pretraining launch recipes |
| `scripts/` | Training launchers, completed-epoch checkpoint selector and downstream launcher |
| `run_unified_finetuning.py` | Full fine-tuning |
| `run_linear_probe.py`, `evaluate_linear_probe.py` | Train, save and evaluate pooled or no-pool linear probes |
| `downstream_state.py` | Selected-label counts and validation selection across FT resume |
| `legacy_25k/` | Quark training implementation |
| `multitask/` | Shared EEG–Qwen3 workflows |
| `site/` | Project page, report PDF and result tables |

Point the example data paths at your own copies of the prepared datasets. Dataset-specific preprocessing, dataset/split construction and baseline implementations are maintained separately. See the [repository inventory](../MANIFEST.md) for entrypoints and [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution.
