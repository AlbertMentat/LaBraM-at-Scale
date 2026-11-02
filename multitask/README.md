# Shared EEG–Qwen3 classification

An average-pooling projector converts features from a frozen EEG encoder into 32 tokens for Qwen3. The projector and LoRA adapters train jointly across all 23 classification tasks, while the base language-model weights stay frozen. Molecule-H + Qwen3-8B reaches **50.50 ± 2.17%** mean balanced accuracy in the report.

## Setup and weights

From the repository root, install the [core environment](../docs/INSTALLATION.md), then:

```bash
python -m pip install -r multitask/requirements.txt
python -m pip check
python scripts/download_checkpoint.py --model molecule-base --output-dir /data/checkpoints
cd multitask
python train.py --help
```

Use Python 3.10 and CUDA. Large Qwen3 configurations use multiple GPUs and FSDP. Qwen3 weights are loaded at each YAML's pinned revision; set `llm.path` for a local snapshot.

## Configurations

[`configs/registry.json`](configs/registry.json) maps **66 configurations** to 11 EEG weight choices × six Qwen3 sizes (0.6B, 1.7B, 4B, 8B, 14B and 32B). Each entry records its HF file, SHA-256 and language-model revision.

| Directories | EEG weights |
|:---|:---|
| `quark_b`, `quark_l`, `quark_h`, `quark_g` | Default Quark weights |
| `atom_b`, `atom_l`, `atom_h` | Default Atom weights |
| `atom_l_reference` | Atom-L* from `best-full-ft/` |
| `molecule_b`, `molecule_l`, `molecule_h` | Default Molecule weights |

For `atom_l_reference`, download with `--model atom-large --variant best-full-ft --output-dir /data/checkpoints`. The configs load safetensors files directly and verify their SHA-256.

Set these paths in the YAML or pass dotted `--set key=value` overrides:

| Field | Purpose |
|:---|:---|
| `encoder.checkpoint` | Downloaded EEG weight file |
| `data.index_root` | Task indices and split manifests |
| `data.path_map` | Optional path-prefix relocation |
| `feature_cache.root` | Extracted feature cache |
| `feature_cache.local_dir` | Optional node-local cache copy |
| `output_root` | Training outputs |
| `llm.path` | Optional local Qwen3 snapshot |

Run the commands below from `multitask/`. Relative `tasks_config` and `encoder.repo_root` paths resolve against this directory. The `implementation` field selects the training backend.

## Prepare task inputs

```text
/data/omnibench/<output_id>/sample_index.json
/data/omnibench/<output_id>/splits/cross_subject_seed42/split_manifest.json
/data/omnibench/<output_id>/splits/cross_subject_seed10/split_manifest.json
/data/omnibench/<output_id>/splits/cross_subject_seed5/split_manifest.json
```

EEGDenoiseNet uses `within_subject_seed<seed>`. The [task registry](configs/tasks_omni23.yaml) defines task IDs, classes and prompts. Each sample index has an `entries` list with `path`, integer `label`, `subject_id`, and optional `ch_names` and `item_index`. Each pickle holds float32 `X` with shape `[channels, time]` at 200 Hz, `y`, and `ch_names`. A split manifest contains `seed` and `splits: {train: [...], val: [...], test: [...]}`, indexing that list.

Prepare training splits with at most **40 windows per subject–class pair**. This cap is shared by every downstream setting in the report, including FT and both LP readouts. Validation and test splits retain their full samples. The loader aligns channels and patch lengths and truncates HMC to 16 patches.

## Extract features

```bash
python extract.py --config configs/molecule_b/0p6b.yaml --check-only
python extract.py --config configs/molecule_b/0p6b.yaml --wait-merge
```

Each encoder processes the union of all three seeds' samples once and saves one cache per task. The six Qwen3 sizes share these features; normalization statistics are computed separately from each seed's training split. For parallel extraction, use `--shard i --num-shards N`, followed by `--merge --num-shards N`, or let shard zero merge with `--wait-merge`.

Cache keys hash the encoder, input settings, sample index, all configured split manifests, path mappings and task layout. Extraction and training check these hashes. Re-extract older caches without this metadata; use a new training output directory when the data change or an older run lacks the hash record. Raw EEG files must remain unchanged for a given index and path mapping.

## Train and evaluate

```bash
python train.py --config configs/molecule_b/0p6b.yaml --seed 42 --check-config
python train.py --config configs/molecule_b/0p6b.yaml --seed 42
python train.py --config configs/molecule_b/0p6b.yaml --seed 10
python train.py --config configs/molecule_b/0p6b.yaml --seed 5

torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/molecule_h/8b.yaml --seed 42
```

The recipe uses global batch 64, task sampling proportional to training-count^0.3 and per-task inverse class-frequency weights. Training runs for up to 16,000 steps, validates every 1,000 steps and stops after four evaluations without improvement. `train.joint` controls joint training; keep inherited LoRA settings in `train.stages` consistent with it because FSDP inserts LoRA before wrapping layers.

Mean validation balanced accuracy over the configured validation subsets selects one joint checkpoint per seed. The selected model is evaluated on each task's full validation and test splits using closed-set candidate-answer log-likelihood.

| Output | Contents |
|:---|:---|
| `run_contract.json` | Resolved configuration, seed and feature-cache data identities |
| `state/last.pt` | Trainable weights, full optimizer state, sampler state and per-rank PyTorch RNG |
| `state/best_joint.pt` | Selected projector and LoRA state |
| `result_llm.json` | Final task metrics and training curves |

Rerun the same command with the same number of processes to resume an incomplete run. Checkpoints restore optimizer, random-number and task/prompt sampler states; completed runs are skipped. Use separate output directories for different configurations and seeds. See [development notes](../docs/DEVELOPMENT.md) for checkpoint internals and tests.

```bash
python evaluate.py --run-dir /data/outputs/multitask/<run_name>__seed42
python summarize.py --results /data/outputs/multitask --output ./summary
```

Use `torchrun` for re-evaluating an FSDP configuration. Evaluation writes `reevaluation.json`; `--config` can relocate paths while preserving the recipe. The summarizer groups by encoder hash, LLM revision and recipe, deduplicates identical files and reports three-seed means and sample SD when all seeds are complete. `__macro__` is the equal-weight task average.
