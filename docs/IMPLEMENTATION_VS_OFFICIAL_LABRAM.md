# Extending LaBraM for large-scale EEG pretraining

The extension keeps LaBraM's masked neural-code prediction objective and Base/Large/Huge architecture definitions. It adds a Giant encoder and the data and training support needed for larger heterogeneous EEG collections.

## Encoder scale

| Size | Width | Transformer blocks | Attention heads | Encoder parameters |
|:---|---:|---:|---:|---:|
| Base | 200 | 12 | 10 | 5.8M |
| Large | 400 | 24 | 16 | 46.3M |
| Huge | 800 | 48 | 16 | 369M |
| Giant | 1600 | 64 | 20 | 1.97B |

Definitions are in [`models/labram_pretrain.py`](../models/labram_pretrain.py) and [`models/labram.py`](../models/labram.py). Encoder counts exclude the masked prediction and projection heads in the released pretrained state.

## Heterogeneous data

Different recordings have different electrode layouts and durations. Channel-dependent windows limit the sequence length as channel counts grow. The data loaders group compatible layouts, align channel identities and distinguish padding from valid signal. `eeg_data/` contains the legacy and WDS-549 readers; `tools/` contains the shard writer, indexing, catalog and release checks.

Quark and Atom use signal scaling by 1/100. Molecule uses per-channel z-score normalization clipped at ±15 and a revised source-selection and signal-preparation pipeline. The collection-specific recipes and normalization must be used together. The report lists the pretraining datasets and processing settings.

## Distributed training

The shared DDP entrypoint is `run_labram_pretraining.py`. `run_labram_pretraining_fsdp.py` adds fully sharded training for larger encoders. Streaming shards, local caching, mixed precision and gradient checkpointing reduce data and memory costs. FSDP checkpoints are gathered for subsequent loading by the downstream scripts.

Both pretraining entrypoints support periodic checkpointing and resume. To resume within an epoch, keep the batch configuration and data order unchanged. When changing the batch configuration, select a completed epoch with `scripts/select_completed_epoch_checkpoint.py` and use the continuation options.

## Downstream use

`run_unified_finetuning.py` updates the encoder and classifier. `run_linear_probe.py` freezes the encoder and supports pooled or no-pool features. Both accept released `.safetensors` files directly, as well as compatible PyTorch checkpoints. Their model definitions, channel mapping and input normalization are shared with the multitask encoder loader.

### ISRUC_S2 channel alignment

ISRUC_S2 uses the channel-suffix alignment from the reported experiments. Some channel names map to zero-filled or unknown-channel slots. A different mapping requires fresh feature extraction and evaluation.

Quark training code is in `legacy_25k/`. See [Usage](USAGE.md) for commands and file formats, and the [technical report](../site/assets/report.pdf) for evaluation settings and results.
