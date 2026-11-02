# Repository contents

| Directory or file | Contents |
|:---|:---|
| `models/` | EEG encoders, pretraining model, VQ-NSP tokenizer and quantizer |
| `eeg_data/`, `tools/` | Data loading, shard preparation, indexing and release checks |
| `run_*`, `train_vqnsp_wds549.py`, `evaluate_linear_probe.py` | Tokenizer training, EEG pretraining, FT and LP entrypoints |
| `downstream_state.py`, `tests/` | Downstream resume/label helpers and regression checks |
| `configs/recipes/`, `scripts/` | Training recipes, evaluation launcher and weight downloader |
| `legacy_25k/` | Quark data and training implementation |
| `multitask/` | Shared EEG–Qwen3 training, evaluation and 66 configurations |
| `site/` | Static project page, technical report and complete benchmark tables |
| `docs/` | Installation, usage and implementation guides |
| `data_manifests/` | Pretraining source inventory, downstream split snapshots and sample indices |

Released EEG weights, architectures, file hashes and per-tensor metadata are maintained in the [Hugging Face manifest](https://huggingface.co/Pan-Junyu/LaBraM-at-Scale/blob/main/manifest.json). Model selection and experimental results are described in the [technical report](site/assets/report.pdf).
