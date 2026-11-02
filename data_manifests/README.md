# Data inventories and split snapshots

## Pretraining sources

[`pretraining/datasets.csv`](pretraining/datasets.csv) lists the 523 datasets appearing across the three collections. Each row includes the dataset title, public source and paper URLs, and the hours, training windows and source identifiers used by Quark, Atom and Molecule. A family inclusion flag of `0` means that dataset is absent from that collection.

[`pretraining/references.csv`](pretraining/references.csv) contains the corresponding papers and dataset records. [`pretraining/summary.json`](pretraining/summary.json) summarizes the collection totals. These files use the inventory reported in the [technical report](../site/assets/report.pdf); they contain metadata only.

## Downstream split snapshots

[`splits/index.json`](splits/index.json) indexes 69 manifests: 23 tasks × seeds 42, 10 and 5. Each compressed JSON retains the original train, validation and test indices, class/subject counts, 40-window subject–class sampling settings and upstream protocol identifier. Training totals are 191,164, 191,859 and 191,879 task entries for seeds 42, 10 and 5, respectively. Validation and test indices retain the supplied full splits.

These manifests contain the split indices used in the multitask experiments. The same 40-window training cap applies to single-task FT and LP. Each split-index entry names its matching sample index under `sample_indices/`. Decompress that file, verify its SHA-256 against `sample_indices/index.json`, and preserve its row order when applying the split indices. The index records both compressed-file and decompressed-byte checksums.

To read a manifest:

```python
import gzip
import json

with gzip.open("data_manifests/splits/TUAB/seed42.json.gz", "rt") as handle:
    manifest = json.load(handle)
train_indices = manifest["splits"]["train"]
```

`index.json` records the compressed-file SHA-256, decompressed JSON SHA-256 and original manifest SHA-256. `SHA256SUMS` covers all files in this directory.

## Original sample ordering

The 23 compressed sample-index files contain signal locators in the order used by the split indices. Use path remapping to locate recordings on your machine while preserving that order. The hashes verify the provided sample-index files.
