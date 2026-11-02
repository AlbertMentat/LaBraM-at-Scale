# Third-party notices

## LaBraM

The EEG model, VQ-NSP tokenizer, quantizer and supporting training utilities build on [LaBraM](https://github.com/935963004/LaBraM) by Wei-Bang Jiang, Li-Ming Zhao and Bao-Liang Lu. The upstream implementation is distributed under the MIT License, copyright © 2024 Weibang Jiang. The complete notice is retained in [third_party/LaBraM-LICENSE.txt](third_party/LaBraM-LICENSE.txt).

> Jiang, W.-B., Zhao, L.-M., and Lu, B.-L. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. ICLR 2024. [Paper](https://openreview.net/forum?id=QzTpTRVtrP).

## Loss utilities and model references

`loss.py` and `legacy_25k/loss.py` retain the Salesforce copyright and BSD-3-Clause identifier from the upstream utilities. The full terms are included in [third_party/Salesforce-BSD-3-Clause.txt](third_party/Salesforce-BSD-3-Clause.txt), with the notice sourced from [Salesforce BLIP](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt).

The LaBraM model-file headers acknowledge [BEiT-v2](https://github.com/microsoft/unilm/tree/master/beitv2), [timm](https://github.com/huggingface/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit) and [DINO](https://github.com/facebookresearch/dino); those acknowledgements remain in the source files. The optional task-aware query selection module cites [UniMind](https://arxiv.org/abs/2506.18962).

## Icons

The GitHub mark uses Primer Octicons under the MIT License; see [the icon license](docs/assets/links/LICENSE-Octicons.txt). The Hugging Face mark is the official brand asset identified in the icon notes. Asset sources are listed in [the icon notes](docs/assets/links/README.md).

## External models, software and datasets

Qwen3 weights are downloaded separately from their original Hugging Face repositories at the revisions specified by the multitask configurations. Their licenses apply independently. Python dependencies retain their respective licenses. Datasets are not bundled; obtain each dataset from its original source and follow its terms.

Original code and released EEG encoder and tokenizer weights are licensed under [MIT](LICENSE). Third-party components retain the licenses listed above.
