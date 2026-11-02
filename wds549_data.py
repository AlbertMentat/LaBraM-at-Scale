#!/usr/bin/env python3
"""WDS-549 data plumbing for LaBraM extension v2 tokenizer training.

Keeps the release-specific contract in one place: input scale, padding mask
semantics and per-patch validity statistics.
"""

import json
from pathlib import Path

import torch
from einops import rearrange

from eeg_data.wds_549_training_dataset import (
    SAMPLES_PER_PATCH,
    TRAIN_VIEWS,
    WDS549TrainingIterable,
)

# release stores dimensionless z-scores; unlike the legacy corpus it must NOT
# be divided by 100 (policy: new_tokenizer_input_scale = 1.0)
NEW_TOKENIZER_INPUT_SCALE = 1.0


def read_release_scale(policy_path):
    contract = json.loads(Path(policy_path).read_text())["signal_contract"]
    scale = float(contract["new_tokenizer_input_scale"])
    if contract.get("output_unit") != "dimensionless_zscore":
        raise ValueError("WDS 549 policy output_unit is not z-score")
    if scale != NEW_TOKENIZER_INPUT_SCALE:
        raise ValueError(f"unexpected tokenizer input scale: {scale}")
    return scale


def build_train_loader(args, num_replicas, rank):
    dataset = WDS549TrainingIterable(
        root=args.wds549_root,
        catalog_root=args.wds549_catalog_root,
        batch_size=args.batch_size,
        views=tuple(args.wds549_views),
        num_replicas=num_replicas,
        rank=rank,
        seed=args.seed,
        shuffle=True,
        drop_last=True,
        data_fraction=args.wds549_data_fraction,
        require_release_ready=not args.wds549_allow_unreleased,
        cache_dir=args.wds549_cache_dir,
        verbose=(rank == 0),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,               # the iterable already yields batches
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return dataset, loader


def prepare_batch(batch, device, scale=NEW_TOKENIZER_INPUT_SCALE):
    """Return flattened signal plus a per-patch keep mask.

    padding_mask is True on padded samples. patch_keep marks patches that
    contain valid samples; the tokenizer uses it for logging padding statistics.
    """
    eeg = batch["eeg"].to(device, non_blocking=True).float()
    if scale != 1.0:
        eeg = eeg * scale
    padding = batch["padding_mask"].to(device, non_blocking=True)

    batch_size, channels, patches, samples = eeg.shape
    if samples != SAMPLES_PER_PATCH:
        raise ValueError(f"unexpected patch length: {samples}")
    if padding.shape != (batch_size, patches * samples):
        raise ValueError(f"unexpected padding mask shape: {tuple(padding.shape)}")

    per_patch = rearrange(padding, "B (N T) -> B N T", T=samples)
    patch_keep = ~per_patch.all(dim=-1)
    signal = rearrange(eeg, "B C N T -> B C (N T)")
    return signal, patch_keep


__all__ = [
    "TRAIN_VIEWS",
    "NEW_TOKENIZER_INPUT_SCALE",
    "build_train_loader",
    "prepare_batch",
    "read_release_scale",
]
