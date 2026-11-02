"""Strict streaming training loader for WDS 549 full releases."""

import hashlib
import io
import json
import math
import os
import random
import tarfile
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from eeg_data.wds_cache import DEFAULT_MIN_FREE_BYTES, LocalTarCache
from eeg_data.wds_sharding import SharedEpoch, rotating_partition


TRAIN_VIEWS = ("main_train", "aux_low_channel", "aux_quality_flagged")
SAMPLES_PER_PATCH = 200
MAX_EFFECTIVE_WEIGHT = 2.0


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_channel_signature(channel_names):
    payload = "\x1f".join(channel_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def weighted_accept(sample_id, effective_weight, weight_ceiling, seed, epoch, sampling_round):
    if not 0 < effective_weight <= weight_ceiling:
        raise ValueError("effective weight exceeds sampler ceiling")
    if effective_weight == weight_ceiling:
        return True
    payload = f"{seed}:{epoch}:{sampling_round}:{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    draw = value / float(1 << 64)
    return draw < (effective_weight / weight_ceiling)


def fraction_selected_count(start_index, count, sample_step):
    if sample_step <= 1:
        return int(count)
    first = (-int(start_index)) % int(sample_step)
    if first >= int(count):
        return 0
    return 1 + (int(count) - 1 - first) // int(sample_step)


def _safe_release_path(root, relative):
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe WDS 549 release path: {relative}")
    resolved = (Path(root) / relative_path).resolve()
    resolved.relative_to(Path(root).resolve())
    return resolved


def _check_release_relative(relative):
    """String-only containment check. resolve() costs a filesystem round trip per
    call, which is fatal across ~290k catalog rows on blobfuse."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe WDS 549 release path: {relative}")
    return relative_path


def _read_jsonl(path, expected):
    # one bulk read: line iteration issues ~8 KiB reads, and on blobfuse every
    # read is a network round trip (~14k round trips for a 110 MiB catalog)
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload)
    rows = []
    total_bytes = 0
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        total_bytes += len(line)
        if not line.strip():
            raise ValueError(f"blank WDS 549 catalog line: {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"invalid WDS 549 catalog row: {path}:{line_number}")
        rows.append(row)
    actual = {"sha256": digest.hexdigest(), "bytes": total_bytes, "rows": len(rows)}
    for key, value in actual.items():
        if value != expected.get(key):
            raise ValueError(
                f"WDS 549 catalog {key} mismatch for {path}: "
                f"{value} != {expected.get(key)}")
    return rows


def _member_kind(name):
    if name.endswith(".mask.npy"):
        return name[:-9], "mask"
    if name.endswith(".npy"):
        return name[:-4], "data"
    if name.endswith(".json"):
        return name[:-5], "metadata"
    raise ValueError(f"unexpected WDS 549 tar member: {name}")


def validate_training_sample(data, padding_mask, metadata, row):
    n_channels = int(row["n_channels"])
    n_patches = int(row["n_patches"])
    total_samples = n_patches * SAMPLES_PER_PATCH
    if data.dtype != np.float16 or data.shape != (n_channels, total_samples):
        raise ValueError(f"WDS 549 payload mismatch: {data.dtype} {data.shape}")
    if padding_mask.dtype != np.bool_ or padding_mask.shape != (total_samples,):
        raise ValueError(
            f"WDS 549 padding mask mismatch: {padding_mask.dtype} "
            f"{padding_mask.shape}")
    if not np.isfinite(data).all():
        raise ValueError("WDS 549 payload contains NaN/Inf")

    required = {
        "schema_version", "sample_id", "dataset_id", "view", "channel_names",
        "channel_signature", "n_channels", "n_patches", "valid_samples",
        "padding_samples", "padding_fraction", "valid_patch_fractions",
        "weights", "unit", "normalization", "payload_dtype", "payload_shape",
        "padding_mask_dtype", "padding_mask_shape",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"WDS 549 metadata missing fields: {missing}")
    if metadata["schema_version"] != 1:
        raise ValueError("WDS 549 metadata schema mismatch")
    for key in ("view", "n_channels", "n_patches", "channel_signature"):
        if metadata[key] != row[key]:
            raise ValueError(f"WDS 549 metadata/catalog mismatch: {key}")
    if metadata["dataset_id"] != row["logical_source"]:
        raise ValueError("WDS 549 dataset identity mismatch")
    channel_names = metadata["channel_names"]
    if (not isinstance(channel_names, list)
            or len(channel_names) != n_channels
            or _ordered_channel_signature(channel_names) != row["channel_signature"]):
        raise ValueError("WDS 549 channel identity mismatch")
    if metadata["unit"] != "dimensionless_zscore":
        raise ValueError("WDS 549 unit mismatch")
    if metadata["normalization"] != "recording_channel_zscore":
        raise ValueError("WDS 549 normalization mismatch")

    valid_samples = int(metadata["valid_samples"])
    padding_samples = total_samples - valid_samples
    if not 1 <= valid_samples <= total_samples:
        raise ValueError("WDS 549 valid_samples out of range")
    if int(metadata["padding_samples"]) != padding_samples:
        raise ValueError("WDS 549 padding_samples mismatch")
    if int(padding_mask.sum()) != padding_samples:
        raise ValueError("WDS 549 mask sum mismatch")
    if padding_mask[:valid_samples].any() or not padding_mask[valid_samples:].all():
        raise ValueError("WDS 549 mask layout mismatch")
    if padding_samples and not np.all(data[:, valid_samples:] == 0):
        raise ValueError("WDS 549 padded payload must be zero")
    if not math.isclose(
            float(metadata["padding_fraction"]), padding_samples / total_samples,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("WDS 549 padding_fraction mismatch")
    fractions = metadata["valid_patch_fractions"]
    expected_fractions = []
    for patch in range(n_patches):
        valid = max(0, min(SAMPLES_PER_PATCH, valid_samples - patch * SAMPLES_PER_PATCH))
        expected_fractions.append(valid / SAMPLES_PER_PATCH)
    if (len(fractions) != n_patches
            or any(not math.isclose(float(a), b, rel_tol=0.0, abs_tol=1e-9)
                   for a, b in zip(fractions, expected_fractions))):
        raise ValueError("WDS 549 valid_patch_fractions mismatch")

    weights = metadata["weights"]
    weight_keys = {
        "source_weight", "channel_weight", "quality_weight",
        "provenance_weight", "effective_weight",
    }
    if not isinstance(weights, dict) or set(weights) != weight_keys:
        raise ValueError("WDS 549 weights contract mismatch")
    values = {key: float(weights[key]) for key in weight_keys}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("WDS 549 weights must be finite and nonnegative")
    product = (
        values["source_weight"] * values["channel_weight"]
        * values["quality_weight"] * values["provenance_weight"])
    if not math.isclose(values["effective_weight"], product, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("WDS 549 effective_weight mismatch")
    if metadata["view"] not in TRAIN_VIEWS or values["effective_weight"] <= 0:
        raise ValueError("WDS 549 non-training sample reached training loader")
    return values["effective_weight"]


class WDS549TrainingIterable(torch.utils.data.IterableDataset):
    def __init__(
            self, root, catalog_root, batch_size, views=TRAIN_VIEWS,
            num_replicas=1, rank=0, seed=42, shuffle=True,
            shuffle_buffer=2000, drop_last=True, data_fraction=1.0,
            require_release_ready=True, verify_tar_sha256=False,
            cache_dir=None, cache_max_bytes=None,
            cache_min_free_bytes=DEFAULT_MIN_FREE_BYTES, verbose=True):
        super().__init__()
        self.root = Path(root)
        self.catalog_root = Path(catalog_root)
        self.batch_size = int(batch_size)
        self.views = tuple(views)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = int(shuffle_buffer)
        self.drop_last = bool(drop_last)
        self.data_fraction = float(data_fraction)
        self.sample_step = (
            max(1, round(1.0 / self.data_fraction))
            if 0 < self.data_fraction < 1.0 else 1)
        self.verify_tar_sha256 = bool(verify_tar_sha256)
        self.local_cache = (
            LocalTarCache(
                cache_dir, max_bytes=cache_max_bytes,
                min_free_bytes=cache_min_free_bytes,
                verbose=verbose and rank == 0)
            if cache_dir else None)
        self.max_effective_weight = MAX_EFFECTIVE_WEIGHT
        self.verbose = bool(verbose)
        self._shared_epoch = SharedEpoch()
        if self.batch_size <= 0:
            raise ValueError("WDS 549 batch_size must be positive")
        if not math.isfinite(self.max_effective_weight) or self.max_effective_weight <= 0:
            raise ValueError("WDS 549 max_effective_weight must be positive")
        if not 0 < self.data_fraction <= 1.0:
            raise ValueError("WDS 549 data_fraction must be in (0, 1]")
        if not self.views or any(view not in TRAIN_VIEWS for view in self.views):
            raise ValueError(f"WDS 549 training views invalid: {self.views}")
        self._validate_release(require_release_ready)
        self.rows = self._load_catalog_rows_shared()
        if not self.rows:
            raise ValueError("WDS 549 selected training catalogs are empty")
        fraction_offsets = defaultdict(int)
        for row in sorted(
                self.rows,
                key=lambda item: (
                    item["logical_source"], item["view"],
                    item["release_relative_tar"])):
            fraction_key = (row["logical_source"], row["view"])
            row["_fraction_offset"] = fraction_offsets[fraction_key]
            fraction_offsets[fraction_key] += int(row["windows"])
        group_counts = defaultdict(int)
        for row in self.rows:
            selected_windows = fraction_selected_count(
                row["_fraction_offset"], row["windows"], self.sample_step)
            group_counts[(
                int(row["n_channels"]), int(row["n_patches"]),
                row["channel_signature"])] += selected_windows
        if self.drop_last:
            global_batches = sum(count // self.batch_size for count in group_counts.values())
        else:
            global_batches = sum(
                (count + self.batch_size - 1) // self.batch_size
                for count in group_counts.values())
        self._len = max(
            1, global_batches // self.num_replicas)
        if self.verbose and self.rank == 0:
            print(
                f"WDS549TrainingIterable: {len(self.rows)} tars, "
                f"{sum(int(row['windows']) for row in self.rows)} windows, "
                f"{len(group_counts)} shape groups, ~{self._len} batches/rank",
                flush=True)

    def _validate_release(self, required):
        marker_path = self.root / "RELEASE_READY.json"
        if not marker_path.is_file():
            if required:
                raise ValueError(f"missing WDS 549 release gate: {marker_path}")
            return
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "pass" or marker.get("is_release_ready") is not True:
            raise ValueError("invalid WDS 549 RELEASE_READY.json")

    def _load_catalog_rows_shared(self):
        """Load catalogs on rank 0 and broadcast; blobfuse degrades badly under
        concurrent cold reads of the same ~200 MiB catalogs from every rank."""
        dist_ready = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.num_replicas > 1)
        if not dist_ready:
            return self._load_catalog_rows()

        t0 = time.time()
        if self.rank == 0:
            print(f"WDS549 catalog: rank0 reading views={list(self.views)} ...",
                  flush=True)
            rows = self._load_catalog_rows()
            print(f"WDS549 catalog: rank0 read {len(rows)} rows in "
                  f"{time.time() - t0:.1f}s, broadcasting", flush=True)
            payload = [rows]
        else:
            payload = [None]
        torch.distributed.broadcast_object_list(payload, src=0)
        rows = payload[0]
        if self.rank == 0:
            print(f"WDS549 catalog: broadcast complete in "
                  f"{time.time() - t0:.1f}s total", flush=True)
        return rows

    def _load_catalog_rows(self):
        complete_path = self.catalog_root / "CATALOGS_COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        if (complete.get("status") != "pass"
                or complete.get("report_version") != "wds-549-phase5-global-catalogs-v4"):
            raise ValueError("invalid WDS 549 CATALOGS_COMPLETE.json")
        rows = []
        seen = set()
        for view in self.views:
            metadata = (complete.get("views") or {}).get(view)
            if not isinstance(metadata, dict):
                raise ValueError(f"missing WDS 549 view catalog: {view}")
            view_path = _safe_release_path(self.catalog_root, metadata["path"])
            for row in _read_jsonl(view_path, metadata):
                required = {
                    "build_id", "build_root", "logical_source", "path",
                    "release_relative_tar", "sha256", "size_bytes", "windows",
                    "view", "n_channels", "n_patches", "channel_signature",
                }
                if required - set(row):
                    raise ValueError("WDS 549 global catalog row is incomplete")
                if row["view"] != view:
                    raise ValueError("WDS 549 global catalog view mismatch")
                relative = row["release_relative_tar"]
                expected = str(Path(row["build_root"]) / row["build_id"] / row["path"])
                if relative != expected or relative in seen:
                    raise ValueError("WDS 549 global catalog path mismatch")
                _check_release_relative(relative)
                seen.add(relative)
                rows.append(row)
        return sorted(rows, key=lambda row: row["release_relative_tar"])

    def set_epoch(self, epoch):
        previous = self._shared_epoch.get()
        if (self.local_cache is not None and self.rank == 0
                and int(epoch) != previous):
            print(f"WDS cache after epoch {previous}: {self.local_cache.summary()}",
                  flush=True)
        self._shared_epoch.set(epoch)

    def __len__(self):
        return self._len

    def _shard_rows(self):
        epoch = self._shared_epoch.get()
        rows = list(self.rows)
        random.Random(self.seed).shuffle(rows)
        rows = rotating_partition(
            rows, self.num_replicas, self.rank, epoch)
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        if worker is not None:
            rows = rotating_partition(
                rows, worker.num_workers, worker.id, epoch)
        if self.shuffle:
            random.Random(
                self.seed + epoch * 100003 + self.rank * 131 + worker_id
            ).shuffle(rows)
        return rows

    def _iter_tar_samples(self, row):
        tar_path = self.root / _check_release_relative(row["release_relative_tar"])
        if self.local_cache is None:
            if tar_path.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"WDS 549 tar size mismatch: {tar_path}")
            if self.verify_tar_sha256 and _sha256_file(tar_path) != row["sha256"]:
                raise ValueError(f"WDS 549 tar SHA mismatch: {tar_path}")
            cache_context = nullcontext(tar_path)
        else:
            cache_context = self.local_cache.open(
                tar_path, row["release_relative_tar"],
                int(row["size_bytes"]), row["sha256"])
        pending = defaultdict(dict)
        sample_decisions = {}
        next_sample_index = 0
        with cache_context as open_path, tarfile.open(open_path) as archive:
            for member in archive:
                sample_id, kind = _member_kind(member.name)
                if self.sample_step > 1:
                    take = sample_decisions.get(sample_id)
                    if take is None:
                        global_sample_index = (
                            int(row.get("_fraction_offset", 0)) + next_sample_index)
                        take = global_sample_index % self.sample_step == 0
                        sample_decisions[sample_id] = take
                        next_sample_index += 1
                    if not take:
                        continue
                raw = archive.extractfile(member).read()
                if kind == "metadata":
                    value = json.loads(raw)
                else:
                    value = np.load(io.BytesIO(raw), allow_pickle=False)
                pending[sample_id][kind] = value
                if set(pending[sample_id]) == {"data", "mask", "metadata"}:
                    parts = pending.pop(sample_id)
                    if parts["metadata"].get("sample_id") != sample_id:
                        raise ValueError("WDS 549 tar sample identity mismatch")
                    weight = validate_training_sample(
                        parts["data"], parts["mask"], parts["metadata"], row)
                    yield parts["data"], parts["mask"], parts["metadata"], weight
        if pending:
            raise ValueError(f"incomplete WDS 549 tar triples: {tar_path}")

    def _iter_samples(self, rows, sampling_round):
        epoch = self._shared_epoch.get()
        rng = random.Random(
            self.seed + epoch * 7919 + self.rank + sampling_round * 104729)
        buffer = []
        for row in rows:
            for sample in self._iter_tar_samples(row):
                metadata = sample[2]
                effective_weight = sample[3]
                if not weighted_accept(
                        metadata["sample_id"], effective_weight,
                        self.max_effective_weight, self.seed, epoch,
                        sampling_round):
                    continue
                if self.shuffle and self.shuffle_buffer > 1:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        index = rng.randrange(len(buffer))
                        buffer[index], buffer[-1] = buffer[-1], buffer[index]
                        yield buffer.pop()
                else:
                    yield sample
        if self.shuffle:
            rng.shuffle(buffer)
        yield from buffer

    @staticmethod
    def _collate(samples):
        data, masks, metadata, weights = zip(*samples)
        return {
            "eeg": torch.stack([
                torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(
                    value.shape[0], -1, SAMPLES_PER_PATCH)
                for value in data]),
            "padding_mask": torch.stack([
                torch.from_numpy(np.asarray(mask, dtype=np.bool_)) for mask in masks]),
            "sample_weight": torch.ones(len(samples), dtype=torch.float32),
            "effective_weight": torch.tensor(weights, dtype=torch.float32),
            "meta": {
                "ch_names": [item["channel_names"] for item in metadata],
                "sample_id": [item["sample_id"] for item in metadata],
                "dataset": [item["dataset_id"] for item in metadata],
                "view": [item["view"] for item in metadata],
            },
        }

    def _gen_batches(self, rows, sampling_round, pending):
        for sample in self._iter_samples(rows, sampling_round):
            metadata = sample[2]
            key = (
                int(metadata["n_channels"]), int(metadata["n_patches"]),
                metadata["channel_signature"], tuple(metadata["channel_names"]))
            pending[key].append(sample)
            if len(pending[key]) == self.batch_size:
                yield self._collate(pending.pop(key))
        if not self.drop_last:
            for samples in pending.values():
                if samples:
                    yield self._collate(samples)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_count = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        target = self._len // worker_count + (worker_id < self._len % worker_count)
        produced = 0
        attempts = 0
        pending = defaultdict(list)
        while produced < target:
            rows = self._shard_rows()
            if not rows:
                return
            for batch in self._gen_batches(rows, attempts, pending):
                yield batch
                produced += 1
                if produced >= target:
                    break
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("WDS 549 batch equalization did not converge")
