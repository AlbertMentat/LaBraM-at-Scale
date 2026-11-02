#!/usr/bin/env python3
"""Signature-safe WDS writer and independent scanner for WDS 549 full builds."""
import hashlib
import io
import json
import os
import sqlite3
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
SAMPLES_PER_PATCH = 200


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_channel_signature(channel_names):
    value = "\x1f".join(channel_names).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:12]


def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def add_member(archive, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(content))


def atomic_write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return hashlib.sha256(content).hexdigest()


def bucket_key(window):
    metadata = window["metadata"]
    return (
        metadata["view"], int(metadata["n_channels"]),
        int(metadata["n_patches"]), metadata["channel_signature"])


class StreamingWDSWriter:
    def __init__(self, root, build_id, max_windows_per_tar=10000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.build_id = build_id
        self.max_windows_per_tar = int(max_windows_per_tar)
        if self.max_windows_per_tar <= 0:
            raise ValueError("max_windows_per_tar must be positive")
        self.spool_path = self.root / f".{build_id}.spool.sqlite3"
        for path in (
                self.spool_path,
                Path(str(self.spool_path) + "-journal"),
                Path(str(self.spool_path) + "-wal"),
                Path(str(self.spool_path) + "-shm")):
            path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.spool_path)
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("""
            CREATE TABLE windows (
                view TEXT NOT NULL,
                n_channels INTEGER NOT NULL,
                n_patches INTEGER NOT NULL,
                signature TEXT NOT NULL,
                sample_id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                mask BLOB NOT NULL,
                metadata BLOB NOT NULL
            ) WITHOUT ROWID
        """)
        self.connection.execute("""
            CREATE INDEX windows_bucket_order ON windows (
                view, n_channels, n_patches, signature, sample_id)
        """)
        self.window_count = 0
        self.finalized = False

    def add(self, window):
        if self.finalized:
            raise RuntimeError("cannot add windows after finalize")
        metadata = dict(window["metadata"])
        data = np.asarray(window["data"], dtype=np.float16)
        mask = np.asarray(window["padding_mask"], dtype=np.bool_)
        metadata.update({
            "payload_dtype": "float16",
            "payload_shape": list(data.shape),
            "padding_mask_dtype": "bool",
            "padding_mask_shape": list(mask.shape),
        })
        sample_id = metadata["sample_id"]
        try:
            self.connection.execute(
                "INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (metadata["view"], int(metadata["n_channels"]),
                 int(metadata["n_patches"]), metadata["channel_signature"],
                 sample_id, npy_bytes(data), npy_bytes(mask),
                 json.dumps(metadata, sort_keys=True).encode("utf-8")))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"duplicate sample ID: {sample_id}") from error
        self.window_count += 1
        if self.window_count % 256 == 0:
            self.connection.commit()

    def _finish_tar(self, archive, temporary, tar_path, bucket, windows):
        archive.close()
        os.replace(temporary, tar_path)
        view, n_channels, n_patches, signature = bucket
        return {
            "path": tar_path.relative_to(self.root).as_posix(),
            "sha256": sha256_file(tar_path),
            "size_bytes": tar_path.stat().st_size,
            "windows": windows,
            "view": view,
            "n_channels": n_channels,
            "n_patches": n_patches,
            "channel_signature": signature,
        }

    def finalize(self):
        if self.finalized:
            raise RuntimeError("writer already finalized")
        self.connection.commit()
        tar_records = []
        buckets = self.connection.execute("""
            SELECT DISTINCT view, n_channels, n_patches, signature
            FROM windows
            ORDER BY view, n_channels, n_patches, signature
        """).fetchall()
        for bucket in buckets:
            view, n_channels, n_patches, signature = bucket
            bucket_dir = (
                self.root / "views" / view
                / f"{n_channels}x{n_patches}-ch{signature}")
            bucket_dir.mkdir(parents=True, exist_ok=True)
            archive = None
            temporary = None
            tar_path = None
            shard_index = 0
            shard_windows = 0
            records = self.connection.execute("""
                SELECT sample_id, data, mask, metadata
                FROM windows
                WHERE view = ? AND n_channels = ? AND n_patches = ?
                    AND signature = ?
                ORDER BY sample_id
            """, bucket)
            for sample_id, data, mask, metadata in records:
                if archive is None:
                    filename = f"{self.build_id}-shard-{shard_index:05d}.tar"
                    tar_path = bucket_dir / filename
                    temporary = tar_path.with_name(
                        tar_path.name + f".tmp.{os.getpid()}")
                    archive = tarfile.open(temporary, "w")
                    shard_windows = 0
                add_member(archive, f"{sample_id}.npy", data)
                add_member(archive, f"{sample_id}.mask.npy", mask)
                add_member(archive, f"{sample_id}.json", metadata)
                shard_windows += 1
                if shard_windows == self.max_windows_per_tar:
                    tar_records.append(self._finish_tar(
                        archive, temporary, tar_path, bucket, shard_windows))
                    archive = None
                    shard_index += 1
            if archive is not None:
                tar_records.append(self._finish_tar(
                    archive, temporary, tar_path, bucket, shard_windows))
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "build_id": self.build_id,
            "windows": self.window_count,
            "tars": len(tar_records),
            "tar_files": tar_records,
        }
        catalog_path = self.root / "build_catalog.json"
        catalog_sha = atomic_write_json(catalog_path, catalog)
        self.finalized = True
        self.close(remove_spool=True)
        return catalog_path, catalog_sha, catalog

    def close(self, remove_spool=False):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if remove_spool:
            for path in (
                    self.spool_path,
                    Path(str(self.spool_path) + "-journal"),
                    Path(str(self.spool_path) + "-wal"),
                    Path(str(self.spool_path) + "-shm")):
                path.unlink(missing_ok=True)


def write_windows(windows, root, build_id, max_windows_per_tar=10000):
    writer = StreamingWDSWriter(
        root, build_id, max_windows_per_tar=max_windows_per_tar)
    try:
        for window in windows:
            writer.add(window)
        return writer.finalize()
    except Exception:
        writer.close()
        raise
def member_kind(name):
    if name.endswith(".mask.npy"):
        return name[:-9], "mask"
    if name.endswith(".npy"):
        return name[:-4], "data"
    if name.endswith(".json"):
        return name[:-5], "metadata"
    return None, None


def load_npy(archive, member):
    return np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False)


def scan_catalog(root, catalog_path, verify_hashes=True):
    root = Path(root)
    catalog = json.loads(Path(catalog_path).read_text())
    seen_samples = set()
    views = defaultdict(int)
    total_windows = 0
    total_padding_samples = 0
    actual_tars = {
        path.relative_to(root).as_posix()
        for path in (root / "views").rglob("*.tar")}
    referenced_tars = {record["path"] for record in catalog["tar_files"]}
    if actual_tars != referenced_tars:
        raise ValueError("catalog and actual tar sets differ")
    for record in catalog["tar_files"]:
        tar_path = root / record["path"]
        if tar_path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"tar size mismatch: {record['path']}")
        if verify_hashes and sha256_file(tar_path) != record["sha256"]:
            raise ValueError(f"tar SHA mismatch: {record['path']}")
        samples = defaultdict(dict)
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                key, kind = member_kind(member.name)
                if kind is None:
                    raise ValueError(f"unexpected tar member: {member.name}")
                if kind in samples[key]:
                    raise ValueError(f"duplicate {kind} member: {member.name}")
                samples[key][kind] = member
            if len(samples) != int(record["windows"]):
                raise ValueError(f"tar window count mismatch: {record['path']}")
            for key in sorted(samples):
                parts = samples[key]
                if set(parts) != {"data", "mask", "metadata"}:
                    raise ValueError(f"incomplete sample {key}: {sorted(parts)}")
                if key in seen_samples:
                    raise ValueError(f"duplicate sample across tars: {key}")
                seen_samples.add(key)
                data = load_npy(archive, parts["data"])
                mask = load_npy(archive, parts["mask"])
                metadata = json.loads(archive.extractfile(parts["metadata"]).read())
                if metadata["sample_id"] != key:
                    raise ValueError(f"sample ID mismatch: {key}")
                signature = ordered_channel_signature(metadata["channel_names"])
                if signature != metadata["channel_signature"]:
                    raise ValueError(f"metadata signature mismatch: {key}")
                if signature != record["channel_signature"]:
                    raise ValueError(f"tar signature mismatch: {key}")
                expected_shape = (
                    int(record["n_channels"]),
                    int(record["n_patches"]) * SAMPLES_PER_PATCH)
                if data.shape != expected_shape or data.dtype != np.float16:
                    raise ValueError(f"payload shape/dtype mismatch: {key}")
                if mask.shape != (expected_shape[1],) or mask.dtype != np.bool_:
                    raise ValueError(f"padding mask shape/dtype mismatch: {key}")
                if not np.isfinite(data).all():
                    raise ValueError(f"nonfinite payload: {key}")
                if data.size and float(np.max(np.abs(data))) > 15.01:
                    raise ValueError(f"payload exceeds clip contract: {key}")
                valid_samples = int(metadata["valid_samples"])
                expected_padding = expected_shape[1] - valid_samples
                if int(mask.sum()) != expected_padding:
                    raise ValueError(f"padding mask count mismatch: {key}")
                if mask[:valid_samples].any() or not mask[valid_samples:].all():
                    raise ValueError(f"padding mask layout mismatch: {key}")
                if expected_padding and np.any(data[:, valid_samples:] != 0):
                    raise ValueError(f"nonzero padding payload: {key}")
                if metadata["view"] != record["view"]:
                    raise ValueError(f"view mismatch: {key}")
                views[metadata["view"]] += 1
                total_windows += 1
                total_padding_samples += expected_padding
    if total_windows != int(catalog["windows"]):
        raise ValueError("catalog total window count mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "windows": total_windows,
        "tars": len(catalog["tar_files"]),
        "views": dict(sorted(views.items())),
        "padding_samples": total_padding_samples,
    }
