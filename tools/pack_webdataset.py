#!/usr/bin/env python3
"""把 tok256 重切池打包成 WebDataset (.tar 分片), 本地中转加速训练IO.

本地中转版 (解决 blobfuse 百万小文件直读瓶颈):
  按集分片(i%n_shards) -> 每集 stage-in 本地 -> 本地读pkl按shape分桶
  -> 每shape写tar到本地 -> push tar回blob -> 写本集index
大集不拆: 输出是大tar文件(写快), 瓶颈仅 stage-in 读(~130文件/s),
          TUH_main 百万pkl ~2-3h, 单集即可, 无需像reslice那样按subject拆.

通道归一化: resolve_channels -> standard_1020, 存 filtered ch_names + 过滤后X(float16省一半).
按 (n_ch_filtered, n_patches) shape分桶, 桶内集内混洗(跨集混洗由训练 GroupedBatchSampler 补偿).

输出: {out_root}/{ch}x{patch}/{ds}-shard-{NNNNN}.tar   (兼容 WebDatasetEEG 扫 {root}/{bucket}/*.tar)
     + {out_root}/_idx_{ds}.json                        (本集分桶统计)
分片内每样本: {key}.npy (归一化X float16) + {key}.json (ch_names+meta).

分片: --shard_id N --n_shards M (按集分片, amlt并行). WebDataset = 标准tarfile(无需webdataset库).
"""
import argparse, ast, glob, hashlib, io, json, math, os, pickle, random, shutil, subprocess, tarfile, time, zlib
from collections import Counter, defaultdict
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 直接按文件路径加载 ch_name_mappings, 绕过 eeg_data/__init__.py(它会 import torch,
# CPU image 无 torch 会崩). ch_name_mappings 只依赖 re, 纯字符串处理.
import importlib.util as _ilu
_cnm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'eeg_data', 'ch_name_mappings.py')
_spec = _ilu.spec_from_file_location('ch_name_mappings', _cnm_path)
_cnm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cnm)
resolve_channels = _cnm.resolve_channels


def get_std1020():
    labram_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'models', 'labram.py')
    for node in ast.parse(open(labram_py).read()).body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if getattr(tg, 'id', '') == 'standard_1020':
                    return ast.literal_eval(node.value)
    raise RuntimeError("standard_1020 not found")


STD_1020 = get_std1020()
STD_SET = set(c.upper() for c in STD_1020)
PACK_SCHEMA_VERSION = 2
PROCESSING_VERSION = "pack-webdataset-v2"


def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def load_manifest(ds_dir):
    """读 manifest.json; 无则合并 manifest_sub*.json (大集按subject拆分的分片)."""
    mp = os.path.join(ds_dir, 'manifest.json')
    if os.path.exists(mp):
        try:
            return json.load(open(mp))
        except Exception:
            return None
    subs = sorted(glob.glob(os.path.join(ds_dir, 'manifest_sub*.json')))
    if subs:
        merged = []
        for s in subs:
            try:
                merged.extend(json.load(open(s)))
            except Exception:
                pass
        return merged or None
    return None


def stage_in(ds, in_root, local_in, sj_shards=1, sj_id=0):
    """blobfuse->本地cp. 分批cp(xargs -n200)+多轮重试补齐(抗blobfuse ENOTCONN)+subject分片(crc32, 多job并行打超大集)+校验n vs total."""
    blob_in = os.path.join(in_root, ds)
    shutil.rmtree(local_in, ignore_errors=True)
    os.makedirs(local_in, exist_ok=True)
    t0 = time.time()
    flist = os.path.join("/tmp", "_flist_%s_%d.txt" % (ds, sj_id))
    _run('find "%s" -maxdepth 1 -name "*.pkl" > "%s" 2>/dev/null' % (blob_in, flist))
    allf = [ln.strip() for ln in open(flist) if ln.strip()]
    if sj_shards > 1:
        allf = [p for p in allf if zlib.crc32(os.path.basename(p).split("__")[0].encode()) % sj_shards == sj_id]
    with open(flist, "w") as fo:
        for p in allf:
            fo.write(p + chr(10))
    total = len(allf)
    n = 0
    for rnd in range(6):
        _run('xargs -r -a "%s" -P 24 -n 200 cp -t "%s/" 2>/dev/null' % (flist, local_in))
        n = len([fn for fn in os.listdir(local_in) if fn.endswith(".pkl")])
        if total and n >= total:
            break
        time.sleep(3 * (rnd + 1))
    try:
        os.remove(flist)
    except Exception:
        pass
    tag = "" if (not total or n >= total) else " WARN_short"
    print("    [stage-in] %s[sj%d/%d]: %d/%d pkl 下载 %ds%s" % (ds, sj_id, sj_shards, n, total, time.time() - t0, tag), flush=True)
    return n


def push_out(ds, local_out, out_root, sj_sfx=""):
    """本地tar->blob: 各shape桶的tar cp到 {out_root}/{ch}x{patch}/ (tar是大文件, cp快)."""
    t0 = time.time()
    n_tar = 0
    for bucket in os.listdir(local_out):
        bdir = os.path.join(local_out, bucket)
        if not os.path.isdir(bdir):
            continue
        blob_bucket = os.path.join(out_root, bucket)
        os.makedirs(blob_bucket, exist_ok=True)
        _run(f'find "{bdir}" -maxdepth 1 -name "*.tar" -print0 | '
             f'xargs -0 -r -P 8 cp -t "{blob_bucket}/" 2>/dev/null')
        n_tar += len(glob.glob(os.path.join(bdir, '*.tar')))
    idx_local = os.path.join(local_out, 'wds_index.json')
    if os.path.exists(idx_local):
        _run(f'cp "{idx_local}" "{os.path.join(out_root, "_idx_" + ds + sj_sfx + ".json")}" 2>/dev/null')
    print(f"    [push] {ds}: {n_tar}tar 写回 {time.time()-t0:.0f}s", flush=True)


def pack_dataset_local(ds, mani, local_in, local_out, shard_size,
                       max_patches, seed, sj_shards=1, sj_id=0,
                       qc_options=None):
    """Validate, normalize, and stream one dataset into signature-safe tar files."""
    options = {
        "source_dataset": ds,
        "min_channels": 1,
        "flatline_std": 1e-8,
        "max_abs": None,
        "max_channel_mean_abs": None,
        "channel_std_min": None,
        "channel_std_max": None,
        "require_provenance": False,
        "require_normalization": False,
        "hash_tars": False,
    }
    if qc_options:
        options.update(qc_options)

    rng = random.Random(seed)
    selected = []
    for item in mani:
        if sj_shards > 1:
            subject = os.path.basename(item.get("file_path", "")).split("__")[0]
            if zlib.crc32(subject.encode()) % sj_shards != sj_id:
                continue
        selected.append(item)
    rng.shuffle(selected)

    rejected = Counter()
    warnings = Counter()
    accepted_by_signature = Counter()
    qc = {
        "schema_version": PACK_SCHEMA_VERSION,
        "processing_version": PROCESSING_VERSION,
        "dataset": ds,
        "source_dataset": options["source_dataset"],
        "subject_shard_id": sj_id,
        "subject_shards": sj_shards,
        "input_manifest_entries": len(selected),
        "accepted_windows": 0,
        "rejected_windows": 0,
        "mapped_channels_dropped": 0,
        "flatline_channels_removed": 0,
        "finite_values_checked": 0,
        "numeric_ranges": {
            "absmax_min": None, "absmax_max": None,
            "channel_mean_abs_max": None,
            "channel_std_min": None, "channel_std_max": None,
        },
        "options": {
            key: value for key, value in options.items()
            if key != "source_dataset"
        },
    }
    resolve_cache = {}
    writers = {}
    suffix = ("-sj%d" % sj_id) if sj_shards > 1 else ""

    def reject(reason):
        rejected[reason] += 1
        qc["rejected_windows"] += 1

    def get_value(payload, item, key):
        if payload.get(key) is not None:
            return payload[key]
        if item.get(key) is not None:
            return item[key]
        provenance = payload.get("provenance") or item.get("provenance")
        if isinstance(provenance, dict):
            return provenance.get(key)
        return None

    def update_range(key, value, use_min=False):
        current = qc["numeric_ranges"][key]
        if current is None:
            qc["numeric_ranges"][key] = float(value)
        elif use_min:
            qc["numeric_ranges"][key] = min(current, float(value))
        else:
            qc["numeric_ranges"][key] = max(current, float(value))

    def writer_for(channels, patch_count):
        channel_tuple = tuple(channels)
        signature_id = hashlib.sha256(
            "\x1f".join(channel_tuple).encode("utf-8")).hexdigest()[:12]
        key = (channel_tuple, patch_count)
        state = writers.get(key)
        if state is None:
            state = {
                "channels": channel_tuple,
                "signature_id": signature_id,
                "patch_count": patch_count,
                "tar": None,
                "shard_idx": 0,
                "in_shard": 0,
                "n_written": 0,
                "tar_path": None,
                "tar_relative_path": None,
                "tar_files": [],
            }
            writers[key] = state
        return state

    def open_tar(state):
        if state["tar"] is not None:
            return
        bucket = "%dx%d" % (len(state["channels"]), state["patch_count"])
        bucket_dir = os.path.join(local_out, bucket)
        os.makedirs(bucket_dir, exist_ok=True)
        tar_name = "%s%s-ch%s-shard-%05d.tar" % (
            ds, suffix, state["signature_id"], state["shard_idx"])
        state["tar_path"] = os.path.join(bucket_dir, tar_name)
        state["tar_relative_path"] = os.path.join(bucket, tar_name)
        state["tar"] = tarfile.open(state["tar_path"], "w")
        state["in_shard"] = 0

    def close_tar(state):
        if state["tar"] is None:
            return
        state["tar"].close()
        record = {
            "path": state["tar_relative_path"],
            "size_bytes": os.path.getsize(state["tar_path"]),
            "n_windows": state["in_shard"],
        }
        if options["hash_tars"]:
            digest = hashlib.sha256()
            with open(state["tar_path"], "rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            record["sha256"] = digest.hexdigest()
        state["tar_files"].append(record)
        state["tar"] = None
        state["tar_path"] = None
        state["tar_relative_path"] = None
        state["shard_idx"] += 1

    for item in selected:
        file_path = item.get("file_path")
        if not file_path:
            reject("manifest_missing_file_path")
            continue
        source_path = os.path.join(local_in, os.path.basename(file_path))
        try:
            with open(source_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            reject("load_error")
            continue
        if not isinstance(payload, dict) or "X" not in payload:
            reject("invalid_payload")
            continue
        try:
            signal = np.asarray(payload["X"], dtype=np.float32)
        except Exception:
            reject("invalid_signal")
            continue
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
            reject("invalid_signal_shape")
            continue

        manifest_channels = item.get("ch_names")
        payload_channels = payload.get("ch_names")
        if manifest_channels is None or payload_channels is None:
            reject("missing_ch_names")
            continue
        manifest_channels = [str(value).strip() for value in manifest_channels]
        payload_channels = [str(value).strip() for value in payload_channels]
        if manifest_channels != payload_channels:
            reject("manifest_channel_mismatch")
            continue
        if len(payload_channels) != signal.shape[0] or not all(payload_channels):
            reject("channel_count_mismatch")
            continue
        try:
            manifest_channel_count = int(item["n_channels"])
            manifest_patch_count = int(item["n_patches"])
        except (KeyError, TypeError, ValueError):
            reject("invalid_manifest_shape")
            continue
        if manifest_channel_count != signal.shape[0]:
            reject("manifest_channel_count_mismatch")
            continue
        if signal.shape[1] % 200 != 0:
            reject("signal_not_patch_aligned")
            continue
        actual_patch_count = signal.shape[1] // 200
        if manifest_patch_count != actual_patch_count:
            reject("manifest_patch_count_mismatch")
            continue
        patch_count = actual_patch_count
        if max_patches:
            patch_count = min(patch_count, max_patches)
        if patch_count <= 0:
            reject("zero_patches")
            continue

        cache_key = (tuple(payload_channels), signal.shape[0])
        if cache_key not in resolve_cache:
            resolve_cache[cache_key] = resolve_channels(
                payload_channels, STD_SET, signal.shape[0],
                dataset_id=options["source_dataset"])
        keep_indices, mapped_channels = resolve_cache[cache_key]
        if not mapped_channels:
            reject("no_mapped_channels")
            continue
        if (len(keep_indices) != len(mapped_channels)
                or any(index < 0 or index >= signal.shape[0]
                       for index in keep_indices)):
            reject("invalid_channel_mapping")
            continue
        qc["mapped_channels_dropped"] += signal.shape[0] - len(keep_indices)
        signal = np.ascontiguousarray(
            signal[keep_indices, :patch_count * 200], dtype=np.float32)
        mapped_channels = list(mapped_channels)

        finite_mask = np.isfinite(signal)
        qc["finite_values_checked"] += int(signal.size)
        if not finite_mask.all():
            reject("nonfinite")
            continue

        channel_std = signal.astype(np.float64, copy=False).std(axis=1)
        flatline_std = options["flatline_std"]
        if flatline_std is not None and flatline_std >= 0:
            keep_channel_mask = channel_std > flatline_std
            removed = int((~keep_channel_mask).sum())
            if removed:
                qc["flatline_channels_removed"] += removed
                signal = signal[keep_channel_mask]
                mapped_channels = [
                    name for name, keep in zip(mapped_channels, keep_channel_mask)
                    if keep]
                channel_std = channel_std[keep_channel_mask]
        if signal.shape[0] < int(options["min_channels"]):
            reject("too_few_channels")
            continue

        channel_mean_abs = np.abs(
            signal.astype(np.float64, copy=False).mean(axis=1))
        absmax = float(np.abs(signal).max())
        mean_abs_max = float(channel_mean_abs.max())
        std_min = float(channel_std.min())
        std_max = float(channel_std.max())
        if options["max_abs"] is not None and absmax > options["max_abs"]:
            reject("absmax_exceeded")
            continue
        if (options["max_channel_mean_abs"] is not None
                and mean_abs_max > options["max_channel_mean_abs"]):
            reject("channel_mean_exceeded")
            continue
        if (options["channel_std_min"] is not None
                and std_min < options["channel_std_min"]):
            reject("channel_std_below_min")
            continue
        if (options["channel_std_max"] is not None
                and std_max > options["channel_std_max"]):
            reject("channel_std_above_max")
            continue

        provenance = payload.get("provenance") or item.get("provenance")
        normalization = get_value(payload, item, "normalization")
        if options["require_provenance"] and not isinstance(provenance, dict):
            reject("missing_provenance")
            continue
        if options["require_normalization"] and normalization is None:
            reject("missing_normalization")
            continue
        if provenance is None:
            warnings["missing_provenance"] += 1
        if normalization is None:
            warnings["missing_normalization"] += 1

        state = writer_for(mapped_channels, patch_count)
        open_tar(state)
        key_name = "%s_%s_%08d" % (
            ds, os.path.splitext(os.path.basename(file_path))[0],
            state["n_written"])
        output_signal = np.ascontiguousarray(signal, dtype=np.float16)
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, output_signal)
        metadata = {
            "schema_version": PACK_SCHEMA_VERSION,
            "processing_version": PROCESSING_VERSION,
            "ch_names": mapped_channels,
            "n_channels": len(mapped_channels),
            "n_patches": patch_count,
            "dataset": ds,
            "source_dataset": options["source_dataset"],
            "channel_signature": state["signature_id"],
            "source_file": file_path,
        }
        for key in ("recording_id", "source_files", "unit"):
            value = get_value(payload, item, key)
            if value is not None:
                metadata[key] = value
        if normalization is not None:
            metadata["normalization"] = normalization
        if provenance is not None:
            metadata["provenance"] = provenance
        metadata_bytes = json.dumps(metadata, default=str).encode("utf-8")
        for extension, data in (("npy", npy_buffer.getvalue()),
                                ("json", metadata_bytes)):
            tar_info = tarfile.TarInfo("%s.%s" % (key_name, extension))
            tar_info.size = len(data)
            state["tar"].addfile(tar_info, io.BytesIO(data))
        state["n_written"] += 1
        state["in_shard"] += 1
        qc["accepted_windows"] += 1
        accepted_by_signature[state["signature_id"]] += 1
        update_range("absmax_min", absmax, use_min=True)
        update_range("absmax_max", absmax)
        update_range("channel_mean_abs_max", mean_abs_max)
        update_range("channel_std_min", std_min, use_min=True)
        update_range("channel_std_max", std_max)
        if state["in_shard"] >= shard_size:
            close_tar(state)

    index = {}
    for state in writers.values():
        close_tar(state)
        if state["n_written"] == 0:
            continue
        channel_count = len(state["channels"])
        index_key = "%dx%d:%s" % (
            channel_count, state["patch_count"], state["signature_id"])
        index[index_key] = {
            "schema_version": PACK_SCHEMA_VERSION,
            "processing_version": PROCESSING_VERSION,
            "n_windows": state["n_written"],
            "n_shards": state["shard_idx"],
            "n_channels": channel_count,
            "n_patches": state["patch_count"],
            "ch_names": list(state["channels"]),
            "channel_signature": state["signature_id"],
            "tar_files": state["tar_files"],
        }
    qc["rejected"] = dict(sorted(rejected.items()))
    qc["warnings"] = dict(sorted(warnings.items()))
    qc["accepted_by_signature"] = dict(sorted(accepted_by_signature.items()))
    qc["output_signatures"] = len(index)
    qc["output_shards"] = sum(value["n_shards"] for value in index.values())
    if qc["accepted_windows"] + qc["rejected_windows"] != len(selected):
        raise RuntimeError("pack QC accounting mismatch")
    return index, qc


def push_out(ds, local_out, out_root, sj_sfx=""):
    """Push tar files plus versioned index/QC sidecars to blob storage."""
    t0 = time.time()
    n_tar = 0
    for bucket in os.listdir(local_out):
        bucket_dir = os.path.join(local_out, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        blob_bucket = os.path.join(out_root, bucket)
        os.makedirs(blob_bucket, exist_ok=True)
        _run(f'find "{bucket_dir}" -maxdepth 1 -name "*.tar" -print0 | '
             f'xargs -0 -r -P 8 cp -t "{blob_bucket}/" 2>/dev/null')
        n_tar += len(glob.glob(os.path.join(bucket_dir, "*.tar")))
    sidecars = (
        ("wds_index.json", "_idx_" + ds + sj_sfx + ".json"),
        ("wds_qc.json", "_qc_" + ds + sj_sfx + ".json"),
    )
    for local_name, remote_name in sidecars:
        local_path = os.path.join(local_out, local_name)
        if os.path.exists(local_path):
            _run(f'cp "{local_path}" "{os.path.join(out_root, remote_name)}" 2>/dev/null')
    print(f"    [push] {ds}: {n_tar}tar 写回 {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="/data/prepared-eeg")
    ap.add_argument("--out_root", default="/data/atom-wds")
    ap.add_argument("--datasets", default=None,
                    help="逗号分隔覆盖(默认扫 in_root 所有有manifest的集, 含下游集)")
    ap.add_argument("--shard_size", type=int, default=10000, help="窗数/分片tar")
    ap.add_argument("--max_patches", type=int, default=16, help="patch上限(对齐time_embed)")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_tmp", default="/scratch/wds_tmp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subject_shards", type=int, default=1, help="大集按subject(crc32)分片, 多job并行")
    ap.add_argument("--subject_shard_id", type=int, default=0)
    ap.add_argument("--out_name", default=None, help="tar/idx别名(配--datasets单集避免不同root同名目录冲突)")
    ap.add_argument("--strict_v2", action="store_true",
                    help="要求 provenance/normalization 并启用保守 z-score 数值门槛")
    ap.add_argument("--min_channels", type=int, default=1)
    ap.add_argument("--flatline_std", type=float, default=1e-8,
                    help="删除 std<=阈值的通道; 负数关闭")
    ap.add_argument("--max_abs", type=float, default=None)
    ap.add_argument("--max_channel_mean_abs", type=float, default=None)
    ap.add_argument("--channel_std_min", type=float, default=None)
    ap.add_argument("--channel_std_max", type=float, default=None)
    ap.add_argument("--require_provenance", action="store_true")
    ap.add_argument("--require_normalization", action="store_true")
    ap.add_argument("--hash_tars", action="store_true",
                    help="record SHA-256 for each output tar; strict v2 enables it")
    args = ap.parse_args()
    if args.strict_v2:
        args.require_provenance = True
        args.require_normalization = True
        args.hash_tars = True
        if args.max_abs is None:
            args.max_abs = 15.5
        if args.max_channel_mean_abs is None:
            args.max_channel_mean_abs = 5.0
        if args.channel_std_min is None:
            args.channel_std_min = 0.05
        if args.channel_std_max is None:
            args.channel_std_max = 5.0

    if args.datasets:
        datasets = sorted(d.strip() for d in args.datasets.split(',') if d.strip())
    else:
        datasets = sorted(d for d in os.listdir(args.in_root)
                          if os.path.isdir(os.path.join(args.in_root, d))
                          and not d.startswith('_') and not d.startswith('.'))
    mine = [ds for i, ds in enumerate(datasets) if i % args.n_shards == args.shard_id]
    print(f"[wds shard {args.shard_id}/{args.n_shards}] 总{len(datasets)}集 本片{len(mine)}集, "
          f"std_1020={len(STD_1020)}ch shard_size={args.shard_size}", flush=True)

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.local_tmp, exist_ok=True)
    tot_win = tot_tar = 0
    for k, ds in enumerate(mine):
        print(f"[{k+1}/{len(mine)}] {ds}", flush=True)
        if not args.force and os.path.exists(os.path.join(args.out_root, f"_idx_{ds}.json")):
            print(f"    [skip] {ds}: 已打包(resume)", flush=True)
            continue
        mani = load_manifest(os.path.join(args.in_root, ds))
        if not mani:
            print(f"    [skip] {ds}: 无manifest", flush=True)
            continue
        local_in = os.path.join(args.local_tmp, 'in', ds)
        local_out = os.path.join(args.local_tmp, 'out', ds)
        shutil.rmtree(local_out, ignore_errors=True)
        os.makedirs(local_out, exist_ok=True)
        out_ds = args.out_name if args.out_name else ds
        n_in = stage_in(ds, args.in_root, local_in, args.subject_shards, args.subject_shard_id)
        if n_in == 0:
            shutil.rmtree(local_in, ignore_errors=True)
            continue
        t1 = time.time()
        index, qc = pack_dataset_local(
            out_ds, mani, local_in, local_out, args.shard_size,
            args.max_patches, args.seed, args.subject_shards,
            args.subject_shard_id, qc_options={
                "source_dataset": ds,
                "min_channels": args.min_channels,
                "flatline_std": args.flatline_std,
                "max_abs": args.max_abs,
                "max_channel_mean_abs": args.max_channel_mean_abs,
                "channel_std_min": args.channel_std_min,
                "channel_std_max": args.channel_std_max,
                "require_provenance": args.require_provenance,
                "require_normalization": args.require_normalization,
                "hash_tars": args.hash_tars,
            })
        with open(os.path.join(local_out, "wds_index.json"), "w") as handle:
            json.dump(index, handle, indent=1)
        with open(os.path.join(local_out, "wds_qc.json"), "w") as handle:
            json.dump(qc, handle, indent=2)
        nw = qc["accepted_windows"]
        nt = qc["output_shards"]
        print(
            f"    [pack] {ds}: accepted={nw} rejected={qc['rejected_windows']} "
            f"{len(index)} signatures {nt}tar {time.time()-t1:.0f}s",
            flush=True)
        push_out(out_ds, local_out, args.out_root, ("_sj%d" % args.subject_shard_id) if args.subject_shards > 1 else "")
        tot_win += nw; tot_tar += nt
        shutil.rmtree(local_in, ignore_errors=True)
        shutil.rmtree(local_out, ignore_errors=True)
    print(f"\n[wds shard {args.shard_id}] 完成: {tot_win}窗 -> {tot_tar}tar", flush=True)


if __name__ == "__main__":
    main()

