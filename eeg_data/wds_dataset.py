"""WebDataset 格式的 EEG 数据集 (map-style, 兼容 GroupedBatchSampler).

打包产物 (tools/pack_webdataset.py) 结构:
  {root}/{ch}x{patch}/shard-{NNNNN}.tar
  每个tar: {key}.npy (归一化X float16) + {key}.json (ch_names+meta)

设计:
- map-style (非IterableDataset) → 兼容现有 GroupedBatchSampler + unified_collate_fn_v2
- 启动时建轻量索引: 扫描所有tar的成员offset (用tarfile.getmembers, 只读header快)
- __getitem__: 按 (tar文件, member) 定位, 流式读单样本
- ch_names已在打包时归一化(全standard_1020), 加载零通道处理开销
- 返回字典与 UnifiedEEGDataset_v2.__getitem__ 一致

⚠️ 通道顺序: 打包时resolve_channels已统一, ch_names即standard_1020规范名.
   input_chans由训练侧 get_input_chans(ch_names) 计算(查pos_embed).
"""
import hashlib, io, json, os, re, tarfile
import numpy as np
import torch
from collections import defaultdict
from contextlib import nullcontext

from eeg_data.wds_cache import DEFAULT_MIN_FREE_BYTES, LocalTarCache
from eeg_data.wds_sharding import SharedEpoch, rotating_partition


WDS_INDEX_CACHE_VERSION = 2


def channel_signature(ch_names):
    return hashlib.sha256(
        "\x1f".join(ch_names).encode("utf-8")).hexdigest()[:12]


def validate_array(array, n_channels, n_patches, source, strict_v2):
    if array.ndim != 2:
        raise ValueError(f"{source}: expected 2D EEG, got {array.shape}")
    expected_samples = n_patches * 200
    if array.shape[0] != n_channels:
        raise ValueError(
            f"{source}: channel shape {array.shape[0]} != metadata {n_channels}")
    if array.shape[1] < expected_samples:
        raise ValueError(
            f"{source}: samples {array.shape[1]} < expected {expected_samples}")
    if strict_v2 and array.shape[1] != expected_samples:
        raise ValueError(
            f"{source}: strict v2 samples {array.shape[1]} != {expected_samples}")
    if strict_v2 and not np.isfinite(array).all():
        raise ValueError(f"{source}: strict v2 sample contains NaN/Inf")
    return np.asarray(array[:, :expected_samples], dtype=np.float32)


def _get_collate_fn():
    """延迟 import unified_collate_fn_v2 (避免顶层 import 触发 unified_dataset 的 scipy 依赖)."""
    from eeg_data.unified_dataset import unified_collate_fn_v2
    return unified_collate_fn_v2


class WebDatasetEEG(torch.utils.data.Dataset):
    def __init__(self, root, mode='train', max_patches=16, image_size=224,
                 cache_index=True, verbose=True, strict_v2=False,
                 require_catalog=False):
        self.root = root
        self.mode = mode
        self.max_patches = max_patches
        self.image_size = image_size
        self.verbose = verbose
        self.strict_v2 = strict_v2
        self.require_catalog = require_catalog
        if require_catalog and not strict_v2:
            raise ValueError("require_catalog requires strict_v2")
        self.catalog = None
        self.catalog_digest = None
        self.catalog_tar_records = None
        self._load_catalog()
        # 索引: list of (tar_path, member_basename, n_ch, n_patches)
        self.index = []
        self.shape_groups = defaultdict(list)   # (ch, patch, ch_names_tuple) -> [idx]
        self._tar_cache = {}                     # tar_path -> TarFile (lazy, per-worker)
        self._build_index(cache_index)

    def _load_catalog(self):
        catalog_path = os.path.join(self.root, "_wds_v2_catalog.json")
        checksum_path = os.path.join(self.root, "_wds_v2_catalog.sha256")
        if not os.path.exists(catalog_path):
            if self.require_catalog:
                raise ValueError(f"missing required WDS v2 catalog: {catalog_path}")
            return
        catalog_bytes = open(catalog_path, "rb").read()
        digest = hashlib.sha256(catalog_bytes).hexdigest()
        if not os.path.exists(checksum_path):
            raise ValueError(f"catalog checksum sidecar missing: {checksum_path}")
        expected = open(checksum_path, encoding="utf-8").read().split()[0]
        if expected != digest:
            raise ValueError(f"catalog checksum mismatch: {digest} != {expected}")
        catalog = json.loads(catalog_bytes)
        if (catalog.get("schema_version") != 2
                or catalog.get("processing_version") != "wds-v2-catalog-v1"):
            raise ValueError("invalid WDS v2 catalog schema/version")
        records = catalog.get("tar_files")
        if not isinstance(records, list) or len(records) != catalog.get("total_tars"):
            raise ValueError("invalid WDS v2 catalog tar list")
        record_map = {}
        for record in records:
            relative = record.get("path")
            if (not relative or os.path.isabs(relative)
                    or ".." in relative.replace("\\", "/").split("/")):
                raise ValueError(f"unsafe catalog tar path: {relative}")
            if relative in record_map:
                raise ValueError(f"duplicate catalog tar path: {relative}")
            record_map[relative] = record
        self.catalog = catalog
        self.catalog_digest = digest
        self.catalog_tar_records = record_map

    def _build_index(self, cache_index):
        import time, pickle
        t0 = time.time()
        cache_flavor = "strict-v2" if self.strict_v2 else "legacy-v2"
        idx_cache = os.path.join(
            self.root, f".wds_index_{self.mode}_{cache_flavor}.pkl.gz")

        def _load_cache():
            import gzip
            opener = gzip.open if idx_cache.endswith('.gz') else open
            with opener(idx_cache, 'rb') as fh:
                d = pickle.load(fh)
            if (d.get("cache_version") != WDS_INDEX_CACHE_VERSION
                    or bool(d.get("strict_v2")) != self.strict_v2
                    or d.get("catalog_digest") != self.catalog_digest):
                return False
            self.index = d['index']
            self.shape_groups = defaultdict(list, d['shape_groups'])
            if self.verbose:
                print(f"WebDatasetEEG: loaded index {len(self.index)} samples from cache in {time.time()-t0:.1f}s", flush=True)
            return True

        # 已有兼容缓存: 所有 rank 直接读。
        if cache_index and os.path.exists(idx_cache) and _load_cache():
            return

        # 分布式: 仅rank0扫tar建索引(避免8rank并发打爆blobfuse), 其余rank barrier等待后读缓存
        rank = 0; world = 1; dist_ok = False
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank(); world = dist.get_world_size(); dist_ok = True
        except Exception:
            pass

        if rank == 0:
            self._scan_tars(t0)
            if cache_index:
                try:
                    import gzip
                    tmp = idx_cache + f".tmp{os.getpid()}"
                    with gzip.open(tmp, 'wb', compresslevel=4) as fh:
                        pickle.dump({
                            "cache_version": WDS_INDEX_CACHE_VERSION,
                            "strict_v2": self.strict_v2,
                            "catalog_digest": self.catalog_digest,
                            'index': self.index,
                            'shape_groups': dict(self.shape_groups),
                        }, fh)
                    os.replace(tmp, idx_cache)
                    if self.verbose:
                        print(f"WebDatasetEEG: cached index -> {idx_cache}", flush=True)
                except Exception as e:
                    if self.verbose:
                        print(f"  [warn] cache write failed: {e}", flush=True)
        # 同步: 非rank0等rank0扫完
        if dist_ok:
            try:
                import torch.distributed as dist
                dist.barrier()
            except Exception:
                pass
        # 非rank0(或缓存写好后)读缓存; 若无缓存则各自扫(降级)
        if rank != 0:
            if cache_index and os.path.exists(idx_cache):
                _load_cache()
            else:
                self._scan_tars(t0)

    def _scan_tars(self, t0):
        import time
        if self.catalog_tar_records is not None:
            tar_items = []
            for relative, record in sorted(self.catalog_tar_records.items()):
                tar_path = os.path.join(self.root, relative)
                if not os.path.isfile(tar_path):
                    raise ValueError(f"catalog tar missing: {relative}")
                actual_size = os.path.getsize(tar_path)
                if actual_size != int(record["size_bytes"]):
                    raise ValueError(
                        f"catalog tar size mismatch {relative}: "
                        f"{actual_size} != {record['size_bytes']}")
                tar_items.append((tar_path, record))
        else:
            tar_items = []
            buckets = sorted(
                bucket for bucket in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, bucket)))
            for bucket in buckets:
                bucket_dir = os.path.join(self.root, bucket)
                for filename in sorted(os.listdir(bucket_dir)):
                    if filename.endswith(".tar"):
                        tar_items.append((os.path.join(bucket_dir, filename), None))

        n_tar = 0
        for tar_path, catalog_record in tar_items:
            filename = os.path.basename(tar_path)
            try:
                with tarfile.open(tar_path) as archive:
                    members = archive.getmembers()
                    json_members = [
                        member for member in members if member.name.endswith(".json")]
                    if not json_members:
                        raise ValueError(f"{tar_path}: no JSON members")
                    metadata = json.load(archive.extractfile(json_members[0]))
                    raw_channels = [str(value).upper() for value in metadata["ch_names"]]
                    n_channels = int(metadata["n_channels"])
                    source_patches = int(metadata["n_patches"])
                    n_patches = min(source_patches, self.max_patches)
                    if (n_channels != len(raw_channels) or n_channels <= 0
                            or source_patches <= 0):
                        raise ValueError(f"{tar_path}: invalid metadata shape")
                    signature = channel_signature(raw_channels)
                    if self.strict_v2:
                        if int(metadata.get("schema_version", 0)) < 2:
                            raise ValueError(f"{tar_path}: missing v2 schema")
                        if metadata.get("processing_version") != "pack-webdataset-v2":
                            raise ValueError(f"{tar_path}: wrong processing version")
                        if metadata.get("channel_signature") != signature:
                            raise ValueError(f"{tar_path}: channel signature mismatch")
                        bucket = os.path.basename(os.path.dirname(tar_path))
                        if bucket != f"{n_channels}x{source_patches}":
                            raise ValueError(f"{tar_path}: bucket/metadata mismatch")
                        if not re.search(
                                rf"-ch{re.escape(signature)}-shard-\d+\.tar$",
                                filename):
                            raise ValueError(f"{tar_path}: filename/signature mismatch")
                        json_keys = {member.name[:-5] for member in json_members}
                        npy_keys = {
                            member.name[:-4] for member in members
                            if member.name.endswith(".npy")}
                        if json_keys != npy_keys:
                            raise ValueError(f"{tar_path}: NPY/JSON key mismatch")
                        if catalog_record is not None:
                            if catalog_record["channel_signature"] != signature:
                                raise ValueError(f"{tar_path}: catalog signature mismatch")
                            if int(catalog_record["n_channels"]) != n_channels:
                                raise ValueError(f"{tar_path}: catalog channel mismatch")
                            if int(catalog_record["n_patches"]) != source_patches:
                                raise ValueError(f"{tar_path}: catalog patch mismatch")
                            if int(catalog_record["n_windows"]) != len(json_members):
                                raise ValueError(f"{tar_path}: catalog window mismatch")
                    shape_key = (n_channels, n_patches, tuple(raw_channels))
                    for member in json_members:
                        key = member.name[:-5]
                        index = len(self.index)
                        self.index.append((
                            tar_path, key, n_channels, n_patches, raw_channels))
                        self.shape_groups[shape_key].append(index)
            except Exception as error:
                if self.strict_v2:
                    raise
                if self.verbose:
                    print(f"  [warn] {tar_path}: {error}", flush=True)
            n_tar += 1
            if self.verbose and n_tar % 200 == 0:
                print(
                    f"WebDatasetEEG: scanned {n_tar} tars, "
                    f"{len(self.index)} samples, {time.time()-t0:.0f}s",
                    flush=True)
        if self.verbose:
            source = "catalog" if self.catalog_tar_records is not None else "directory"
            print(
                f"WebDatasetEEG: indexed {len(self.index)} samples, "
                f"{len(self.shape_groups)} shape groups from {n_tar} tars "
                f"via {source} in {time.time()-t0:.1f}s", flush=True)

    def get_shape_groups(self):
        return self.shape_groups

    def __len__(self):
        return len(self.index)

    def _get_tar(self, tar_path):
        # per-worker lazy open (避免pickle TarFile)
        tf = self._tar_cache.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path)
            self._tar_cache[tar_path] = tf
        return tf

    def __getitem__(self, i):
        tar_path, key, n_ch, n_p, ch_names = self.index[i]
        tf = self._get_tar(tar_path)
        X = np.load(io.BytesIO(tf.extractfile(key + '.npy').read()), allow_pickle=False)
        X = validate_array(X, n_ch, n_p, f"{tar_path}:{key}", self.strict_v2)
        eeg = torch.from_numpy(X).float()
        # reshape到 (C, n_patches, 200) — 与UnifiedEEGDataset_v2一致
        eeg = eeg.reshape(eeg.shape[0], -1, 200)
        # img零占位(纯EEG无图像, 与UnifiedEEGDataset_v2一致, collate要求Tensor)
        img = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        return {
            'eeg': eeg,
            'img': img,
            'label': -1,
            'label_idx': -1,
            'meta': {
                'dataset': 'eegpt_pkl',
                'subject': '',
                'sample_id': key,
                'ch_names': list(ch_names),
            }
        }


# ====================================================================== #
#  WebDatasetEEGIterable — 顺序流式读 tar (IterableDataset, 根治随机seek)
# ====================================================================== #
#
# 背景: map-style WebDatasetEEG 用随机 idx 访问 blobfuse 上的 tar → 对 tar 随机
# seek + _tar_cache 累积几百句柄 → blobfuse IO风暴 → DDP 首batch卡死 NCCL超时.
# (实测 A100 8卡 knng4 在 Reset codebook 后 rank6/7 卡死 480s SIGABRT)
#
# 正道: WebDataset 应**顺序流式**读 tar (IterableDataset):
#   - 每个 tar 只打开一次, 从头读到尾, 读完即关 → 零随机seek
#   - 按 shape_group(同tar内 n_ch/n_patch/ch_names 完全相同) 累积满 batch 即 yield
#     → 满足 engine "ch_names = batch.meta.ch_names[0] # same within batch" 硬约束
#   - DDP: 按 rank 切 tar 列表; DataLoader worker 再切 → 每个tar被唯一worker读(无重复)
#   - shuffle: epoch级 shuffle tar顺序 + 样本级 shuffle buffer (流式近似随机)
#
# 用法 (替代 map-style + GroupedBatchSampler):
#   ds = WebDatasetEEGIterable(root, batch_size=256, num_replicas=W, rank=R, seed=42)
#   loader = DataLoader(ds, batch_size=None, num_workers=K, collate_fn=None)
#   # 注意: batch_size=None (dataset 自己组 batch 并 collate), 不要 batch_sampler
#   ds.set_epoch(epoch)  # 每轮前调用
#
class WebDatasetEEGIterable(torch.utils.data.IterableDataset):
    def __init__(self, root, batch_size, mode='train', max_patches=16,
                 image_size=224, num_replicas=1, rank=0, seed=42,
                 shuffle=True, shuffle_buffer=2000, drop_last=True, verbose=True,
                 cache_dir=None, cache_max_bytes=None,
                 cache_min_free_bytes=DEFAULT_MIN_FREE_BYTES,
                 data_fraction=1.0, strict_v2=False, require_catalog=False):
        super().__init__()
        self.root = root
        # 本地分片缓存目录(通常指向节点本地SSD): 命中先读本地, 未命中从blob拷到本地再读.
        # 空/None=关闭(直读blob). 也可由 env WDS_CACHE_DIR 提供.
        self.cache_dir = (cache_dir or os.environ.get('WDS_CACHE_DIR', '') or '').strip() or None
        self.cache_max_bytes = cache_max_bytes
        self.cache_min_free_bytes = cache_min_free_bytes
        # data_fraction: 样本级间隔采样(每个数据集每隔_sample_step取1个, 保持数据集间比例, 总量x fraction, 用于data scaling)
        self.data_fraction = float(data_fraction) if data_fraction else 1.0
        self._sample_step = max(1, round(1.0 / self.data_fraction)) if 0 < self.data_fraction < 1.0 else 1
        if verbose and rank == 0:
            print("WebDatasetEEGIterable: local shard cache = %s"
                  % (self.cache_dir if self.cache_dir else 'OFF (直读blob)'), flush=True)
        self.batch_size = batch_size
        self.mode = mode
        self.max_patches = max_patches
        self.image_size = image_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.drop_last = drop_last
        self.verbose = verbose
        self.strict_v2 = strict_v2
        self._shared_epoch = SharedEpoch()
        # 复用 map-style 的索引(含每样本 tar_path) 来枚举 tar 列表 + 估算 step 数.
        # 索引按 tar 聚合: tar_path -> (n_ch, n_p, raw_ch, [keys...])
        base = WebDatasetEEG(
            root, mode=mode, max_patches=max_patches,
            image_size=image_size, cache_index=True, verbose=verbose,
            strict_v2=strict_v2, require_catalog=require_catalog)
        self._tar_meta = {}   # tar_path -> dict(n_ch, n_p, raw_ch, n_samples, sg_key)
        for (tar_path, key, n_ch, n_p, raw_ch) in base.index:
            m = self._tar_meta.get(tar_path)
            if m is None:
                ch_names = tuple(c.upper() for c in raw_ch)
                relative = os.path.relpath(tar_path, self.root).replace(os.sep, "/")
                catalog_record = (
                    base.catalog_tar_records.get(relative)
                    if base.catalog_tar_records is not None else None)
                expected_size = int(
                    catalog_record["size_bytes"] if catalog_record is not None
                    else os.path.getsize(tar_path))
                expected_sha256 = (
                    catalog_record.get("sha256")
                    if catalog_record is not None else None)
                m = dict(
                    n_ch=n_ch, n_p=n_p, raw_ch=raw_ch,
                    sg_key=(n_ch, n_p, ch_names), n_samples=0,
                    relative_path=relative, size_bytes=expected_size,
                    sha256=expected_sha256)
                self._tar_meta[tar_path] = m
            m['n_samples'] += 1
        self.tar_paths = sorted(self._tar_meta.keys())
        self.total_samples = len(base.index)
        if not self.tar_paths or self.total_samples <= 0:
            raise ValueError(
                "WebDataset root contains no legacy-v2 training samples; "
                "use --wds_format wds-549 for a WDS 549 release root")
        self.local_cache = (
            LocalTarCache(
                self.cache_dir, max_bytes=self.cache_max_bytes,
                min_free_bytes=self.cache_min_free_bytes,
                verbose=verbose and rank == 0)
            if self.cache_dir else None)
        # 估算每 rank 的 batch 数(给 LR scheduler / log). 按 shape_group 各自 //batch_size 求和,
        # 再按 num_replicas 均分 — 与实际顺序组 batch 的 drop_last 行为一致(近似).
        sg_counts = defaultdict(int)
        for m in self._tar_meta.values():
            sg_counts[m['sg_key']] += m['n_samples']
        if self.drop_last:
            global_batches = sum(c // self.batch_size for c in sg_counts.values())
        else:
            global_batches = sum((c + self.batch_size - 1) // self.batch_size
                                 for c in sg_counts.values())
        self._len = max(1, int((global_batches // self.num_replicas) / self._sample_step))
        if self.verbose and rank == 0:
            print(f"WebDatasetEEGIterable: {len(self.tar_paths)} tars, "
                  f"{self.total_samples} samples, {len(sg_counts)} shape-groups, "
                  f"~{self._len} batches/rank (bs={batch_size}, world={self.num_replicas}, "
                  f"data_fraction={self.data_fraction} step={self._sample_step})",
                  flush=True)

    def set_epoch(self, epoch):
        previous = self._shared_epoch.get()
        if (self.local_cache is not None and self.rank == 0
                and int(epoch) != previous):
            print(f"WDS cache after epoch {previous}: {self.local_cache.summary()}",
                  flush=True)
        self._shared_epoch.set(epoch)

    def __len__(self):
        return self._len

    def _shard_tars(self):
        """Rotate disjoint tar ownership across ranks and persistent workers.

        A one-rank cyclic shift preserves most node-local cache affinity while
        guaranteeing that every tar visits every rank within ``world_size``
        epochs. The shared epoch also prevents persistent workers from replaying
        the same sample order forever.
        """
        import random as _random
        epoch = self._shared_epoch.get()
        tars = list(self.tar_paths)
        _random.Random(self.seed).shuffle(tars)
        tars = rotating_partition(
            tars, self.num_replicas, self.rank, epoch)
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if info is not None:
            tars = rotating_partition(
                tars, info.num_workers, info.id, epoch)
        if self.shuffle:
            _random.Random(
                self.seed + epoch * 100003 + self.rank * 131 + wid
            ).shuffle(tars)
        return tars

    def _iter_samples(self, tars):
        """Read each owned tar once, preferring the bounded node-local cache."""
        import random as _random
        epoch = self._shared_epoch.get()
        rng = _random.Random(self.seed + epoch * 7919 + self.rank)
        buffer = []
        for tar_path in tars:
            metadata = self._tar_meta[tar_path]
            n_patches = metadata['n_p']
            raw_channels = metadata['raw_ch']
            cache_context = (
                self.local_cache.open(
                    tar_path, metadata['relative_path'],
                    metadata['size_bytes'], metadata['sha256'])
                if self.local_cache is not None else nullcontext(tar_path))
            try:
                with cache_context as open_path, tarfile.open(open_path) as archive:
                    sample_index = 0
                    for member in archive:
                        if not member.name.endswith('.npy'):
                            continue
                        if self._sample_step > 1:
                            take = sample_index % self._sample_step == 0
                            sample_index += 1
                            if not take:
                                continue
                        try:
                            raw = archive.extractfile(member).read()
                        except Exception:
                            continue
                        array = np.load(io.BytesIO(raw), allow_pickle=False)
                        try:
                            array = validate_array(
                                array, metadata['n_ch'], n_patches,
                                f"{open_path}:{member.name}", self.strict_v2)
                        except ValueError as error:
                            if self.strict_v2:
                                raise
                            if self.verbose:
                                print(f"  [iter warn] {error}", flush=True)
                            continue
                        sample = (
                            array, n_patches, raw_channels,
                            metadata['sg_key'])
                        if self.shuffle and self.shuffle_buffer > 1:
                            buffer.append(sample)
                            if len(buffer) >= self.shuffle_buffer:
                                index = rng.randrange(len(buffer))
                                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                                yield buffer.pop()
                        else:
                            yield sample
            except Exception as error:
                if self.strict_v2:
                    raise
                if self.verbose:
                    print(
                        f"  [iter warn] open {tar_path}: {error}", flush=True)
        if self.shuffle:
            rng.shuffle(buffer)
        yield from buffer

    def _make_item(self, X, n_p, raw_ch):
        T = n_p * 200
        if X.shape[1] > T:
            X = X[:, :T]
        eeg = torch.from_numpy(X).float().reshape(X.shape[0], -1, 200)
        img = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        return {
            'eeg': eeg, 'img': img, 'label': -1, 'label_idx': -1,
            'meta': {'dataset': 'eegpt_pkl', 'subject': '',
                     'sample_id': '', 'ch_names': [c.upper() for c in raw_ch]},
        }

    def _gen_batches(self, tars, collate):
        """单遍历: 顺序读 tars, 按 shape_group 攒满 batch_size 即 yield 一个 collated batch."""
        from collections import defaultdict as _dd
        pending = _dd(list)
        for X, n_p, raw_ch, sg_key in self._iter_samples(tars):
            pending[sg_key].append(self._make_item(X, n_p, raw_ch))
            if len(pending[sg_key]) >= self.batch_size:
                yield collate(pending[sg_key])
                pending[sg_key] = []
        if not self.drop_last:
            for sg_key, items in pending.items():
                if items:
                    yield collate(items)

    def __iter__(self):
        """产出**恰好 target 个 batch** (DDP 各 rank/worker 等长, 否则 NCCL all-reduce 卡死).

        engine 不按 num_training_steps 截断而是跑到 dataloader 耗尽, 故各 rank 必须
        严格等长. 每个 worker 分摊 self._len 的一份; 数据不足时循环重遍历补齐.
        """
        collate = _get_collate_fn()
        info = torch.utils.data.get_worker_info()
        nworkers = info.num_workers if info is not None else 1
        wid = info.id if info is not None else 0
        # 本 worker 目标 batch 数: 均分 self._len, worker0 收余数
        target = self._len // nworkers + (1 if wid < (self._len % nworkers) else 0)
        if target <= 0:
            return
        produced = 0
        guard = 0
        while produced < target:
            tars = self._shard_tars()   # 每轮按 epoch shuffle (重遍历补齐时复用同序)
            if not tars:
                return
            for batch in self._gen_batches(tars, collate):
                yield batch
                produced += 1
                if produced >= target:
                    break
            guard += 1
            if guard > 1000:   # 防御: 极端情况下避免死循环
                break
