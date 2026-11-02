"""Task-index loading, channel alignment, split checks and weighted task sampling."""
from __future__ import annotations

import collections
import json
import os
import pickle
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PATCH = 200
SPLITS = ('train', 'val', 'test')


# ----------------------------------------------------------------------------- safe pickle
class RestrictedUnpickler(pickle.Unpickler):
    ALLOWED = {
        ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
        ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
        ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.numeric', '_frombuffer'),
        ('numpy._core.numeric', '_frombuffer'), ('_codecs', 'encode'),
        ('collections', 'OrderedDict'), ('builtins', 'list'), ('builtins', 'dict'), ('builtins', 'tuple'),
        ('builtins', 'set'), ('builtins', 'int'), ('builtins', 'float'), ('builtins', 'str'),
        ('builtins', 'bytes'), ('builtins', 'bytearray'), ('builtins', 'complex'), ('builtins', 'slice'),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or (module == 'numpy.dtypes' and name.endswith('DType')):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global in sample pickle: {module}.{name}')


def safe_pickle_load(path: str) -> Any:
    with open(path, 'rb') as handle:
        return RestrictedUnpickler(handle).load()


# ----------------------------------------------------------------------------- task specs
@dataclass
class TaskSpec:
    name: str
    output_id: str
    split_mode: str
    indexed: bool
    source_dataset_type: str
    truncate_patches: int | None
    classes: list[str]
    instruction: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_task_specs(path: str, names: list[str] | None = None) -> dict[str, TaskSpec]:
    doc = yaml.safe_load(Path(path).read_text())['tasks']
    names = names or list(doc)
    specs = {}
    for n in names:
        t = doc[n]
        specs[n] = TaskSpec(name=n, output_id=t['output_id'], split_mode=t['split_mode'], indexed=bool(t['indexed']),
                            source_dataset_type=t['source_dataset_type'], truncate_patches=t.get('truncate_patches'),
                            classes=list(t['classes']), instruction=t['instruction'])
    return specs


def norm_ch(name: str) -> str:
    return str(name).strip().upper()


# ----------------------------------------------------------------------------- sample processing
def to_patches(x: np.ndarray, truncate: int | None) -> np.ndarray:
    """[C, T] -> [C, nf, 200] (UnifiedEEGDataset_v2 'auto' reshape + truncation)."""
    if x.ndim == 3:
        out = x
    elif x.ndim == 2:
        nf = max(1, x.shape[1] // PATCH)
        out = x[:, :nf * PATCH].reshape(x.shape[0], nf, PATCH)
    else:
        raise ValueError(f'unexpected EEG shape {x.shape}')
    if truncate is not None:
        out = out[:, :truncate, :]
    return out


def align(x: np.ndarray, ch_names: list[str], canonical_ch: list[str], canonical_n: int) -> np.ndarray:
    """UnifiedFinetuneWrapper: channel alignment by name (zero fill) + patch pad/crop; returns [C, N*200]."""
    samp = [norm_ch(c) for c in ch_names]
    if samp and samp != canonical_ch:
        pos = {c: i for i, c in enumerate(samp)}
        out = np.zeros((len(canonical_ch), *x.shape[1:]), dtype=x.dtype)
        for i, c in enumerate(canonical_ch):
            j = pos.get(c)
            if j is not None and j < x.shape[0]:
                out[i] = x[j]
        x = out
    if x.shape[1] != canonical_n:
        if x.shape[1] > canonical_n:
            x = x[:, :canonical_n]
        else:
            pad = np.zeros((x.shape[0], canonical_n - x.shape[1], x.shape[2]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)
    c, n, t = x.shape
    return np.ascontiguousarray(x.reshape(c, n * t), dtype=np.float32)


class PathMapper:
    def __init__(self, mapping: dict[str, str] | None):
        self.items = sorted((mapping or {}).items(), key=lambda kv: -len(kv[0]))

    def __call__(self, path: str) -> str:
        for src, dst in self.items:
            if path.startswith(src):
                return dst + path[len(src):]
        return path


class TaskData:
    """One task: index entries, seed-42 split indices, canonical layout, sample reader."""

    def __init__(self, spec: TaskSpec, index_root: str, path_map: dict[str, str] | None = None,
                 subset: dict[str, list[int]] | None = None, shard_cache_dir: str | None = None,
                 split_seed: int = 42):
        self.spec = spec
        root = Path(index_root) / spec.output_id
        self.entries = json.loads((root / 'sample_index.json').read_text())['entries']
        manifest_path = root / 'splits' / f'{spec.split_mode}_seed{split_seed}' / 'split_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('seed') != split_seed or manifest.get('task_id') not in (None, spec.name):
            raise ValueError(f'unexpected split manifest {manifest_path}')
        self.full_splits = {s: list(manifest['splits'][s]) for s in SPLITS}
        used = {i for s in SPLITS for i in self.full_splits[s]}
        if any(not isinstance(i, int) or not 0 <= i < len(self.entries) for i in used):
            raise ValueError(f'{spec.name}: invalid split index')
        self.entries = {i: self.entries[i] for i in sorted(used)}  # keep only split members (memory)
        self._check_disjoint()
        self.splits = {s: list(v) for s, v in self.full_splits.items()}
        if subset is not None:
            for s in SPLITS:
                keep = set(subset[s])
                self.splits[s] = [i for i in self.full_splits[s] if i in keep]
        self.map_path = PathMapper(path_map)
        self.shard_cache_dir = shard_cache_dir
        self._lru: collections.OrderedDict[str, Any] = collections.OrderedDict()
        first = self.entries[self.full_splits['train'][0]]
        x0, ch0, _ = self._read_raw(first)
        self.canonical_ch = [norm_ch(c) for c in ch0]
        self.canonical_n = int(to_patches(x0, spec.truncate_patches).shape[1])
        self.raw_ch_names = list(ch0)

    def _check_disjoint(self):
        seen: set[int] = set()
        for s in SPLITS:
            cur = set(self.full_splits[s])
            if seen & cur:
                raise ValueError(f'{self.spec.name}: split overlap')
            seen |= cur
        if self.spec.split_mode == 'cross_subject':
            subj = {s: {str(self.entries[i]['subject_id']) for i in self.full_splits[s]} for s in SPLITS}
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                if subj[a] & subj[b]:
                    raise ValueError(f'{self.spec.name}: subject leakage {a}/{b}')

    # -- io
    def _local(self, path: str) -> str:
        """Sharded files (~32 MB, many items) are copied once to node-local disk, like EVAL_CACHE_DIR in FT."""
        path = self.map_path(path)
        if not self.shard_cache_dir:
            return path
        local = os.path.join(self.shard_cache_dir, path.lstrip('/'))
        try:
            if os.path.exists(local) and os.path.getsize(local) > 0:
                return local
            os.makedirs(os.path.dirname(local), exist_ok=True)
            tmp = f'{local}.tmp.{os.getpid()}'
            shutil.copyfile(path, tmp)
            os.replace(tmp, local)
            return local
        except OSError:
            return path

    def _load(self, path: str, sharded: bool) -> Any:
        if not sharded:
            return safe_pickle_load(self.map_path(path))
        if path in self._lru:
            self._lru.move_to_end(path)
            return self._lru[path]
        data = safe_pickle_load(self._local(path))
        self._lru[path] = data
        while len(self._lru) > 2:
            self._lru.popitem(last=False)
        return data

    def _read_raw(self, entry: dict) -> tuple[np.ndarray, list[str], int]:
        item = entry.get('item_index')
        data = self._load(entry['path'], sharded=item is not None)
        x = None
        for key in ('X', 'eeg', 'signal'):
            if key in data:
                x = data[key]
                break
        if x is None:
            raise KeyError(f'no EEG array in {entry["path"]}')
        y = data.get('y', data.get('label'))
        if item is not None:
            x = x[item]
            if isinstance(y, (list, tuple, np.ndarray)):
                y = y[item]
        label = int(entry['label'])
        if y is not None and not isinstance(y, (list, tuple, np.ndarray)) and int(y) != label:
            raise ValueError(f'label mismatch index={label} pkl={y} for {entry["path"]}')
        ch = entry.get('ch_names') or list(data.get('ch_names', []))
        return np.asarray(x, dtype=np.float32), ch, label

    def read(self, index: int) -> tuple[np.ndarray, int]:
        x, ch, label = self._read_raw(self.entries[index])
        x = to_patches(x, self.spec.truncate_patches)
        return align(x, ch, self.canonical_ch, self.canonical_n), label

    def labels(self, split: str) -> list[int]:
        return [int(self.entries[i]['label']) for i in self.splits[split]]

    def capped(self, split: str, cap: int | None, seed: int = 42) -> list[int]:
        """Fixed uniform random subset (seeded) of a split, kept in manifest order."""
        idx = self.splits[split]
        if cap is None or len(idx) <= cap:
            return list(idx)
        chosen = set(random.Random(f'{seed}-{self.spec.name}-{split}').sample(idx, cap))
        return [i for i in idx if i in chosen]


# ----------------------------------------------------------------------------- torch datasets
class SampleDataset(torch.utils.data.Dataset):
    """Flat list of (task, sample index) pairs."""

    def __init__(self, tasks: dict[str, TaskData], items: list[tuple[str, int]]):
        self.tasks = tasks
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        task, idx = self.items[i]
        x, y = self.tasks[task].read(idx)
        return {'task': task, 'index': idx, 'eeg': torch.from_numpy(x), 'label': y}


def collate(batch: list[dict]) -> dict:
    """Keeps sample order; groups EEG by task (each task has one canonical [C, N*200] shape)."""
    groups: dict[str, dict] = {}
    for pos, b in enumerate(batch):
        g = groups.setdefault(b['task'], {'pos': [], 'eeg': []})
        g['pos'].append(pos)
        g['eeg'].append(b['eeg'])
    for g in groups.values():
        g['eeg'] = torch.stack(g['eeg'])
    return {'tasks': [b['task'] for b in batch], 'labels': torch.tensor([b['label'] for b in batch]),
            'indices': [b['index'] for b in batch], 'groups': groups}


def task_probabilities(sizes: dict[str, int], method: str = 'temperature', alpha: float = 0.5) -> dict[str, float]:
    """proportional: p ~ n; balanced: uniform; temperature: p ~ n**alpha (alpha=1 proportional, 0 balanced)."""
    if method == 'proportional':
        alpha = 1.0
    elif method == 'balanced':
        alpha = 0.0
    elif method != 'temperature':
        raise ValueError(method)
    w = {k: float(v) ** alpha for k, v in sizes.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


class MixedTaskSampler(torch.utils.data.Sampler):
    """Infinite sampler: draw a task with `probs`, then the next sample of that task's reshuffled permutation
    (without replacement within a pass).  Deterministic in (seed, rank); `start` fast-forwards for resume."""

    def __init__(self, sizes: dict[str, int], probs: dict[str, float], seed: int, rank: int = 0, start: int = 0):
        self.names = list(sizes)
        self.sizes = [sizes[n] for n in self.names]
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.p = np.array([probs[n] for n in self.names], dtype=np.float64)
        self.p /= self.p.sum()
        self.seed, self.rank, self.start = seed, rank, start

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.rank, 7919])
        perms = [rng.permutation(n) for n in self.sizes]
        pos = [0] * len(self.sizes)
        k = 0
        while True:
            t = int(rng.choice(len(self.sizes), p=self.p))
            if pos[t] >= self.sizes[t]:
                perms[t] = rng.permutation(self.sizes[t])
                pos[t] = 0
            local = int(perms[t][pos[t]])
            pos[t] += 1
            if k >= self.start:
                yield self.offsets[t] + local
            k += 1

    def __len__(self):  # pragma: no cover - infinite
        return 2 ** 62


def build_tasks(cfg: dict, specs: dict[str, TaskSpec]) -> dict[str, TaskData]:
    dcfg = cfg['data']
    subset = None
    if dcfg.get('subset_manifest'):
        subset = json.loads(Path(dcfg['subset_manifest']).read_text())['subset']
    return {n: TaskData(s, dcfg['index_root'], dcfg.get('path_map'), subset[n] if subset else None,
                        dcfg.get('shard_cache_dir')) for n, s in specs.items()}

