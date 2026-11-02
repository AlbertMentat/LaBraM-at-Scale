#!/usr/bin/env python3
"""Omni23 x 3-seed feature extraction for ANY encoder (1020b; copy of extract_1020.py whose autocast dtype follows
encoder.amp: fp16 for the legacy FT runs, bf16 where that encoder's FT recipe used bf16, e.g. 25k Giant).
Every sample of the UNION of the seed-42/10/5 splits is encoded once.

Per task cache dir (key = encoder sha + input contract + task + seeds + data identity):
  <root>/<enc_sha12>/<contract>/<TASK>__<output_id>__<split_mode>__seeds42-10-5__data<sha16>/
      all.pt         feats [N, A, D] fp16 (channel-mean patch tokens), labels, indices (sorted),
                     tok_sum / tok_sumsq [N, D] fp32 (per-sample sums over all C*A patch tokens, for exact per-seed
                     train-split normalisation statistics computed later from the cache)
      COMPLETE.json  counts, per-seed split sizes, n_channels/n_patches, sha256 of all.pt, provenance
      shards/<N>-<i>/  partial files + DONE.json
Canonical layout (channels, patch count) = first train sample of the seed-42 split, exactly as in the FT runtime
(verified identical channel list / length for the first train sample of seeds 42, 10 and 5 for all 23 tasks).
Shard i of N takes union positions k % N == i (sharded-pkl tasks: whole files by first-appearance rank % N).

  N single-GPU jobs:  python extract_1020.py --config C --shard i --num-shards N [--wait-merge on shard 0]
  merge only:         python extract_1020.py --config C --merge --num-shards N
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegllm.config import load_config, resolve_path  # noqa: E402
from cache_identity import cache_identity, require_cache_identity, require_unchanged_data  # noqa: E402
from eegllm.data import SampleDataset, TaskData, collate, load_task_specs  # noqa: E402
from extract_1019b import ShardItems, atomic_json, atomic_save, contract_name, sha256_file  # noqa: E402
from extract_1020 import task_dir, cache_complete, shard_complete, seed_splits, check_splits, shard_positions  # noqa: E402,F401  (same cache layout)

SEEDS = (42, 10, 5)
SPLITS = ('train', 'val', 'test')


def log(*a):
    print(time.strftime('[%m-%d %H:%M:%S]'), *a, flush=True)


def seed_splits(cfg, spec) -> dict[int, dict[str, list[int]]]:
    root = Path(cfg['data']['index_root']) / spec.output_id
    out = {}
    for s in cfg.get('seeds', SEEDS):
        m = json.loads((root / 'splits' / f'{spec.split_mode}_seed{s}' / 'split_manifest.json').read_text())
        if m.get('seed') != s:
            raise ValueError(f'{spec.name}: manifest seed mismatch for {s}')
        out[s] = {k: list(m['splits'][k]) for k in SPLITS}
    return out


def check_splits(spec, entries, splits) -> dict:
    """Disjointness of train/val/test per seed (samples always; subjects when split_mode is cross_subject)."""
    rep = {}
    for s, sp in splits.items():
        sets = {k: set(v) for k, v in sp.items()}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if sets[a] & sets[b]:
                raise ValueError(f'{spec.name} seed {s}: sample overlap {a}/{b}')
        subj = {k: {str(entries[i]['subject_id']) for i in v} for k, v in sp.items()}
        overlap = {f'{a}/{b}': len(subj[a] & subj[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
        if spec.split_mode == 'cross_subject' and any(overlap.values()):
            raise ValueError(f'{spec.name} seed {s}: subject leakage {overlap}')
        rep[s] = {'sizes': {k: len(v) for k, v in sp.items()}, 'subjects': {k: len(v) for k, v in subj.items()},
                  'subject_overlap': overlap}
    return rep


def build_union_task(cfg, spec):
    dcfg = cfg['data']
    td = TaskData(spec, dcfg['index_root'], dcfg.get('path_map'), None, dcfg.get('shard_cache_dir'), split_seed=42)
    entries = json.loads((Path(dcfg['index_root']) / spec.output_id / 'sample_index.json').read_text())['entries']
    splits = seed_splits(cfg, spec)
    rep = check_splits(spec, entries, splits)
    union = sorted({i for sp in splits.values() for v in sp.values() for i in v})
    td.entries = {i: entries[i] for i in union}
    return td, union, splits, rep


def shard_positions(td, union, shard, n):
    if td.entries[union[0]].get('item_index') is not None:
        rank = {}
        for i in union:
            rank.setdefault(td.entries[i]['path'], len(rank))
        return [i for i in union if rank[td.entries[i]['path']] % n == shard]
    return [i for k, i in enumerate(union) if k % n == shard]


def extract_shard(cfg, names, shard, n, device, log_every=50):
    from einops import rearrange
    from eegllm.encoder import build_encoder
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    identities = {t: cache_identity(cfg, t, specs[t]) for t in names}
    todo = [t for t in names if not cache_complete(cfg, t, specs[t], identities[t])
            and not shard_complete(cfg, t, specs[t], n, shard, identities[t])]
    if not todo:
        log('nothing to do for this shard')
        return
    enc, lu, rep = build_encoder(cfg['encoder'])
    enc.to(device).eval()
    nw = int(cfg['data'].get('num_workers', 16))
    bs = int(cfg['feature_cache'].get('batch_size', 128))
    for t in todo:
        td, union, splits, srep = build_union_task(cfg, specs[t])
        identity = identities[t]
        require_unchanged_data(cfg, t, specs[t], identity)
        sdir = task_dir(cfg, t, specs[t], identity) / 'shards' / f'{n}-{shard}'
        sdir.mkdir(parents=True, exist_ok=True)
        chans = torch.tensor(lu.get_input_chans(td.raw_ch_names), device=device)
        C, A = len(td.canonical_ch), td.canonical_n
        pos = shard_positions(td, union, shard, n)
        feats, labels, order, s1, s2 = [], [], [], [], []
        t0 = time.time()

        def run(xs, ys, ii):
            x = torch.from_numpy(np.stack(xs)).to(device).float()
            if cfg['encoder'].get('zscore', False):
                x = ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)).clamp(-15, 15)
            x = rearrange(x * float(cfg['encoder']['input_scale']), 'B C (A T) -> B C A T', T=200)
            with torch.no_grad(), torch.autocast('cuda', dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[cfg['encoder'].get('amp', 'fp16')]):
                f = enc.forward_features(x, input_chans=chans, return_patch_tokens=True)
            f = f.float()
            s1.append(f.sum(1).cpu())
            s2.append((f * f).sum(1).cpu())
            feats.append(f.view(f.shape[0], C, A, -1).mean(1).half().cpu())
            labels.extend(ys)
            order.extend(ii)
            if len(feats) % log_every == 0:
                log(f'{t} shard {shard}/{n}: {len(order)}/{len(pos)} ({len(order)/(time.time()-t0):.1f} samples/s)')

        if pos and td.entries[pos[0]].get('item_index') is not None:
            dl = torch.utils.data.DataLoader(ShardItems(td, pos), batch_size=1, num_workers=nw, collate_fn=lambda b: b[0])
            buf = []
            for items in dl:
                buf.extend(items)
                while len(buf) >= bs:
                    c, buf = buf[:bs], buf[bs:]
                    run([x[1] for x in c], [x[2] for x in c], [x[0] for x in c])
            if buf:
                run([x[1] for x in buf], [x[2] for x in buf], [x[0] for x in buf])
        elif pos:
            dl = torch.utils.data.DataLoader(SampleDataset({t: td}, [(t, i) for i in pos]), batch_size=bs,
                                             num_workers=nw, collate_fn=collate)
            for b in dl:
                run(list(b['groups'][t]['eeg'].numpy()), b['labels'].tolist(), b['indices'])
        D = enc.embed_dim
        obj = {'feats': torch.cat(feats) if feats else torch.zeros(0, A, D, dtype=torch.float16),
               'labels': torch.tensor(labels, dtype=torch.long), 'indices': torch.tensor(order, dtype=torch.long),
               'tok_sum': torch.cat(s1) if s1 else torch.zeros(0, D), 'tok_sumsq': torch.cat(s2) if s2 else torch.zeros(0, D)}
        require_unchanged_data(cfg, t, specs[t], identity)
        atomic_save(sdir / 'part.pt', obj)
        atomic_json(sdir / 'DONE.json', {'cache_identity': identity, 'count': len(order), 'n_channels': C, 'n_patches': A, 'encoder': rep,
                                         'split_report': {str(k): v for k, v in srep.items()},
                                         'files': {'part.pt': sha256_file(sdir / 'part.pt')},
                                         'seconds': time.time() - t0, 'finished': dt.datetime.now(dt.timezone.utc).isoformat()})
        log(f'{t} shard {shard}/{n}: {len(order)} samples in {time.time()-t0:.0f}s')
    del enc
    torch.cuda.empty_cache()


def merge(cfg, names, n):
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    for t in names:
        identity = cache_identity(cfg, t, specs[t])
        tdir = task_dir(cfg, t, specs[t], identity)
        if cache_complete(cfg, t, specs[t], identity):
            log(f'{t}: already complete')
            continue
        td, union, splits, srep = build_union_task(cfg, specs[t])
        sdirs = [tdir / 'shards' / f'{n}-{i}' for i in range(n)]
        missing = [str(s) for i, s in enumerate(sdirs)
                   if not shard_complete(cfg, t, specs[t], n, i, identity)]
        if missing:
            raise SystemExit(f'{t}: shards not done: {missing}')
        dones = [json.loads((s / 'DONE.json').read_text()) for s in sdirs]
        for s, d in zip(sdirs, dones):
            require_cache_identity(d, identity, s / 'DONE.json')
            if sha256_file(s / 'part.pt') != d['files']['part.pt']:
                raise SystemExit(f'{t}: hash mismatch {s}')
        parts = [torch.load(s / 'part.pt', weights_only=True) for s in sdirs]
        idx = torch.cat([p['indices'] for p in parts])
        if len(idx) != len(union) or set(idx.tolist()) != set(union) or len(set(idx.tolist())) != len(union):
            raise SystemExit(f'{t}: shard union {len(idx)} != split union {len(union)}')
        perm = torch.argsort(idx)
        obj = {k: torch.cat([p[k] for p in parts])[perm] for k in ('feats', 'labels', 'tok_sum', 'tok_sumsq')}
        obj['indices'] = idx[perm]
        exp = torch.tensor([int(td.entries[i]['label']) for i in union])
        if not torch.equal(obj['labels'], exp):
            raise SystemExit(f'{t}: label mismatch vs index')
        require_unchanged_data(cfg, t, specs[t], identity)
        atomic_save(tdir / 'all.pt', obj)
        atomic_json(tdir / 'COMPLETE.json', {
            'cache_identity': identity,
            'task': t, 'classes': specs[t].classes, 'n_channels': dones[0]['n_channels'], 'n_patches': dones[0]['n_patches'],
            'n_tokens_per_sample': dones[0]['n_channels'] * dones[0]['n_patches'], 'count': len(union),
            'seeds': {str(s): {k: len(v) for k, v in sp.items()} for s, sp in splits.items()},
            'split_report': {str(k): v for k, v in srep.items()}, 'num_shards': n,
            'files': {'all.pt': sha256_file(tdir / 'all.pt')}, 'encoder_checkpoint': cfg['encoder']['checkpoint'],
            'encoder_sha256': cfg['encoder']['checkpoint_sha256'], 'contract': contract_name(cfg['encoder']),
            'extract_seconds_per_shard': [d['seconds'] for d in dones],
            'merged': dt.datetime.now(dt.timezone.utc).isoformat()})
        log(f'{t}: merged {n} shards ({len(union)} samples) -> {tdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--wait-merge', action='store_true')
    ap.add_argument('--wait-hours', type=float, default=36.0)
    ap.add_argument('--check-only', action='store_true', help='only validate splits (no extraction)')
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    names = list(cfg['tasks'])
    if a.check_only:
        specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
        for t in names:
            td, union, splits, rep = build_union_task(cfg, specs[t])
            log(t, 'union', len(union), 'C', len(td.canonical_ch), 'A', td.canonical_n, json.dumps(rep))
        return
    if a.merge:
        merge(cfg, names, a.num_shards)
        return
    torch.cuda.set_device(0)
    extract_shard(cfg, names, a.shard, a.num_shards, torch.device('cuda', 0))
    if a.num_shards == 1:
        merge(cfg, names, 1)
    elif a.wait_merge:
        specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
        deadline = time.time() + a.wait_hours * 3600
        while True:
            pending = [f'{t}/{i}' for t in names if not cache_complete(cfg, t, specs[t]) for i in range(a.num_shards)
                       if not (task_dir(cfg, t, specs[t]) / 'shards' / f'{a.num_shards}-{i}' / 'DONE.json').exists()]
            if not pending:
                break
            if time.time() > deadline:
                raise SystemExit(f'timed out waiting for shards: {pending[:10]}')
            log(f'waiting for {len(pending)} shard-task parts, e.g. {pending[:4]}')
            time.sleep(120)
        merge(cfg, names, a.num_shards)


if __name__ == '__main__':
    main()
