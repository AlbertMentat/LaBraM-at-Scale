#!/usr/bin/env python3
"""Sharded frozen-encoder feature extraction + merge (1019b).

Cache layout (key = encoder checkpoint sha + input contract + task split, so the encoder/LLM/data grid reuses it):
  <cache_root>/<enc_sha12>/<contract>/<TASK>__<output_id>__<split_mode>_seed<seed>[__subset<sha8>]/
      {train,val,test}.pt   feats [N, A, D] fp16 (channel-mean patch tokens), labels, indices (manifest order)
      stats.pt              per-dim mean/std over ALL train patch tokens (exact, from shard sums)
      COMPLETE.json         counts + sha256 of every file + provenance   (written last by the merge)
      shards/<N>-<i>/       per-shard partial files + DONE.json
Shard i of N takes positions k with k % N == i of every split (sharded-pkl tasks such as TUEV: whole shard files,
assigned by first-appearance file rank % N, so each 32 MB file is read once).

  single job, N shards: python extract_1019b.py --config C --shard i --num-shards N   (repeat for i) then --merge
  one multi-GPU job:    torchrun --nproc_per_node N extract_1019b.py --config C        (rank = shard; rank 0 merges)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
from eegllm.data import SampleDataset, build_tasks, collate, load_task_specs  # noqa: E402

SPLITS = ('train', 'val', 'test')


def log(*a):
    print(time.strftime('[%m-%d %H:%M:%S]'), f'[rank {os.environ.get("RANK", "-")}]', *a, flush=True)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def atomic_save(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- cache keys
def contract_name(ecfg: dict) -> str:
    return f"scale{float(ecfg['input_scale']):g}_z{int(bool(ecfg.get('zscore', False)))}_amp-{ecfg.get('amp', 'fp16')}_chmean-v1"


def task_dir(cfg: dict, task: str, spec) -> Path:
    ec = cfg['encoder']
    root = Path(cfg['feature_cache']['root']) / ec['checkpoint_sha256'][:12] / contract_name(ec)
    name = f'{task}__{spec.output_id}__{spec.split_mode}_seed{cfg.get("split_seed", 42)}'
    sub = cfg['data'].get('subset_manifest')
    if sub:
        name += f'__subset{sha256_file(sub)[:8]}'
    return root / name


def cache_complete(cfg, task, spec) -> bool:
    return (task_dir(cfg, task, spec) / 'COMPLETE.json').exists()


def shard_positions(td, split, shard, num_shards):
    idx = td.splits[split]
    if td.entries[idx[0]].get('item_index') is not None:  # sharded pkl files: assign whole files
        rank = {}
        for i in idx:
            rank.setdefault(td.entries[i]['path'], len(rank))
        return [i for i in idx if rank[td.entries[i]['path']] % num_shards == shard]
    return [i for k, i in enumerate(idx) if k % num_shards == shard]


class ShardItems(torch.utils.data.Dataset):
    def __init__(self, td, indices):
        by = {}
        for i in indices:
            by.setdefault(td.entries[i]['path'], []).append(i)
        self.td, self.groups = td, list(by.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, k):
        out = [(i, *self.td.read(i)) for i in self.groups[k]]
        self.td._lru.clear()
        return out


# ----------------------------------------------------------------------------- extraction (one shard)
def extract_shard(cfg, names, shard, num_shards, device, log_every=20):
    from einops import rearrange
    from eegllm.encoder import build_encoder
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    todo = [t for t in names if not cache_complete(cfg, t, specs[t])]
    if not todo:
        log('all requested task caches complete')
        return
    tasks = build_tasks(cfg, {t: specs[t] for t in todo})
    enc, lu, rep = build_encoder(cfg['encoder'])
    enc.to(device).eval()
    nw = int(cfg['data'].get('num_workers', 12))
    bs = int(cfg['feature_cache'].get('batch_size', 128))
    for t in todo:
        td = tasks[t]
        sdir = task_dir(cfg, t, specs[t]) / 'shards' / f'{num_shards}-{shard}'
        if (sdir / 'DONE.json').exists():
            log(f'{t}: shard {shard}/{num_shards} already done')
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        chans = torch.tensor(lu.get_input_chans(td.raw_ch_names), device=device)
        C, A = len(td.canonical_ch), td.canonical_n
        stats = {'sum': None, 'sumsq': None, 'n_tokens': 0}
        counts = {}
        for split in SPLITS:
            pos = shard_positions(td, split, shard, num_shards)
            feats, labels, order = [], [], []
            t0 = time.time()

            def run(xs, ys, ii):
                x = torch.from_numpy(np.stack(xs)).to(device).float()
                if cfg['encoder'].get('zscore', False):
                    x = ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)).clamp(-15, 15)
                x = rearrange(x * float(cfg['encoder']['input_scale']), 'B C (A T) -> B C A T', T=200)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    f = enc.forward_features(x, input_chans=chans, return_patch_tokens=True)
                f = f.float()
                if split == 'train':
                    fd = f.reshape(-1, f.shape[-1]).double()
                    stats['sum'] = fd.sum(0) if stats['sum'] is None else stats['sum'] + fd.sum(0)
                    stats['sumsq'] = (fd * fd).sum(0) if stats['sumsq'] is None else stats['sumsq'] + (fd * fd).sum(0)
                    stats['n_tokens'] += fd.shape[0]
                feats.append(f.view(f.shape[0], C, A, -1).mean(1).half().cpu())
                labels.extend(ys)
                order.extend(ii)
                if len(feats) % log_every == 0:
                    log(f'{t} {split}: {len(order)}/{len(pos)} ({len(order)/(time.time()-t0):.1f} samples/s)')

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
                dl = torch.utils.data.DataLoader(SampleDataset(tasks, [(t, i) for i in pos]), batch_size=bs,
                                                 num_workers=nw, collate_fn=collate)
                for b in dl:
                    run(list(b['groups'][t]['eeg'].numpy()), b['labels'].tolist(), b['indices'])
            obj = {'feats': torch.cat(feats) if feats else torch.zeros(0, A, enc.embed_dim, dtype=torch.float16),
                   'labels': torch.tensor(labels, dtype=torch.long), 'indices': torch.tensor(order, dtype=torch.long)}
            atomic_save(sdir / f'{split}.pt', obj)
            counts[split] = len(order)
            log(f'{t} {split}: shard {shard}/{num_shards} {len(order)} samples in {time.time()-t0:.0f}s')
        atomic_save(sdir / 'stats_partial.pt', {'sum': stats['sum'].cpu() if stats['sum'] is not None else None,
                                                'sumsq': stats['sumsq'].cpu() if stats['sumsq'] is not None else None,
                                                'n_tokens': stats['n_tokens']})
        atomic_json(sdir / 'DONE.json', {'counts': counts, 'n_channels': C, 'n_patches': A, 'encoder': rep,
                                         'files': {p.name: sha256_file(p) for p in sorted(sdir.glob('*.pt'))},
                                         'finished': dt.datetime.now(dt.timezone.utc).isoformat()})
    del enc
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- merge
def merge(cfg, names, num_shards):
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    tasks = None
    for t in names:
        tdir = task_dir(cfg, t, specs[t])
        if (tdir / 'COMPLETE.json').exists():
            log(f'{t}: already complete')
            continue
        if tasks is None:
            tasks = build_tasks(cfg, {n: specs[n] for n in names})
        td = tasks[t]
        sdirs = [tdir / 'shards' / f'{num_shards}-{i}' for i in range(num_shards)]
        missing = [str(s) for s in sdirs if not (s / 'DONE.json').exists()]
        if missing:
            raise SystemExit(f'{t}: shards not done: {missing}')
        dones = [json.loads((s / 'DONE.json').read_text()) for s in sdirs]
        for s, d in zip(sdirs, dones):  # integrity of every shard file
            for name, h in d['files'].items():
                if sha256_file(s / name) != h:
                    raise SystemExit(f'{t}: hash mismatch {s / name}')
        files = {}
        for split in SPLITS:
            parts = [torch.load(s / f'{split}.pt', weights_only=True) for s in sdirs]
            idx = torch.cat([p['indices'] for p in parts]).tolist()
            want = td.splits[split]
            if len(idx) != len(want) or set(idx) != set(want) or len(set(idx)) != len(idx):
                raise SystemExit(f'{t} {split}: shard union {len(idx)} != manifest {len(want)}')
            pos = {i: k for k, i in enumerate(idx)}
            perm = torch.tensor([pos[i] for i in want])
            obj = {'feats': torch.cat([p['feats'] for p in parts])[perm], 'labels': torch.cat([p['labels'] for p in parts])[perm],
                   'indices': torch.tensor(want)}
            exp = torch.tensor([int(td.entries[i]['label']) for i in want])
            if not torch.equal(obj['labels'], exp):
                raise SystemExit(f'{t} {split}: label mismatch vs index')
            atomic_save(tdir / f'{split}.pt', obj)
            files[f'{split}.pt'] = sha256_file(tdir / f'{split}.pt')
        st = [torch.load(s / 'stats_partial.pt', weights_only=True) for s in sdirs]
        n = sum(x['n_tokens'] for x in st)
        s1 = sum(x['sum'] for x in st if x['sum'] is not None)
        s2 = sum(x['sumsq'] for x in st if x['sumsq'] is not None)
        mu = s1 / n
        sd = (s2 / n - mu * mu).clamp(min=1e-8).sqrt()
        atomic_save(tdir / 'stats.pt', {'mean': mu.float(), 'std': sd.float() + 1e-4, 'n_tokens': n})
        files['stats.pt'] = sha256_file(tdir / 'stats.pt')
        atomic_json(tdir / 'COMPLETE.json', {
            'task': t, 'classes': specs[t].classes, 'n_channels': dones[0]['n_channels'], 'n_patches': dones[0]['n_patches'],
            'counts': {s: len(td.splits[s]) for s in SPLITS}, 'num_shards': num_shards, 'files': files,
            'encoder_checkpoint': cfg['encoder']['checkpoint'], 'encoder_sha256': cfg['encoder']['checkpoint_sha256'],
            'contract': contract_name(cfg['encoder']), 'merged': dt.datetime.now(dt.timezone.utc).isoformat()})
        log(f'{t}: merged {num_shards} shards -> {tdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--num-shards', type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--wait-merge', action='store_true', help='after this shard, wait for all shards (DONE.json) then merge')
    ap.add_argument('--wait-hours', type=float, default=24.0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    names = list(cfg['tasks'])
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.merge:
        merge(cfg, names, int(args.num_shards))
        return
    if world > 1:  # torchrun: rank = shard, rank 0 merges after a barrier
        import torch.distributed as dist
        dist.init_process_group('gloo', timeout=dt.timedelta(hours=6))
        rank, local = dist.get_rank(), int(os.environ.get('LOCAL_RANK', '0'))
        if os.environ.get('EEGLLM_SAME_GPU') == '1':  # local tests: several ranks share one GPU
            local = 0
        torch.cuda.set_device(local)
        extract_shard(cfg, names, rank, world, torch.device('cuda', local))
        dist.barrier()
        if rank == 0:
            merge(cfg, names, world)
        dist.barrier()
        dist.destroy_process_group()
        return
    shard, n = (args.shard or 0), (args.num_shards or 1)
    extract_shard(cfg, names, shard, n, torch.device('cuda', 0))
    if n == 1:
        merge(cfg, names, 1)
    elif args.wait_merge:
        specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
        deadline = time.time() + args.wait_hours * 3600
        while True:
            pending = [f'{t}/{i}' for t in names if not cache_complete(cfg, t, specs[t]) for i in range(n)
                       if not (task_dir(cfg, t, specs[t]) / 'shards' / f'{n}-{i}' / 'DONE.json').exists()]
            if not pending:
                break
            if time.time() > deadline:
                raise SystemExit(f'timed out waiting for shards: {pending[:10]}')
            log(f'waiting for {len(pending)} shard(s), e.g. {pending[:4]}')
            time.sleep(120)
        merge(cfg, names, n)


if __name__ == '__main__':
    main()
