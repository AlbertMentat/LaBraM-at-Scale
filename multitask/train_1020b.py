#!/usr/bin/env python3
"""Encoder grid (1020b): train_1020 for ANY encoder (feature dim from the cache: Base 200 / Large 400 / Huge 800 /
Giant 1600) with host-memory-friendly features.

Differences vs train_1020 (imported, unchanged logic otherwise):
  * the projector input width = the cached feature dim (train_1019b / train_1019 hard-code 400 for Atom-L)
  * features stay on the memory-mapped union cache; each split holds an index view (no per-rank copies), batches are
    gathered on demand (Giant caches are ~4x Atom-L)
  * controls (LogReg LP, no-LLM head) run with features in host memory and the model/batches on GPU
Same protocol: 23 tasks x seeds 42/10/5, multi-template prompts, val-only selection, full test per task.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_1019 as T19  # noqa: E402
import train_1019b as TB  # noqa: E402
import train_1020 as T20  # noqa: E402  (installs the 23-task templates and the EEGLLM1020 builder)
from eegllm import prompts_1020 as P20  # noqa: E402
from cache_identity import training_data_identity, require_run_data_identity  # noqa: E402
from eegllm.config import load_config, resolve_path  # noqa: E402
from eegllm.data import load_task_specs  # noqa: E402
from eegllm.metrics import balanced_accuracy  # noqa: E402
from eegllm.projector import EEGProjector  # noqa: E402
from extract_1020 import SPLITS, seed_splits, task_dir  # noqa: E402
from cache_identity import cache_identity, require_cache_identity, require_unchanged_data  # noqa: E402

ENC_DIM = {'value': 400}


def _projector(enc_dim, *a, **k):  # replaces the hard-coded 400 in train_1019b.build_llm / train_1019.NoLLM
    return EEGProjector(ENC_DIM['value'], *a, **k)


TB.EEGProjector = _projector
T19.EEGProjector = _projector


class FeatView:
    """Rows `idx` of a (memory-mapped) union feature tensor; supports t[ix] with a LongTensor and .shape."""

    def __init__(self, base: torch.Tensor, idx: torch.Tensor):
        self.base, self.idx = base, idx
        self.device = torch.device('cpu')
        self.shape = (len(idx), *base.shape[1:])

    def __getitem__(self, ix):
        return self.base[self.idx[ix.cpu() if torch.is_tensor(ix) else ix]]

    def __len__(self):
        return len(self.idx)


def local_copy(src: Path, cfg, D) -> Path:
    """Copy a task's all.pt to node-local disk once per node (mmap over the blob mount would random-read)."""
    root = cfg['feature_cache'].get('local_dir')
    if not root:
        return src
    # keep the full cache key (encoder sha / contract / task split) in the local path; size + sha of COMPLETE files
    dst = Path(root) / src.relative_to(Path(cfg['feature_cache']['root']))
    want = json.loads((src.parent / 'COMPLETE.json').read_text())['files']['all.pt']
    marker = dst.with_name(dst.name + '.sha256')
    if D.local == 0 and not (dst.exists() and marker.exists() and marker.read_text().strip() == want):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f'.tmp{os.getpid()}')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        marker.write_text(want)  # recorded sha of the merged cache file this copy was made from
    D.barrier()
    return dst


def load_seed_views(cfg, names, seed, D=None):
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    P20.init_templates(specs)
    meta, data, stats, dims = {'tasks': {}}, {}, {}, set()
    for t in names:
        identity = cache_identity(cfg, t, specs[t])
        tdir = task_dir(cfg, t, specs[t], identity)
        comp = tdir / 'COMPLETE.json'
        if not comp.exists():
            raise SystemExit(f'feature cache for {t} not complete for the current index and splits: {tdir}; run extract.py again')
        c = json.loads(comp.read_text())
        require_cache_identity(c, identity, comp)
        require_unchanged_data(cfg, t, specs[t], identity)
        if len(c['classes']) != specs[t].num_classes:
            raise SystemExit(f'{t}: class count differs between cache and registry')
        allc = torch.load(local_copy(tdir / 'all.pt', cfg, D) if D is not None else tdir / 'all.pt',
                          weights_only=True, mmap=True)
        dims.add(int(allc['feats'].shape[-1]))
        pos = {int(i): k for k, i in enumerate(allc['indices'].tolist())}
        sp = seed_splits(cfg, specs[t])[seed]
        meta['tasks'][t] = {'classes': specs[t].classes, 'n_channels': c['n_channels'], 'n_patches': c['n_patches'],
                            'cache': str(tdir), 'cache_identity_sha256': identity['sha256']}
        data[t] = {}
        for s in SPLITS:
            ix = torch.tensor([pos[i] for i in sp[s]])
            data[t][s] = {'feats': FeatView(allc['feats'], ix), 'labels': allc['labels'][ix].clone(),
                          'indices': torch.tensor(sp[s])}
        tr = torch.tensor([pos[i] for i in sp['train']])
        n = len(tr) * int(c['n_tokens_per_sample'])
        s1 = allc['tok_sum'][tr].double().sum(0)
        s2 = allc['tok_sumsq'][tr].double().sum(0)
        mu = s1 / n
        stats[t] = (mu.float(), ((s2 / n - mu * mu).clamp(min=1e-8).sqrt().float() + 1e-4))
        require_unchanged_data(cfg, t, specs[t], identity)
    if len(dims) != 1:
        raise SystemExit(f'inconsistent feature dims across tasks: {dims}')
    ENC_DIM['value'] = dims.pop()
    return meta, data, stats


def pooled(data, t, s, mu, sd, chunk=4096):
    v = data[t][s]['feats']
    out = []
    for a in range(0, len(v), chunk):
        ix = torch.arange(a, min(len(v), a + chunk))
        out.append(((v[ix].float() - mu) / sd).mean(1))
    return torch.cat(out).numpy()


def run_lp_b(data, meta, names, stats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    for t in names:
        mu, sd = (x.cpu() for x in stats[t])
        X = {s: pooled(data, t, s, mu, sd) for s in SPLITS}
        Y = {s: data[t][s]['labels'].numpy() for s in SPLITS}
        sc = StandardScaler().fit(X['train'])
        best = None
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(sc.transform(X['train']), Y['train'])
            v = balanced_accuracy(Y['val'], clf.predict(sc.transform(X['val'])))
            if best is None or v > best[0]:
                best = (v, C, clf)
        k = len(meta['tasks'][t]['classes'])
        res[t] = {'C': best[1], **{s: T19.summarise(Y[s], best[2].predict(sc.transform(X[s])), k) for s in ('val', 'test')}}
    return {'mode': 'lp', 'per_task': res}


def run_nollm_b(cfg, data, meta, names, stats, dev):
    tc = cfg['train']
    model = T19.NoLLM(meta, names, int(cfg['projector']['num_tokens']), 1024).to(dev)
    model.stats = {t: (m.to(dev), s.to(dev)) for t, (m, s) in stats.items()}
    steps = int(tc['nollm_steps'])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc['nollm_lr']), weight_decay=0.01)
    bat = T19.Batcher(data, names, float(cfg['sampling']['alpha']), cfg['seed'])
    vpos = {t: T19.capped_positions(len(data[t]['val']['labels']), cfg['eval']['val_cap_per_task'], f'{t}-val') for t in names}

    def predict(t, s, positions, bs=2048):
        preds = []
        with torch.no_grad():
            for a in range(0, len(positions), bs):
                f = data[t][s]['feats'][torch.tensor(positions[a:a + bs])].to(dev)
                preds += model.logits(t, f).argmax(-1).tolist()
        return preds

    curves, best, best_state = [], None, None
    for step in range(steps):
        for g in opt.param_groups:
            g['lr'] = T19.lr_at(step, steps, 200, float(tc['nollm_lr']))
        b = T19.make_batch(data, 'train', bat.next(int(tc['batch_size'])))
        loss = 0
        for t, g in b['groups'].items():
            y = b['labels'][g['pos']].to(dev)
            loss = loss + F.cross_entropy(model.logits(t, g['feats'].to(dev)), y, reduction='sum')
        loss = loss / len(b['tasks'])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % int(tc['eval_every']) == 0 or step + 1 == steps:
            model.eval()
            rec = {'type': 'eval', 'step': step + 1, 'loss': loss.item(),
                   'val': {t: balanced_accuracy(data[t]['val']['labels'][vpos[t]].numpy(), predict(t, 'val', vpos[t])) for t in names}}
            model.train()
            rec['val_mean'] = float(np.mean(list(rec['val'].values())))
            curves.append(rec)
            if best is None or rec['val_mean'] > best['val_mean']:
                best, best_state = rec, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    res = {t: {s: T19.summarise(data[t][s]['labels'].numpy(), predict(t, s, list(range(len(data[t][s]['labels'])))),
                                len(meta['tasks'][t]['classes'])) for s in ('val', 'test')} for t in names}
    return {'mode': 'nollm', 'best_step': best['step'], 'per_task': res, 'curves': curves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True, choices=[42, 10, 5])
    ap.add_argument('--mode', required=True, choices=['lp', 'nollm', 'llm', 'controls'])
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    cfg['seed'] = a.seed
    D = TB.Dist(cfg)
    np.random.seed(a.seed)
    names = list(cfg['tasks'])
    out = Path(cfg['output_root']) / f"{cfg['run_name']}__seed{a.seed}"
    data_identity = training_data_identity(cfg)
    if D.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        contract = out / 'run_contract.json'
        if contract.exists():
            previous = json.loads(contract.read_text())
            if previous['config'] != cfg:
                raise SystemExit(f'{out} belongs to a different config; refusing')
            require_run_data_identity(previous, data_identity, contract)
        else:
            TB.atomic_json(contract, {'config': cfg, 'mode': a.mode, 'seed': a.seed, 'world': D.world,
                                      'created': dt.datetime.now(dt.timezone.utc).isoformat(),
                                      'feature_cache_identities': data_identity})
    D.barrier()
    meta, data, stats = load_seed_views(cfg, names, a.seed, D)
    for mode in (['lp', 'nollm'] if a.mode == 'controls' else [a.mode]):
        rp = out / f'result_{mode}.json'
        if rp.exists():
            TB.log(D, mode, 'already complete')
            continue
        tm = time.time()
        if mode == 'llm':
            r = TB.run_llm(cfg, D, data, meta, names, {t: (m.to(D.device), s.to(D.device)) for t, (m, s) in stats.items()}, out)
        elif D.rank == 0:
            r = run_lp_b(data, meta, names, stats) if mode == 'lp' else run_nollm_b(cfg, data, meta, names, stats, D.device)
        else:
            continue
        if D.rank == 0:
            r.update({'seed': a.seed, 'enc_dim': ENC_DIM['value'], 'minutes': (time.time() - tm) / 60,
                      'feature_cache': {t: meta['tasks'][t]['cache'] for t in names},
                      'n': {t: {s: int(len(data[t][s]['labels'])) for s in SPLITS} for t in names}})
            TB.atomic_json(rp, r)
            if r.get('per_task'):
                TB.log(D, 'DONE', mode, json.dumps({t: (round(v['val']['bacc'], 4), round(v['test']['bacc'], 4))
                                                    for t, v in r['per_task'].items()}), f'{r["minutes"]:.1f} min')
    D.barrier()
    if D.on:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
