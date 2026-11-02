#!/usr/bin/env python3
"""Omni23 x seed multitask EEG->Qwen3 training (1020) on the union feature cache of extract_1020.py.

One run = one seed s: train on the union of all tasks' train_s, select on val_s only (per-stage best by mean val BAcc
over the run's tasks; val subset during training, test never used for selection), report full test_s per task.
Per-task normalisation statistics are computed per seed from that seed's TRAIN samples (exact token-level mean/std
from the cached per-sample token sums).  Training/eval/DDP/FSDP machinery = train_1019b (imported unchanged);
1020 swaps in the 23-task prompt templates (eegllm/prompts_1020.py, model_1020.EEGLLM1020) and the union cache.

  python train_1020.py --config configs/pilot_1020_<size>.yaml --seed 42 --mode llm|controls
  torchrun --nproc_per_node N train_1020.py ... (DDP; dist.fsdp for large models)
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

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_1019 as T19  # noqa: E402
import train_1019b as TB  # noqa: E402
from eegllm import prompts_1020 as P20  # noqa: E402
from cache_identity import training_data_identity, require_run_data_identity  # noqa: E402
from eegllm.config import load_config, resolve_path  # noqa: E402
from eegllm.data import load_task_specs  # noqa: E402
from extract_1020 import SPLITS, seed_splits, task_dir  # noqa: E402
from cache_identity import cache_identity, require_cache_identity, require_unchanged_data  # noqa: E402

# 23-task templates for prompt sampling and per-template evaluation (functions in train_1019/1019b look these up
# as module globals at call time)
T19.num_templates = P20.num_templates
TB.num_templates = P20.num_templates


def build_llm_1020(cfg, names, stats, D):
    """train_1019b.build_llm with the 1020 model class (23-task templates)."""
    import eegllm.model_1019b as M19b
    from eegllm.model_1020 import EEGLLM1020
    orig = M19b.EEGLLM1019b
    M19b.EEGLLM1019b = EEGLLM1020  # build_llm imports the class from this module at call time
    try:
        return _orig_build_llm(cfg, names, stats, D)
    finally:
        M19b.EEGLLM1019b = orig


_orig_build_llm = TB.build_llm
TB.build_llm = build_llm_1020


def load_seed(cfg, names, seed, device):
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    P20.init_templates(specs)
    meta, data, stats = {'tasks': {}}, {}, {}
    for t in names:
        identity = cache_identity(cfg, t, specs[t])
        tdir = task_dir(cfg, t, specs[t], identity)
        comp = tdir / 'COMPLETE.json'
        deadline = time.time() + 3600 * float(cfg['feature_cache'].get('wait_hours', 0))
        while not comp.exists() and time.time() < deadline:  # optional: wait for the extraction merge
            time.sleep(300)
        if not comp.exists():
            raise SystemExit(f'union feature cache for {t} not complete for the current index and splits: {tdir}; run extract.py again')
        c = json.loads(comp.read_text())
        require_cache_identity(c, identity, comp)
        require_unchanged_data(cfg, t, specs[t], identity)
        allc = torch.load(tdir / 'all.pt', weights_only=True, mmap=True)  # page cache shared by co-located ranks
        pos = {int(i): k for k, i in enumerate(allc['indices'].tolist())}
        sp = seed_splits(cfg, specs[t])[seed]
        if len(c['classes']) != specs[t].num_classes:
            raise SystemExit(f'{t}: class count differs between cache and registry')
        # class NAMES come from the registry (prompt wording may change without re-extracting features)
        meta['tasks'][t] = {'classes': specs[t].classes, 'n_channels': c['n_channels'], 'n_patches': c['n_patches'],
                            'cache': str(tdir), 'cache_identity_sha256': identity['sha256']}
        data[t] = {}
        for s in SPLITS:
            ix = torch.tensor([pos[i] for i in sp[s]])
            data[t][s] = {'feats': allc['feats'][ix].to(device), 'labels': allc['labels'][ix].to(device),
                          'indices': torch.tensor(sp[s])}
            if len(data[t][s]['labels']) != c['seeds'][str(seed)][s]:
                raise SystemExit(f'{t} seed {seed} {s}: count mismatch')
        tr = torch.tensor([pos[i] for i in sp['train']])
        n = len(tr) * int(c['n_tokens_per_sample'])
        s1 = allc['tok_sum'][tr].double().sum(0)
        s2 = allc['tok_sumsq'][tr].double().sum(0)
        mu = s1 / n
        sd = (s2 / n - mu * mu).clamp(min=1e-8).sqrt()
        stats[t] = (mu.float().to(device), (sd.float() + 1e-4).to(device))
        require_unchanged_data(cfg, t, specs[t], identity)
        del allc
    return meta, data, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True, choices=[42, 10, 5])
    ap.add_argument('--mode', required=True, choices=['lp', 'nollm', 'llm', 'controls'])
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cfg['seed'] = args.seed  # training seed == data split seed (as in the Omni23 multiseed protocol)
    D = TB.Dist(cfg)
    np.random.seed(args.seed)
    names = list(cfg['tasks'])
    out = Path(cfg['output_root']) / f"{cfg['run_name']}__seed{args.seed}"
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
            TB.atomic_json(contract, {'config': cfg, 'mode': args.mode, 'seed': args.seed, 'world': D.world,
                                      'created': dt.datetime.now(dt.timezone.utc).isoformat(),
                                      'feature_cache_identities': data_identity})
    D.barrier()
    dev = D.device if (args.mode != 'llm' or cfg['data'].get('feats_on_gpu', False)) else torch.device('cpu')
    meta, data, stats = load_seed(cfg, names, args.seed, dev)
    stats = {t: (m.to(D.device), s.to(D.device)) for t, (m, s) in stats.items()}
    for mode in (['lp', 'nollm'] if args.mode == 'controls' else [args.mode]):
        rp = out / f'result_{mode}.json'
        if rp.exists():
            TB.log(D, mode, 'already complete')
            continue
        tm = time.time()
        if mode == 'llm':
            r = TB.run_llm(cfg, D, data, meta, names, stats, out)
        elif D.rank == 0:
            r = T19.run_lp(data, meta, names, stats) if mode == 'lp' else T19.run_nollm(cfg, data, meta, names, stats, out)
        else:
            continue
        if D.rank == 0:
            r.update({'seed': args.seed, 'minutes': (time.time() - tm) / 60,
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
