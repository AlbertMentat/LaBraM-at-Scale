#!/usr/bin/env python3
"""1020d encoder grid: the 1020d training (train_1020d_r3.run_llm_d: joint projector+LoRA from step 0, all-task class
balancing, 16k cosine steps with patience, alpha 0.3, checkpoint loads mapped to CPU) on ANY encoder's 1020b
channel-mean union cache.

Taken from train_1020b (imported): the projector input width follows the cache dim (Base 200 / Large 400 / Huge 800 /
Giant 1600), features stay memory-mapped (per-node sha-verified local copy of each task's all.pt; split views gather rows
on demand, so DDP ranks share one page cache instead of holding private copies).  Everything else (prompts, eval, DDP,
val-only selection, full final val/test) is unchanged from 1020d.

  python train_1020d_grid.py --config configs/grid_1020d/llm__<enc>__<size>.yaml --seed 42
  torchrun --nproc_per_node N train_1020d_grid.py ...
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
import train_1019b as TB  # noqa: E402
import train_1020b as T20B  # noqa: E402  (projector width = cache dim; mmap union views; local copy)
import train_1020d_r3 as TD  # noqa: E402  (1020d joint training loop, CPU-mapped checkpoint loads)
from cache_identity import training_data_identity, require_run_data_identity  # noqa: E402
from eegllm.config import load_config  # noqa: E402
from extract_1020 import SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True, choices=[42, 10, 5])
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
            TB.atomic_json(contract, {'config': cfg, 'mode': 'llm', 'seed': a.seed, 'world': D.world,
                                      'created': dt.datetime.now(dt.timezone.utc).isoformat(),
                                      'feature_cache_identities': data_identity})
    D.barrier()
    rp = out / 'result_llm.json'
    if rp.exists():
        TB.log(D, 'llm already complete')
        return
    meta, data, stats = T20B.load_seed_views(cfg, names, a.seed, D)
    stats = {t: (m.to(D.device), s.to(D.device)) for t, (m, s) in stats.items()}
    TB.log(D, 'encoder', cfg['encoder'].get('name'), 'enc_dim', T20B.ENC_DIM['value'])
    tm = time.time()
    r = TD.run_llm_d(cfg, D, data, meta, names, stats, out)
    if D.rank == 0:
        r.update({'seed': a.seed, 'minutes': (time.time() - tm) / 60, 'enc_dim': T20B.ENC_DIM['value'],
                  'encoder': {k: cfg['encoder'].get(k) for k in ('name', 'checkpoint', 'checkpoint_sha256')},
                  'feature_cache': {t: meta['tasks'][t]['cache'] for t in names},
                  'n': {t: {s: int(len(data[t][s]['labels'])) for s in SPLITS} for t in names}})
        TB.atomic_json(rp, r)
        TB.log(D, 'DONE llm', json.dumps({t: (round(v['val']['bacc'], 4), round(v['test']['bacc'], 4))
                                          for t, v in r['per_task'].items()}), f'{r["minutes"]:.1f} min')
    D.barrier()
    if D.on:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
