#!/usr/bin/env python3
"""Joint projector and LoRA training for the reference Large-encoder feature cache.

Resume loads tensors onto CPU before moving state to the active rank.
Validation macro balanced accuracy selects the joint checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
import train_1020 as T20  # noqa: E402  (23-task templates, EEGLLM1020 builder, union-cache loader)
from cache_identity import training_data_identity, require_run_data_identity  # noqa: E402
from eegllm.config import load_config  # noqa: E402
from extract_1020 import SPLITS  # noqa: E402
from train_1019 import Batcher, capped_positions, class_weights, lr_at, prompt_keys  # noqa: E402
from eegllm.prompts_1020 import num_templates  # noqa: E402
from checkpoint_state import (collect_optimizer_state, collect_rng_states, optimizer_format,
                              restore_optimizer_state, restore_rng_states)  # noqa: E402


def joint_param_groups(model, jc):
    """Make projector + LoRA trainable (LoRA injected if absent, lora_B = 0 so step 0 == base LLM); returns
    (param groups, replicated params, sharded params) for AdamW / sync_and_clip."""
    params, rep, shd = TB.configure_stage(model, {'name': 'lora', 'lora': jc['lora']})
    proj = {id(p) for p in model.projector.parameters()}
    pg = [p for p in params if id(p) in proj]
    lg = [p for p in params if id(p) not in proj]
    if not pg or not lg:
        raise RuntimeError('joint stage needs both projector and LoRA parameters')
    groups = [{'name': 'projector', 'params': pg, 'base_lr': float(jc['lr_projector']),
               'weight_decay': float(jc.get('weight_decay_projector', 0.0))},
              {'name': 'lora', 'params': lg, 'base_lr': float(jc['lr_lora']),
               'weight_decay': float(jc.get('weight_decay_lora', 0.01))}]
    return groups, rep, shd


def set_lrs(opt, k, steps, warmup, scale):
    for g in opt.param_groups:
        g['lr'] = lr_at(k, steps, warmup, g['base_lr'] * scale)


def run_llm_d(cfg, D, data, meta, names, stats, out: Path):
    tc, ec, jc = cfg['train'], cfg['eval'], cfg['train']['joint']
    model, llm_path = TB.build_llm(cfg, names, stats, D)
    G = int(tc['batch_size'])
    if G % D.world:
        raise SystemExit(f'global batch {G} not divisible by world {D.world}')
    micro = int(tc.get('micro_batch') or G // D.world)
    base = float(tc.get('base_global_batch', G))
    scale = {'none': 1.0, 'linear': G / base, 'sqrt': math.sqrt(G / base)}[tc.get('lr_scaling', 'none')]
    use_tpl = bool(cfg.get('prompts', {}).get('multi_template', False))
    cb = names if tc.get('class_balance') == 'all' else list(tc.get('class_balanced_tasks', []))
    cw = {t: v.to(D.device) for t, v in class_weights(data, names, cb).items()}
    bat = Batcher(data, names, float(cfg['sampling']['alpha']), cfg['seed'])
    TB.log(D, f'world {D.world} global batch {G} micro {micro} lr x{scale:.3f} alpha {cfg["sampling"]["alpha"]} '
              f'class-balanced {len(cb)}/{len(names)} sampling p {json.dumps({t: round(p, 4) for t, p in bat.p.items()})}')
    prng = np.random.default_rng(cfg['seed'] + 99)
    vpos = {t: capped_positions(len(data[t]['val']['labels']), ec['val_cap_per_task'], f'{t}-val') for t in names}
    tpos = {t: capped_positions(len(data[t]['test']['labels']), ec['test_cap_per_task'], f'{t}-test') for t in names}
    ebs = int(ec['batch_size'])
    state_dir = out / 'state'
    if D.rank == 0:
        state_dir.mkdir(parents=True, exist_ok=True)
    D.barrier()
    groups, rep, shd = joint_param_groups(model, jc)
    opt = torch.optim.AdamW(groups, lr=0.0, betas=(0.9, 0.98))
    st = torch.load(state_dir / 'last.pt', weights_only=True, map_location='cpu') if (state_dir / 'last.pt').exists() else None
    curves = json.loads(st['curves']) if st else []
    best = json.loads(st['best']) if st else {}
    k, bad, done = (st['step'], st['bad'], st['done']) if st else (0, 0, False)
    if st:
        TB.log(D, f'resuming joint step {k} (done={done})')
        TB.set_state(model, st['trainable'])
        if not done:
            restored = restore_optimizer_state(
                model, opt, st.get('optimizer'), st.get('optimizer_format'),
                allow_empty=os.environ.get('EEGLLM_ALLOW_OPTIMIZER_RESET') == '1')
            if not restored:
                TB.log(D, 'Optimizer history is absent; restarting AdamW moments as explicitly requested.')
        bat.load(st['batcher'])
        prng.bit_generator.state = json.loads(st['prng'])
        if not restore_rng_states(st.get('torch_rng'), D.device) and not done:
            TB.log(D, 'This older checkpoint has no saved PyTorch RNG state; dropout resumes from the current RNG.')
        del st
    steps, warmup = int(jc['steps']), int(jc.get('warmup', 500))
    patience, every = int(jc.get('patience', 0)), int(tc['eval_every'])
    model.train()
    win, t0 = [], time.time()
    while not done and k < steps:
        set_lrs(opt, k, steps, warmup, scale)
        pairs = bat.next(G)
        keys = prompt_keys(pairs, meta, prng, use_tpl)
        opt.zero_grad(set_to_none=True)
        loss = TB.train_step(model, D, data, pairs, keys, cw, micro, bool(tc.get('length_bucketing', False)))
        gn = TB.sync_and_clip(D, rep, shd, float(tc.get('grad_clip', 1.0)))
        opt.step()
        k += 1
        win.append((loss, gn))
        if k % int(tc['log_every']) == 0:
            el = time.time() - t0
            rec = {'type': 'train', 'stage': 'joint', 'step': k, 'loss': float(np.mean([x[0] for x in win])),
                   'grad_norm': float(np.mean([x[1] for x in win])), 'lr': {g['name']: g['lr'] for g in opt.param_groups},
                   'it_s': len(win) / el, 'samples_s': len(win) * G / el,
                   'max_mem_gb': torch.cuda.max_memory_allocated(D.device) / 2 ** 30}
            TB.log(D, json.dumps(rec))
            curves.append(rec)
            win, t0 = [], time.time()
        if k % every == 0 or k == steps:
            te = time.time()
            val = TB.llm_eval(model, D, data, meta, names, 'val', vpos, ebs)
            test = TB.llm_eval(model, D, data, meta, names, 'test', tpos, ebs)
            rec = {'type': 'eval', 'stage': 'joint', 'step': k, 'stage_step': k,
                   'val': {t: v['bacc'] for t, v in val.items()}, 'test': {t: v['bacc'] for t, v in test.items()},
                   'val_pred_hist': {t: v['pred_hist'] for t, v in val.items()}, 'eval_s': time.time() - te}
            rec['val_mean'] = float(np.mean(list(rec['val'].values())))
            rec['test_mean'] = float(np.mean(list(rec['test'].values())))
            TB.log(D, 'EVAL', json.dumps({x: rec[x] for x in ('step', 'val', 'test', 'val_mean', 'test_mean', 'eval_s')}))
            curves.append(rec)
            state = TB.get_state(model)
            if rec['val_mean'] > best.get('joint', {}).get('val_mean', -1):
                best['joint'] = {x: rec[x] for x in ('step', 'stage_step', 'val_mean', 'test_mean')}
                best['overall'] = {'stage': 'joint', **best['joint']}
                if D.rank == 0:
                    TB.atomic_save(state_dir / 'best_joint.pt', {'trainable': state, 'has_lora': True})
                bad = 0
            else:
                bad += 1
            done = k >= steps or (patience > 0 and bad >= patience)
            optimizer_state = collect_optimizer_state(model, opt)
            rng_states = collect_rng_states(D.device)
            if D.rank == 0:
                TB.atomic_save(state_dir / 'last.pt', {
                    'step': k, 'bad': bad, 'done': done, 'trainable': state,
                    'optimizer': optimizer_state, 'optimizer_format': optimizer_format(model),
                    'torch_rng': rng_states,
                    'batcher': bat.state(), 'prng': json.dumps(prng.bit_generator.state),
                    'curves': json.dumps(curves), 'best': json.dumps(best), 'world': D.world})
            del optimizer_state, rng_states
            D.barrier()
            if done and k < steps:
                TB.log(D, f'early stop at step {k} (patience {patience})')
            t0 = time.time()
    ck = torch.load(state_dir / 'best_joint.pt', weights_only=True, map_location='cpu')
    TB.set_state(model, ck['trainable'])
    allv = {t: list(range(len(data[t]['val']['labels']))) for t in names}
    allt = {t: list(range(len(data[t]['test']['labels']))) for t in names}
    val = TB.llm_eval(model, D, data, meta, names, 'val', allv, ebs)
    test = TB.llm_eval(model, D, data, meta, names, 'test', allt, ebs, gen_cap=int(ec.get('final_gen_cap_per_task', 500)),
                       gen_bs=ec.get('gen_batch_size'))
    final = {'joint': {'selected': best['joint'], 'per_task': {t: {'val': val[t], 'test': test[t]} for t in names},
                       'val_mean': float(np.mean([val[t]['bacc'] for t in names])),
                       'test_mean': float(np.mean([test[t]['bacc'] for t in names]))}}
    if use_tpl:
        tv = {t: capped_positions(len(data[t]['test']['labels']), ec.get('template_eval_cap', 1000), f'{t}-tpl') for t in names}
        per_tpl = {tpl: {t: r['bacc'] for t, r in TB.llm_eval(model, D, data, meta, names, 'test', tv, ebs, template=tpl).items()}
                   for tpl in range(min(num_templates(t) for t in names))}
        final['joint']['template_test'] = {t: {'per_template': [per_tpl[i][t] for i in sorted(per_tpl)],
                                               'mean': float(np.mean([per_tpl[i][t] for i in per_tpl])),
                                               'std': float(np.std([per_tpl[i][t] for i in per_tpl]))} for t in names}
    TB.log(D, 'FINAL joint', json.dumps({t: (round(val[t]['bacc'], 4), round(test[t]['bacc'], 4)) for t in names}))
    return {'mode': 'llm', 'design': '1020d-joint', 'llm_path': llm_path, 'world': D.world, 'fsdp': D.fsdp, 'global_batch': G,
            'micro_batch': micro, 'lr_scale': scale, 'class_weights': {t: v.tolist() for t, v in cw.items()},
            'sampling_p': bat.p, 'best': best, 'final': final, 'per_task': final['joint']['per_task'], 'curves': curves}


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
    dev = D.device if cfg['data'].get('feats_on_gpu', False) else torch.device('cpu')
    meta, data, stats = T20.load_seed(cfg, names, a.seed, dev)
    stats = {t: (m.to(D.device), s.to(D.device)) for t, (m, s) in stats.items()}
    tm = time.time()
    r = run_llm_d(cfg, D, data, meta, names, stats, out)
    if D.rank == 0:
        r.update({'seed': a.seed, 'minutes': (time.time() - tm) / 60,
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

