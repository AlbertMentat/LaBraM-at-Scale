#!/usr/bin/env python3
"""Full-data multitask pilot (1019) on cached frozen Atom-L features.

Pipeline (one job = one mode):
  1. features: frozen encoder over the FULL Omni23 seed-42 splits (FT input contract), stored as the channel-mean
     patch tokens [N, A, D] (fp16) + exact per-task per-dim mean/std over all TRAIN patch tokens.  Because
     standardisation is per-dim affine, mean_c((f-mu)/sd) == (mean_c f - mu)/sd, so the avg-pooling projector
     sees exactly what it would see on full token tensors.  Shared cache on blob (COMPLETE.json marker), else
     extracted locally and published.
  2. modes: lp (LogReg per task, C by val) | nollm (avg projector + multitask heads) | llm (Qwen3; projector-only
     stage then projector+LoRA with early stopping).  Per-stage best by mean val BAcc (val subset), test never used
     for selection; final full val/test for the projector-only best and for the overall best.
Resumable (state/last.pt); outputs under <output_root>/<run_name>/.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegllm.config import load_config, resolve_path  # noqa: E402
from eegllm.data import SampleDataset, build_tasks, collate, load_task_specs, task_probabilities  # noqa: E402
from eegllm.metrics import balanced_accuracy  # noqa: E402
from eegllm.projector import EEGProjector  # noqa: E402
from eegllm.prompts import parse_generation  # noqa: E402
from eegllm.prompts_1019 import num_templates  # noqa: E402

SPLITS = ('train', 'val', 'test')


def log(*a):
    print(time.strftime('[%m-%d %H:%M:%S]'), *a, flush=True)


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def atomic_save(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================================================================ features
class ShardItems(torch.utils.data.Dataset):
    """One item = one shard file of an indexed task: reads it once and returns all needed samples."""

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


def extract_features(cfg, tasks_names, out_dir: Path, device):
    from einops import rearrange
    from eegllm.encoder import build_encoder
    specs = load_task_specs(resolve_path(cfg['tasks_config']), tasks_names)
    tasks = build_tasks(cfg, specs)
    enc, lu, rep = build_encoder(cfg['encoder'])
    enc.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {'encoder': rep, 'tasks': {}, 'created': dt.datetime.now(dt.timezone.utc).isoformat(),
            'layout': 'channel-mean patch tokens [N, A, D] fp16; stats over all train patch tokens'}
    nw = int(cfg['data'].get('num_workers', 12))
    for t, td in tasks.items():
        chans = torch.tensor(lu.get_input_chans(td.raw_ch_names), device=device)
        C, A = len(td.canonical_ch), td.canonical_n
        meta['tasks'][t] = {'n_channels': C, 'n_patches': A, 'classes': specs[t].classes,
                            'split_sizes': {s: len(td.splits[s]) for s in SPLITS}}
        s1 = s2 = None
        ntok = 0
        for split in SPLITS:
            idx = td.splits[split]
            sharded = td.entries[idx[0]].get('item_index') is not None
            t0 = time.time()
            feats, labels, order = [], [], []

            def run(xs, ys, ii):
                nonlocal s1, s2, ntok
                x = torch.from_numpy(np.stack(xs)).to(device).float() * float(cfg['encoder']['input_scale'])
                x = rearrange(x, 'B C (A T) -> B C A T', T=200)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    f = enc.forward_features(x, input_chans=chans, return_patch_tokens=True)
                f = f.float()
                if split == 'train':
                    fd = f.reshape(-1, f.shape[-1]).double()
                    s1 = fd.sum(0) if s1 is None else s1 + fd.sum(0)
                    s2 = (fd * fd).sum(0) if s2 is None else s2 + (fd * fd).sum(0)
                    ntok += fd.shape[0]
                feats.append(f.view(f.shape[0], C, A, -1).mean(1).half().cpu())
                labels.extend(ys)
                order.extend(ii)

            if sharded:
                dl = torch.utils.data.DataLoader(ShardItems(td, idx), batch_size=1, num_workers=nw,
                                                 collate_fn=lambda b: b[0])
                buf = []
                for items in dl:
                    buf.extend(items)
                    while len(buf) >= 128:
                        chunk, buf = buf[:128], buf[128:]
                        run([c[1] for c in chunk], [c[2] for c in chunk], [c[0] for c in chunk])
                if buf:
                    run([c[1] for c in buf], [c[2] for c in buf], [c[0] for c in buf])
            else:
                dl = torch.utils.data.DataLoader(SampleDataset(tasks, [(t, i) for i in idx]), batch_size=128,
                                                 num_workers=nw, collate_fn=collate)
                for b in dl:
                    run(list(b['groups'][t]['eeg'].numpy()), b['labels'].tolist(), b['indices'])
            pos = {i: k for k, i in enumerate(order)}
            perm = torch.tensor([pos[i] for i in idx])  # back to manifest order
            obj = {'feats': torch.cat(feats)[perm], 'labels': torch.tensor(labels)[perm], 'indices': torch.tensor(idx)}
            atomic_save(out_dir / f'{t}_{split}.pt', obj)
            log(f'features {t} {split}: {len(idx)} samples in {time.time()-t0:.0f}s')
        mu = s1 / ntok
        sd = (s2 / ntok - mu * mu).clamp(min=1e-8).sqrt()
        atomic_save(out_dir / f'{t}_stats.pt', {'mean': mu.float().cpu(), 'std': (sd.float() + 1e-4).cpu(),
                                                'n_tokens': ntok})
    atomic_json(out_dir / 'meta.json', meta)
    del enc
    torch.cuda.empty_cache()


def ensure_features(cfg, names, device) -> Path:
    shared = Path(cfg['feature_cache']['shared_dir'])
    if (shared / 'COMPLETE.json').exists() and all((shared / f'{t}_test.pt').exists() for t in names):
        log('using shared feature cache', shared)
        return shared
    local = Path(cfg['feature_cache']['local_dir'])
    if not ((local / 'meta.json').exists() and all((local / f'{t}_stats.pt').exists() for t in names)):
        extract_features(cfg, names, local, device)
    if cfg['feature_cache'].get('publish', True) and not (shared / 'COMPLETE.json').exists():
        try:
            shared.mkdir(parents=True, exist_ok=True)
            for p in sorted(local.glob('*')):
                tmp = shared / (p.name + f'.tmp{os.getpid()}')
                shutil.copyfile(p, tmp)
                os.replace(tmp, shared / p.name)
            atomic_json(shared / 'COMPLETE.json', {'tasks': names, 'published': dt.datetime.now(dt.timezone.utc).isoformat(),
                                                    'by': os.environ.get('EEGLLM_RUN_ID', 'local')})
            log('published feature cache to', shared)
        except OSError as e:
            log('publish failed (continuing with local cache):', e)
    return local


def load_features(cache: Path, names, device, stats_mode='task'):
    meta = json.loads((cache / 'meta.json').read_text())
    data, stats = {}, {}
    for t in names:
        data[t] = {}
        for s in SPLITS:
            d = torch.load(cache / f'{t}_{s}.pt', weights_only=True)
            data[t][s] = {'feats': d['feats'].to(device), 'labels': d['labels'].to(device), 'indices': d['indices']}
        st = torch.load(cache / f'{t}_stats.pt', weights_only=True)
        stats[t] = (st['mean'].to(device), st['std'].to(device), st['n_tokens'])
    if stats_mode == 'global':
        mus = torch.stack([stats[t][0] for t in names])
        var = torch.stack([stats[t][1] ** 2 for t in names]).mean(0) + mus.var(0, unbiased=False)
        stats = {t: (mus.mean(0), var.sqrt(), 0) for t in names}
    return meta, data, {t: (v[0], v[1]) for t, v in stats.items()}


# ============================================================================ batching / metrics
def summarise(y, p, n_cls):
    labels = list(range(n_cls)) + ([-1] if len(p) and min(p) < 0 else [])
    return {'bacc': balanced_accuracy(y, p), 'n': int(len(y)), 'chance': 1.0 / n_cls,
            'confusion': confusion_matrix(y, p, labels=labels).tolist(),
            'pred_hist': np.bincount(np.asarray(p)[np.asarray(p) >= 0], minlength=n_cls).tolist()}


class Batcher:
    def __init__(self, data, names, alpha, seed):
        self.names = names
        self.sizes = {t: len(data[t]['train']['labels']) for t in names}
        self.p = task_probabilities(self.sizes, 'temperature', alpha)
        self.rng = np.random.default_rng(seed)
        self.perm = {t: self.rng.permutation(self.sizes[t]) for t in names}
        self.pos = {t: 0 for t in names}

    def next(self, bs):
        draws = self.rng.choice(len(self.names), size=bs, p=[self.p[t] for t in self.names])
        out = []
        for d in draws:
            t = self.names[d]
            if self.pos[t] >= self.sizes[t]:
                self.perm[t] = self.rng.permutation(self.sizes[t])
                self.pos[t] = 0
            out.append((t, int(self.perm[t][self.pos[t]])))
            self.pos[t] += 1
        return out

    def state(self):
        return {'rng': json.dumps(self.rng.bit_generator.state), 'pos': dict(self.pos),
                'perm': {t: torch.from_numpy(np.asarray(v)) for t, v in self.perm.items()}}

    def load(self, st):
        self.rng.bit_generator.state = json.loads(st['rng'])
        self.pos = dict(st['pos'])
        self.perm = {t: v.numpy() for t, v in st['perm'].items()}


def make_batch(data, split, pairs, keys=None):
    groups = {}
    for pos, (t, i) in enumerate(pairs):
        g = groups.setdefault(t, {'pos': [], 'ix': []})
        g['pos'].append(pos)
        g['ix'].append(i)
    labels = torch.zeros(len(pairs), dtype=torch.long)
    for t, g in groups.items():
        ix = torch.tensor(g.pop('ix'), device=data[t][split]['feats'].device)
        g['feats'] = data[t][split]['feats'][ix]
        g['shape'] = (1, g['feats'].shape[1])
        labels[g['pos']] = data[t][split]['labels'][ix].cpu()
    return {'tasks': keys if keys is not None else [t for t, _ in pairs], 'labels': labels, 'groups': groups}


def capped_positions(n, cap, tag):
    if cap is None or n <= cap:
        return list(range(n))
    return sorted(random.Random(f'42-{tag}').sample(range(n), cap))


def lr_at(step, total, warmup, base, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def class_weights(data, names, cb_tasks):
    w = {}
    for t in names:
        y = data[t]['train']['labels'].cpu().numpy()
        k = int(y.max()) + 1
        cnt = np.bincount(y, minlength=k).astype(float)
        w[t] = torch.tensor(len(y) / (k * np.maximum(cnt, 1)) if t in cb_tasks else np.ones(k), dtype=torch.float32)
    return w


# ============================================================================ (1) LP
def run_lp(data, meta, names, stats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    for t in names:
        mu, sd = stats[t]
        X = {s: ((data[t][s]['feats'].float() - mu) / sd).mean(1).cpu().numpy() for s in SPLITS}
        Y = {s: data[t][s]['labels'].cpu().numpy() for s in SPLITS}
        sc = StandardScaler().fit(X['train'])
        best = None
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(sc.transform(X['train']), Y['train'])
            v = balanced_accuracy(Y['val'], clf.predict(sc.transform(X['val'])))
            log(f'LP {t} C={C} val {v:.4f}')
            if best is None or v > best[0]:
                best = (v, C, clf)
        k = len(meta['tasks'][t]['classes'])
        res[t] = {'C': best[1], **{s: summarise(Y[s], best[2].predict(sc.transform(X[s])), k) for s in ('val', 'test')}}
    return {'mode': 'lp', 'per_task': res}


# ============================================================================ (2) no-LLM
class NoLLM(nn.Module):
    def __init__(self, meta, names, K, width):
        super().__init__()
        self.projector = EEGProjector(400, width, K, 'avg', 8, 1.0, embed_rms=1.0)
        self.heads = nn.ModuleDict({t: nn.Linear(width, len(meta['tasks'][t]['classes'])) for t in names})
        self.stats = {}

    def logits(self, t, feats):
        mu, sd = self.stats[t]
        tok = self.projector((feats.float() - mu) / sd, 1, feats.shape[1])
        return self.heads[t](tok.mean(1))


def nollm_predict(model, data, t, split, positions, bs=1024):
    preds = []
    with torch.no_grad():
        for s in range(0, len(positions), bs):
            ix = torch.tensor(positions[s:s + bs], device=data[t][split]['feats'].device)
            preds += model.logits(t, data[t][split]['feats'][ix]).argmax(-1).tolist()
    return preds


def run_nollm(cfg, data, meta, names, stats, out):
    dev = data[names[0]]['train']['feats'].device
    tc = cfg['train']
    model = NoLLM(meta, names, int(cfg['projector']['num_tokens']), 1024).to(dev)
    model.stats = stats
    cw = {t: v.to(dev) for t, v in class_weights(data, names, tc.get('class_balanced_tasks', [])).items()}
    steps = int(tc['nollm_steps'])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc['nollm_lr']), weight_decay=0.01)
    bat = Batcher(data, names, float(cfg['sampling']['alpha']), cfg['seed'])
    vpos = {t: capped_positions(len(data[t]['val']['labels']), cfg['eval']['val_cap_per_task'], f'{t}-val') for t in names}
    curves, best, best_state = [], None, None
    for step in range(steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, 200, float(tc['nollm_lr']))
        pairs = bat.next(int(tc['batch_size']))
        b = make_batch(data, 'train', pairs)
        num = den = 0
        for t, g in b['groups'].items():
            y = b['labels'][g['pos']].to(dev)
            w = cw[t][y]
            num = num + (F.cross_entropy(model.logits(t, g['feats']), y, reduction='none') * w).sum()
            den = den + w.sum()
        loss = num / den
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % int(tc['eval_every']) == 0 or step + 1 == steps:
            model.eval()
            rec = {'type': 'eval', 'step': step + 1, 'loss': loss.item(),
                   'val': {t: balanced_accuracy(data[t]['val']['labels'][vpos[t]].cpu().numpy(),
                                                nollm_predict(model, data, t, 'val', vpos[t])) for t in names}}
            model.train()
            rec['val_mean'] = float(np.mean(list(rec['val'].values())))
            curves.append(rec)
            log('nollm', json.dumps(rec))
            if best is None or rec['val_mean'] > best['val_mean']:
                best, best_state = rec, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    res = {}
    for t in names:
        k = len(meta['tasks'][t]['classes'])
        res[t] = {s: summarise(data[t][s]['labels'].cpu().numpy(),
                               nollm_predict(model, data, t, s, list(range(len(data[t][s]['labels'])))), k)
                  for s in ('val', 'test')}
    return {'mode': 'nollm', 'best_step': best['step'], 'per_task': res, 'curves': curves}


# ============================================================================ (3/4/5) LLM
def build_llm(cfg, names, stats, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eegllm.model_1019 import EEGLLM1019
    lc = cfg['llm']
    path = lc.get('path')
    if not (path and Path(path).is_dir()):
        from huggingface_hub import snapshot_download
        path = snapshot_download(lc['hf_id'], revision=lc.get('revision'),
                                 allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'LICENSE'])
    log('LLM path', path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=False,
                                              use_safetensors=True, attn_implementation='sdpa')
    for p in lm.parameters():
        p.requires_grad_(False)
    rms = float(lm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
    pc = cfg['projector']
    proj = EEGProjector(400, lm.config.hidden_size, int(pc['num_tokens']), 'avg', 8, float(pc.get('mlp_ratio', 1.0)),
                        embed_rms=rms)
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    model = EEGLLM1019(None, proj, lm, tok, specs, {}, int(pc['num_tokens']), 0.01, False, 'fp16', 'bf16')
    model.feat_stats = stats
    return model.to(dev), path


def prompt_keys(pairs, meta, rng, use_templates):
    if not use_templates:
        return [t for t, _ in pairs]
    keys = []
    for t, _ in pairs:
        k = len(meta['tasks'][t]['classes'])
        keys.append((t, int(rng.integers(num_templates(t))), tuple(int(i) for i in rng.permutation(k))))
    return keys


def llm_eval(model, data, meta, names, split, positions, bs, template=None, gen_cap=0):
    dev = next(model.projector.parameters()).device
    model.eval()
    out = {}
    for t in names:
        pos = positions[t]
        k = len(meta['tasks'][t]['classes'])
        preds = []
        for s in range(0, len(pos), bs):
            pairs = [(t, i) for i in pos[s:s + bs]]
            keys = None if template is None else [(t, template, tuple(range(k)))] * len(pairs)
            preds += [int(torch.argmax(sc)) for sc in model.score(make_batch(data, split, pairs, keys), dev)]
        r = summarise(data[t][split]['labels'][pos].cpu().numpy(), preds, k)
        if gen_cap:
            gpos = pos[:gen_cap]
            texts = []
            for s in range(0, len(gpos), bs):
                texts += model.generate(make_batch(data, split, [(t, i) for i in gpos[s:s + bs]]), dev, 16)
            gp = [parse_generation(x, meta['tasks'][t]['classes']) for x in texts]
            r['gen'] = {'n': len(gpos), 'parse_rate': float(np.mean([p >= 0 for p in gp])),
                        'bacc': balanced_accuracy(data[t][split]['labels'][gpos].cpu().numpy(), gp),
                        'examples': texts[:3]}
        out[t] = r
    model.train()
    return out


def run_llm(cfg, data, meta, names, stats, out: Path):
    dev = data[names[0]]['train']['feats'].device
    tc, ec = cfg['train'], cfg['eval']
    model, llm_path = build_llm(cfg, names, stats, dev)
    use_tpl = bool(cfg.get('prompts', {}).get('multi_template', False))
    cw = {t: v.to(dev) for t, v in class_weights(data, names, tc.get('class_balanced_tasks', [])).items()}
    bat = Batcher(data, names, float(cfg['sampling']['alpha']), cfg['seed'])
    prng = np.random.default_rng(cfg['seed'] + 99)
    vpos = {t: capped_positions(len(data[t]['val']['labels']), ec['val_cap_per_task'], f'{t}-val') for t in names}
    tpos = {t: capped_positions(len(data[t]['test']['labels']), ec['test_cap_per_task'], f'{t}-test') for t in names}
    state_dir = out / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)
    stages = tc['stages']
    st = None
    if (state_dir / 'last.pt').exists():
        st = torch.load(state_dir / 'last.pt', weights_only=True)
        log(f'resuming stage {st["stage"]} step {st["stage_step"]}')
    curves = json.loads(st['curves']) if st else []
    best = json.loads(st['best']) if st else {}
    gstep = st['gstep'] if st else 0
    if st:
        bat.load(st['batcher'])
        prng.bit_generator.state = json.loads(st['prng'])
    for si, stage in enumerate(stages):
        if st and si < st['stage']:
            if stage['name'] == 'lora' and not model.has_lora:
                model.add_lora(stage['lora'])
            continue
        if stage['name'] == 'lora' and not model.has_lora:
            model.add_lora(stage['lora'])
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.projector.parameters():
            p.requires_grad_(True)
        if stage['name'] == 'lora':
            for n, p in model.causal_lm.named_parameters():
                if 'lora_' in n:
                    p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(stage['lr']), weight_decay=float(stage.get('weight_decay', 0.0)),
                                betas=(0.9, 0.98))
        k0, bad = 0, 0
        if st and si == st['stage']:
            model.load_trainable_state(st['trainable'])
            if st['optimizer']:
                opt.load_state_dict(st['optimizer'])
            k0, bad = st['stage_step'], st['bad']
            st = None
        steps = int(stage['steps'])
        model.train()
        win, t0 = [], time.time()
        k = k0
        while k < steps:
            for g in opt.param_groups:
                g['lr'] = lr_at(k, steps, int(stage.get('warmup', 200)), float(stage['lr']))
            pairs = bat.next(int(tc['batch_size']))
            keys = prompt_keys(pairs, meta, prng, use_tpl)
            b = make_batch(data, 'train', pairs, keys)
            per = model.loss_per_sample(b, dev)
            w = torch.stack([cw[t][int(y)] for (t, _), y in zip(pairs, b['labels'])]).to(dev)
            loss = (per * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            k += 1
            gstep += 1
            win.append((loss.item(), gn))
            if gstep % int(tc['log_every']) == 0:
                rec = {'type': 'train', 'stage': stage['name'], 'step': gstep, 'loss': float(np.mean([x[0] for x in win])),
                       'grad_norm': float(np.mean([x[1] for x in win])), 'it_s': len(win) / (time.time() - t0)}
                log(json.dumps(rec))
                curves.append(rec)
                win, t0 = [], time.time()
            if k % int(tc['eval_every']) == 0 or k == steps:
                te = time.time()
                val = llm_eval(model, data, meta, names, 'val', vpos, int(ec['batch_size']))
                test = llm_eval(model, data, meta, names, 'test', tpos, int(ec['batch_size']))
                rec = {'type': 'eval', 'stage': stage['name'], 'step': gstep, 'stage_step': k,
                       'val': {t: v['bacc'] for t, v in val.items()}, 'test': {t: v['bacc'] for t, v in test.items()},
                       'val_pred_hist': {t: v['pred_hist'] for t, v in val.items()}, 'eval_s': time.time() - te}
                rec['val_mean'] = float(np.mean(list(rec['val'].values())))
                rec['test_mean'] = float(np.mean(list(rec['test'].values())))
                log('EVAL', json.dumps({x: rec[x] for x in ('stage', 'step', 'val', 'test', 'val_mean', 'test_mean', 'eval_s')}))
                curves.append(rec)
                improved_stage = rec['val_mean'] > best.get(stage['name'], {}).get('val_mean', -1)
                if improved_stage:
                    best[stage['name']] = {x: rec[x] for x in ('step', 'stage_step', 'val_mean', 'test_mean')}
                    atomic_save(state_dir / f'best_{stage["name"]}.pt',
                                {'trainable': model.trainable_state(), 'has_lora': model.has_lora})
                    bad = 0
                else:
                    bad += 1
                if rec['val_mean'] > best.get('overall', {}).get('val_mean', -1):
                    best['overall'] = {'stage': stage['name'], **best[stage['name']]}
                stop = stage.get('patience') and bad >= int(stage['patience'])
                atomic_save(state_dir / 'last.pt', {
                    'stage': si if (k < steps and not stop) else si + 1,
                    'stage_step': k if (k < steps and not stop) else 0, 'gstep': gstep, 'bad': bad if not stop else 0,
                    'trainable': model.trainable_state(), 'optimizer': opt.state_dict() if (k < steps and not stop) else {},
                    'batcher': bat.state(), 'prng': json.dumps(prng.bit_generator.state),
                    'curves': json.dumps(curves), 'best': json.dumps(best)})
                if stop:
                    log(f'early stop {stage["name"]} at stage step {k} (patience {stage["patience"]})')
                    break
                t0 = time.time()
    # ---------------------------------------------------------------- final evaluation (full val/test)
    final = {}
    allv = {t: list(range(len(data[t]['val']['labels']))) for t in names}
    allt = {t: list(range(len(data[t]['test']['labels']))) for t in names}
    for which in ['align'] + (['lora'] if 'lora' in best else []):
        ck = torch.load(state_dir / f'best_{which}.pt', weights_only=True)
        model.load_trainable_state(ck['trainable'])
        if model.has_lora and not ck['has_lora']:
            with torch.no_grad():
                for n, p in model.causal_lm.named_parameters():
                    if 'lora_B' in n:
                        p.zero_()
        log('final eval', which)
        val = llm_eval(model, data, meta, names, 'val', allv, int(ec['batch_size']))
        test = llm_eval(model, data, meta, names, 'test', allt, int(ec['batch_size']),
                        gen_cap=int(ec.get('final_gen_cap_per_task', 500)))
        final[which] = {'selected': best[which], 'per_task': {t: {'val': val[t], 'test': test[t]} for t in names},
                        'val_mean': float(np.mean([val[t]['bacc'] for t in names])),
                        'test_mean': float(np.mean([test[t]['bacc'] for t in names]))}
        if use_tpl:
            per_tpl = {}
            tv = {t: capped_positions(len(data[t]['test']['labels']), ec.get('template_eval_cap', 1000), f'{t}-tpl') for t in names}
            for tpl in range(min(num_templates(t) for t in names)):
                r = llm_eval(model, data, meta, names, 'test', tv, int(ec['batch_size']), template=tpl)
                per_tpl[tpl] = {t: r[t]['bacc'] for t in names}
            final[which]['template_test'] = {
                t: {'per_template': [per_tpl[i][t] for i in sorted(per_tpl)],
                    'mean': float(np.mean([per_tpl[i][t] for i in per_tpl])),
                    'std': float(np.std([per_tpl[i][t] for i in per_tpl]))} for t in names}
        log('FINAL', which, json.dumps({t: (round(val[t]['bacc'], 4), round(test[t]['bacc'], 4)) for t in names}))
    sel = best['overall']['stage']
    return {'mode': 'llm', 'llm_path': llm_path, 'selection': 'per-stage best by mean val BAcc on val subset '
            f'(cap {ec["val_cap_per_task"]}); overall = {sel}', 'best': best, 'final': final,
            'per_task': final[sel]['per_task'], 'lora_beats_projector_only_val': (
                final['lora']['val_mean'] > final['align']['val_mean']) if 'lora' in final else None,
            'curves': curves}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', required=True, choices=['features', 'lp', 'nollm', 'llm', 'controls'])
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dev = torch.device('cuda', 0)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    names = list(cfg['tasks'])
    out = Path(cfg['output_root']) / cfg['run_name']
    out.mkdir(parents=True, exist_ok=True)
    contract = out / 'run_contract.json'
    if contract.exists():
        if json.loads(contract.read_text())['config'] != cfg:
            raise SystemExit(f'{out} belongs to a different config; refusing')
    else:
        atomic_json(contract, {'config': cfg, 'mode': args.mode, 'created': dt.datetime.now(dt.timezone.utc).isoformat()})
    t0 = time.time()
    if args.mode == 'llm':  # fail fast on HF hub access before spending time on features
        build_llm_check = cfg['llm'].get('path')
        if not (build_llm_check and Path(build_llm_check).is_dir()):
            from huggingface_hub import snapshot_download
            p = snapshot_download(cfg['llm']['hf_id'], revision=cfg['llm'].get('revision'),
                                  allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'LICENSE'])
            log('HF hub OK:', p, sorted(os.listdir(p)))
    cache = ensure_features(cfg, names, dev)
    if args.mode == 'features':
        return
    meta, data, stats = load_features(cache, names, dev, cfg.get('feature_norm', {}).get('stats', 'task'))
    results = {}
    for mode in (['lp', 'nollm'] if args.mode == 'controls' else [args.mode]):
        rp = out / f'result_{mode}.json'
        if rp.exists():
            log(mode, 'already complete')
            continue
        tm = time.time()
        if mode == 'lp':
            r = run_lp(data, meta, names, stats)
        elif mode == 'nollm':
            r = run_nollm(cfg, data, meta, names, stats, out)
        else:
            r = run_llm(cfg, data, meta, names, stats, out)
        r.update({'minutes': (time.time() - tm) / 60, 'feature_cache': str(cache),
                  'n': {t: {s: int(len(data[t][s]['labels'])) for s in SPLITS} for t in names}})
        atomic_json(rp, r)
        results[mode] = r
        log('DONE', mode, json.dumps({t: (round(v['val']['bacc'], 4), round(v['test']['bacc'], 4))
                                      for t, v in r['per_task'].items()}), f'{r["minutes"]:.1f} min')
    log(f'total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
