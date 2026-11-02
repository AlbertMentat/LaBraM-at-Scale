#!/usr/bin/env python3
"""Multitask EEG->Qwen3 training on a COMPLETED feature cache (1019b): 40 GB + multi-GPU version of train_1019.

Changes vs train_1019:
  * consumes per-task caches from extract_1019b.py (key = encoder sha + input contract + task split); never extracts
  * answer-position-only LM-head logits for loss and chunked closed-set scoring (eegllm/model_1019b.py)
  * gradient accumulation (train.micro_batch, per rank) with the identical sampling stream; optional LLM gradient
    checkpointing (llm.gradient_checkpointing); configurable eval / generation batch and scoring chunk sizes
  * torchrun DDP: every rank draws the same GLOBAL batch (train.batch_size) from the same seeded temperature sampler
    and takes its contiguous 1/world slice, so 1 GPU and N GPUs see identical samples; each slice's loss is
    normalised by the global class-weight sum and gradients are averaged (x world) -> identical to the 1-GPU gradient.
    LR scaling is explicit (train.lr_scaling none|linear|sqrt vs train.base_global_batch).  Eval/generation are
    sharded across ranks (padded to equal length) and gathered.  Rank 0 writes checkpoints; all ranks resume.
  * optional FSDP (dist.fsdp: true): each LLM decoder layer is an FSDP unit (use_orig_params, LoRA inside), for
    >=4B models; embeddings/lm_head/projector stay replicated and are averaged manually.
  * optional length bucketing of each rank's slice before micro-batching (train.length_bucketing).
Modes: lp | nollm | controls (single process) | llm (single or multi-GPU).
"""
from __future__ import annotations

import argparse
import contextlib
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
import torch.distributed as dist

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegllm.config import load_config, resolve_path  # noqa: E402
from eegllm.data import load_task_specs  # noqa: E402
from eegllm.metrics import balanced_accuracy  # noqa: E402
from eegllm.projector import EEGProjector  # noqa: E402
from eegllm.prompts import parse_generation  # noqa: E402
from eegllm.prompts_1019 import num_templates  # noqa: E402
from extract_1019b import task_dir  # noqa: E402
from train_1019 import (Batcher, capped_positions, class_weights, lr_at, make_batch, prompt_keys,  # noqa: E402,F401
                        run_lp, run_nollm, summarise)
from checkpoint_state import (collect_optimizer_state, collect_rng_states, optimizer_format,
                              restore_optimizer_state, restore_rng_states)  # noqa: E402

SPLITS = ('train', 'val', 'test')


# ============================================================================ dist helpers
class Dist:
    def __init__(self, cfg=None, world=None):
        self.world = int(os.environ.get('WORLD_SIZE', '1')) if world is None else world
        self.rank = int(os.environ.get('RANK', '0')) if world is None else 0
        self.local = int(os.environ.get('LOCAL_RANK', '0')) if world is None else 0
        dcfg = (cfg or {}).get('dist', {})
        self.fsdp = bool(dcfg.get('fsdp', False))
        same_gpu = os.environ.get('EEGLLM_SAME_GPU') == '1'  # tests: several ranks share one GPU (gloo)
        self.device = torch.device('cuda', 0 if same_gpu else self.local)
        torch.cuda.set_device(self.device)
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group(dcfg.get('backend', 'nccl'), timeout=dt.timedelta(minutes=90))

    @property
    def on(self):
        return self.world > 1

    def barrier(self):
        if self.on:
            dist.barrier()

    def gather(self, obj):
        if not self.on:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def all_sum(self, x: float) -> float:
        if not self.on:
            return x
        t = torch.tensor([x], dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return float(t.item())


def log(D, *a):
    if D is None or D.rank == 0:
        print(time.strftime('[%m-%d %H:%M:%S]'), *a, flush=True)


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def atomic_save(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================================================================ cache
def load_cache(cfg, names, device, stats_mode='task'):
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    meta, data, stats = {'tasks': {}}, {}, {}
    for t in names:
        tdir = task_dir(cfg, t, specs[t])
        comp = tdir / 'COMPLETE.json'
        if not comp.exists():
            raise SystemExit(f'feature cache for {t} is not complete: {tdir} (run extract_1019b.py first)')
        c = json.loads(comp.read_text())
        meta['tasks'][t] = {'classes': c['classes'], 'n_channels': c['n_channels'], 'n_patches': c['n_patches'],
                            'cache': str(tdir), 'files': c['files']}
        data[t] = {}
        for s in SPLITS:
            d = torch.load(tdir / f'{s}.pt', weights_only=True)
            if len(d['labels']) != c['counts'][s]:
                raise SystemExit(f'{t} {s}: cache count mismatch')
            data[t][s] = {'feats': d['feats'].to(device), 'labels': d['labels'].to(device), 'indices': d['indices']}
        st = torch.load(tdir / 'stats.pt', weights_only=True)
        stats[t] = (st['mean'].to(device), st['std'].to(device))
    if stats_mode == 'global':
        mus = torch.stack([stats[t][0] for t in names])
        var = torch.stack([stats[t][1] ** 2 for t in names]).mean(0) + mus.var(0, unbiased=False)
        stats = {t: (mus.mean(0), var.sqrt()) for t in names}
    return meta, data, stats


# ============================================================================ model
def build_llm(cfg, names, stats, D):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eegllm.model_1019b import EEGLLM1019b
    lc, pc, ec = cfg['llm'], cfg['projector'], cfg.get('eval', {})
    path = lc.get('path')
    if not (path and Path(path).is_dir()):
        from huggingface_hub import snapshot_download
        kw = dict(revision=lc.get('revision'), allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'LICENSE'])
        if D.local == 0:
            snapshot_download(lc['hf_id'], **kw)
        D.barrier()
        path = snapshot_download(lc['hf_id'], **kw)
    torch.manual_seed(cfg['seed'])  # identical init (projector, LoRA) on every rank
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[lc.get('dtype', 'bf16')]  # fp32 only for exactness tests
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=False,
                                              use_safetensors=True, attn_implementation=lc.get('attn', 'sdpa'))
    for p in lm.parameters():
        p.requires_grad_(False)
    rms = float(lm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
    proj = EEGProjector(400, lm.config.hidden_size, int(pc['num_tokens']), 'avg', 8, float(pc.get('mlp_ratio', 1.0)),
                        embed_rms=rms)
    specs = load_task_specs(resolve_path(cfg['tasks_config']), names)
    model = EEGLLM1019b(None, proj, lm, tok, specs, {}, int(pc['num_tokens']), 0.01, False, 'fp16', lc.get('dtype', 'bf16'))
    model.feat_stats = stats
    model.cand_rows = int(ec.get('cand_rows', 256))
    model.logit_chunk = int(ec.get('logit_chunk', 512))
    model.to(D.device)
    if lc.get('gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
    model.fsdp_layers = []
    if D.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        lora_stage = [s for s in cfg['train']['stages'] if s['name'] == 'lora']
        if lora_stage:  # inject before wrapping (lora_B = 0 -> no effect until the LoRA stage trains it)
            model.add_lora(lora_stage[0]['lora'])
        layers = model.causal_lm.model.layers
        for layer in layers:  # FSDP flattens one dtype per unit: LoRA weights follow the base dtype under FSDP
            layer.to(dtype)
        for i in range(len(layers)):
            layers[i] = FSDP(layers[i], use_orig_params=True, device_id=D.device, sync_module_states=False)
            model.fsdp_layers.append(layers[i])
    return model, path


@contextlib.contextmanager
def full_params(model, writeback=False):
    if not getattr(model, 'fsdp_layers', None):
        yield
        return
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with contextlib.ExitStack() as st:
        for layer in model.fsdp_layers:
            st.enter_context(FSDP.summon_full_params(layer, writeback=writeback, recurse=True))
        yield


def get_state(model):
    with full_params(model):
        return model.trainable_state()


def set_state(model, state):
    with full_params(model, writeback=True):
        model.load_trainable_state(state)


def zero_lora_b(model):
    with full_params(model, writeback=True), torch.no_grad():
        for n, p in model.causal_lm.named_parameters():
            if 'lora_B' in n:
                p.zero_()


def configure_stage(model, stage):
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
    sharded = set()
    for layer in getattr(model, 'fsdp_layers', []):
        sharded |= {id(p) for p in layer.parameters()}
    return params, [p for p in params if id(p) not in sharded], [p for p in params if id(p) in sharded]


def sync_and_clip(D, rep, shd, max_norm):
    """Average replicated grads across ranks (FSDP units average their own); global grad-norm clipping."""
    if D.on and rep:
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in rep]
        for p, g in zip(rep, grads):
            p.grad = g
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= D.world
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)
    sq = sum(float(p.grad.float().pow(2).sum()) for p in rep if p.grad is not None)
    if shd:
        sq += D.all_sum(sum(float(p.grad.float().pow(2).sum()) for p in shd if p.grad is not None))
    norm = math.sqrt(sq)
    coef = max_norm / (norm + 1e-6)
    if coef < 1:
        for p in rep + shd:
            if p.grad is not None:
                p.grad.mul_(coef)
    return norm


def train_step(model, D, data, pairs, keys, cw, micro, bucket=False):
    """`pairs`/`keys` = the GLOBAL batch (identical on every rank); this rank backpropagates its contiguous slice in
    micro-batches.  Returns the global class-weighted mean loss."""
    B = len(pairs) // D.world
    labels = [int(data[t]['train']['labels'][i]) for t, i in pairs]
    w_all = torch.stack([cw[t][y] for (t, _), y in zip(pairs, labels)]).to(D.device)
    denom = w_all.sum()
    order = list(range(D.rank * B, (D.rank + 1) * B))
    if bucket:
        order.sort(key=lambda j: len(model.prefix_ids[keys[j]]) + max(len(c) for c in model.cand_ids[keys[j]]))
    local = 0.0
    for s in range(0, B, micro):
        ix = order[s:s + micro]
        b = make_batch(data, 'train', [pairs[j] for j in ix], [keys[j] for j in ix])
        per = model.loss_per_sample(b, D.device)
        part = (per * w_all[ix]).sum() / denom
        (part * D.world).backward()  # grads are averaged across ranks -> world * mean == sum of parts
        local += part.item()
    return D.all_sum(local)


def accumulate_step(model, data, pairs, keys, cw, micro, dev):
    """Single-process gradient accumulation (used by tools/mem_1019b.py)."""
    D = Dist(world=1)
    D.device = dev
    return train_step(model, D, data, pairs, keys, cw, micro)


# ============================================================================ sharded evaluation
def _shard(pos_all, D):
    n_per = math.ceil(len(pos_all) / D.world)
    mine = pos_all[D.rank::D.world]
    real = len(mine)
    return mine + [mine[-1] if mine else pos_all[-1]] * (n_per - real), real  # equal work per rank (FSDP-safe)


def llm_eval(model, D, data, meta, names, split, positions, bs, template=None, gen_cap=0, gen_bs=None):
    model.eval()
    out = {}
    for t in names:
        pos_all = positions[t]
        k = len(meta['tasks'][t]['classes'])
        mine, real = _shard(pos_all, D)
        preds = []
        for s in range(0, len(mine), bs):
            pairs = [(t, i) for i in mine[s:s + bs]]
            keys = None if template is None else [(t, template, tuple(range(k)))] * len(pairs)
            preds += [int(torch.argmax(sc)) for sc in model.score(make_batch(data, split, pairs, keys), D.device)]
        got = {p: q for part in D.gather(list(zip(mine[:real], preds[:real]))) for p, q in part}
        r = summarise(data[t][split]['labels'][pos_all].cpu().numpy(), [got[p] for p in pos_all], k)
        if gen_cap:
            gpos = pos_all[:gen_cap]
            gm, greal = _shard(gpos, D)
            texts = []
            gbs = int(gen_bs or bs)
            for s in range(0, len(gm), gbs):
                b = make_batch(data, split, [(t, i) for i in gm[s:s + gbs]])
                texts += model.generate(b, D.device, 16) if not model.fsdp_layers else _gen_synced(model, b, D.device)
            gt = {p: x for part in D.gather(list(zip(gm[:greal], texts[:greal]))) for p, x in part}
            texts = [gt[p] for p in gpos]
            gp = [parse_generation(x, meta['tasks'][t]['classes']) for x in texts]
            r['gen'] = {'n': len(gpos), 'parse_rate': float(np.mean([p >= 0 for p in gp])),
                        'bacc': balanced_accuracy(data[t][split]['labels'][gpos].cpu().numpy(), gp), 'examples': texts[:3]}
        out[t] = r
    model.train()
    return out


@torch.no_grad()
def _gen_synced(model, batch, device, max_new_tokens=16):
    """Generation under FSDP: every rank must run the same number of forwards (synced_gpus)."""
    from eegllm.model import autocast
    eeg = model.encode(batch['groups'], len(batch['tasks']), device)
    with autocast(model.compute):
        emb, attn, _ = model.assemble(batch['tasks'], eeg, None, left_pad=True)
        out = model.causal_lm.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                                       temperature=None, top_p=None, top_k=None, synced_gpus=True,
                                       eos_token_id=[model.im_end_id, model.tok.eos_token_id], pad_token_id=model.pad_id)
    return model.tok.batch_decode(out, skip_special_tokens=False)


# ============================================================================ LLM training
def run_llm(cfg, D, data, meta, names, stats, out: Path):
    tc, ec = cfg['train'], cfg['eval']
    model, llm_path = build_llm(cfg, names, stats, D)
    G = int(tc['batch_size'])
    if G % D.world:
        raise SystemExit(f'global batch {G} not divisible by world {D.world}')
    micro = int(tc.get('micro_batch') or G // D.world)
    base = float(tc.get('base_global_batch', G))
    scale = {'none': 1.0, 'linear': G / base, 'sqrt': math.sqrt(G / base)}[tc.get('lr_scaling', 'none')]
    log(D, f'world {D.world} fsdp {D.fsdp} global batch {G} per-rank {G // D.world} micro {micro} lr x{scale:.3f}')
    use_tpl = bool(cfg.get('prompts', {}).get('multi_template', False))
    cw = {t: v.to(D.device) for t, v in class_weights(data, names, tc.get('class_balanced_tasks', [])).items()}
    bat = Batcher(data, names, float(cfg['sampling']['alpha']), cfg['seed'])
    prng = np.random.default_rng(cfg['seed'] + 99)
    vpos = {t: capped_positions(len(data[t]['val']['labels']), ec['val_cap_per_task'], f'{t}-val') for t in names}
    tpos = {t: capped_positions(len(data[t]['test']['labels']), ec['test_cap_per_task'], f'{t}-test') for t in names}
    ebs = int(ec['batch_size'])
    state_dir = out / 'state'
    if D.rank == 0:
        state_dir.mkdir(parents=True, exist_ok=True)
    D.barrier()
    st = torch.load(state_dir / 'last.pt', weights_only=True, map_location='cpu') if (state_dir / 'last.pt').exists() else None
    curves = json.loads(st['curves']) if st else []
    best = json.loads(st['best']) if st else {}
    gstep = st['gstep'] if st else 0
    if st:
        log(D, f'resuming stage {st["stage"]} step {st["stage_step"]}')
        bat.load(st['batcher'])
        prng.bit_generator.state = json.loads(st['prng'])
    for si, stage in enumerate(tc['stages']):
        if st and si < st['stage']:
            if stage['name'] == 'lora' and not model.has_lora:
                model.add_lora(stage['lora'])
            continue
        resuming = st is not None and si == st['stage']
        resume_rng = st.get('torch_rng') if resuming else None
        stage_boundary = resuming and st['stage_step'] == 0
        if stage_boundary and resume_rng is not None:
            # A new stage must initialize new LoRA tensors from the RNG saved
            # at the end of the previous stage, just as uninterrupted training.
            restore_rng_states(resume_rng, D.device)
        params, rep, shd = configure_stage(model, stage)
        lr = float(stage['lr']) * scale
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(stage.get('weight_decay', 0.0)), betas=(0.9, 0.98))
        k0, bad = 0, 0
        if resuming:
            if stage_boundary and resume_rng is not None:
                resume_rng = collect_rng_states(D.device)
            set_state(model, st['trainable'])
            restored = restore_optimizer_state(
                model, opt, st.get('optimizer'), st.get('optimizer_format'),
                allow_empty=st['stage_step'] == 0 or os.environ.get('EEGLLM_ALLOW_OPTIMIZER_RESET') == '1')
            if not restored and st['stage_step'] > 0:
                log(D, 'Optimizer history is absent; restarting AdamW moments as explicitly requested.')
            # Mid-stage reconstruction can initialize LoRA before its saved
            # tensors are loaded. Restore dropout RNG after that work.
            if not restore_rng_states(resume_rng, D.device):
                log(D, 'This older checkpoint has no saved PyTorch RNG state; dropout resumes from the current RNG.')
            k0, bad = st['stage_step'], st['bad']
            st = None
        steps = int(stage['steps'])
        model.train()
        win, t0 = [], time.time()
        k = k0
        while k < steps:
            for g in opt.param_groups:
                g['lr'] = lr_at(k, steps, int(stage.get('warmup', 200)), lr)
            pairs = bat.next(G)
            keys = prompt_keys(pairs, meta, prng, use_tpl)
            opt.zero_grad(set_to_none=True)
            loss = train_step(model, D, data, pairs, keys, cw, micro, bool(tc.get('length_bucketing', False)))
            gn = sync_and_clip(D, rep, shd, float(tc.get('grad_clip', 1.0)))
            opt.step()
            k += 1
            gstep += 1
            win.append((loss, gn))
            if gstep % int(tc['log_every']) == 0:
                el = time.time() - t0
                rec = {'type': 'train', 'stage': stage['name'], 'step': gstep, 'loss': float(np.mean([x[0] for x in win])),
                       'grad_norm': float(np.mean([x[1] for x in win])), 'it_s': len(win) / el,
                       'samples_s': len(win) * G / el, 'max_mem_gb': torch.cuda.max_memory_allocated(D.device) / 2 ** 30,
                       'max_reserved_gb': torch.cuda.max_memory_reserved(D.device) / 2 ** 30}
                log(D, json.dumps(rec))
                curves.append(rec)
                win, t0 = [], time.time()
            if k % int(tc['eval_every']) == 0 or k == steps:
                te = time.time()
                val = llm_eval(model, D, data, meta, names, 'val', vpos, ebs)
                test = llm_eval(model, D, data, meta, names, 'test', tpos, ebs)
                rec = {'type': 'eval', 'stage': stage['name'], 'step': gstep, 'stage_step': k,
                       'val': {t: v['bacc'] for t, v in val.items()}, 'test': {t: v['bacc'] for t, v in test.items()},
                       'val_pred_hist': {t: v['pred_hist'] for t, v in val.items()}, 'eval_s': time.time() - te}
                rec['val_mean'] = float(np.mean(list(rec['val'].values())))
                rec['test_mean'] = float(np.mean(list(rec['test'].values())))
                log(D, 'EVAL', json.dumps({x: rec[x] for x in ('stage', 'step', 'val', 'test', 'val_mean', 'test_mean', 'eval_s')}))
                curves.append(rec)
                state = get_state(model)
                if rec['val_mean'] > best.get(stage['name'], {}).get('val_mean', -1):
                    best[stage['name']] = {x: rec[x] for x in ('step', 'stage_step', 'val_mean', 'test_mean')}
                    if D.rank == 0:
                        atomic_save(state_dir / f'best_{stage["name"]}.pt', {'trainable': state, 'has_lora': model.has_lora})
                    bad = 0
                else:
                    bad += 1
                if rec['val_mean'] > best.get('overall', {}).get('val_mean', -1):
                    best['overall'] = {'stage': stage['name'], **best[stage['name']]}
                stop = bool(stage.get('patience')) and bad >= int(stage['patience'])
                done = k >= steps or stop
                # A completed stage starts its next optimizer from scratch.
                optimizer_state = {} if done else collect_optimizer_state(model, opt)
                rng_states = collect_rng_states(D.device)
                if D.rank == 0:
                    atomic_save(state_dir / 'last.pt', {
                        'stage': si + 1 if done else si, 'stage_step': 0 if done else k, 'gstep': gstep,
                        'bad': 0 if done else bad, 'trainable': state,
                        'optimizer': optimizer_state, 'optimizer_format': optimizer_format(model),
                        'torch_rng': rng_states,
                        'batcher': bat.state(), 'prng': json.dumps(prng.bit_generator.state),
                        'curves': json.dumps(curves), 'best': json.dumps(best), 'world': D.world})
                del optimizer_state, rng_states
                D.barrier()
                if stop:
                    log(D, f'early stop {stage["name"]} at stage step {k}')
                    break
                t0 = time.time()
    if cfg['eval'].get('skip_final', False):
        return {'mode': 'llm', 'world': D.world, 'best': best, 'curves': curves, 'per_task': {}}
    final = {}
    allv = {t: list(range(len(data[t]['val']['labels']))) for t in names}
    allt = {t: list(range(len(data[t]['test']['labels']))) for t in names}
    for which in ['align'] + (['lora'] if 'lora' in best else []):
        ck = torch.load(state_dir / f'best_{which}.pt', weights_only=True)
        set_state(model, ck['trainable'])
        if model.has_lora and not ck['has_lora']:
            zero_lora_b(model)
        val = llm_eval(model, D, data, meta, names, 'val', allv, ebs)
        test = llm_eval(model, D, data, meta, names, 'test', allt, ebs, gen_cap=int(ec.get('final_gen_cap_per_task', 500)),
                        gen_bs=ec.get('gen_batch_size'))
        final[which] = {'selected': best[which], 'per_task': {t: {'val': val[t], 'test': test[t]} for t in names},
                        'val_mean': float(np.mean([val[t]['bacc'] for t in names])),
                        'test_mean': float(np.mean([test[t]['bacc'] for t in names]))}
        if use_tpl:
            tv = {t: capped_positions(len(data[t]['test']['labels']), ec.get('template_eval_cap', 1000), f'{t}-tpl') for t in names}
            per_tpl = {tpl: {t: r['bacc'] for t, r in llm_eval(model, D, data, meta, names, 'test', tv, ebs, template=tpl).items()}
                       for tpl in range(min(num_templates(t) for t in names))}
            final[which]['template_test'] = {t: {'per_template': [per_tpl[i][t] for i in sorted(per_tpl)],
                                                 'mean': float(np.mean([per_tpl[i][t] for i in per_tpl])),
                                                 'std': float(np.std([per_tpl[i][t] for i in per_tpl]))} for t in names}
        log(D, 'FINAL', which, json.dumps({t: (round(val[t]['bacc'], 4), round(test[t]['bacc'], 4)) for t in names}))
    sel = best['overall']['stage']
    return {'mode': 'llm', 'llm_path': llm_path, 'world': D.world, 'fsdp': D.fsdp, 'global_batch': G, 'micro_batch': micro,
            'lr_scale': scale, 'best': best, 'final': final, 'per_task': final[sel]['per_task'],
            'lora_beats_projector_only_val': (final['lora']['val_mean'] > final['align']['val_mean']) if 'lora' in final else None,
            'curves': curves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', required=True, choices=['lp', 'nollm', 'llm', 'controls'])
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    D = Dist(cfg)
    np.random.seed(cfg['seed'])
    names = list(cfg['tasks'])
    out = Path(cfg['output_root']) / cfg['run_name']
    if D.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        contract = out / 'run_contract.json'
        if contract.exists():
            if json.loads(contract.read_text())['config'] != cfg:
                raise SystemExit(f'{out} belongs to a different config; refusing')
        else:
            atomic_json(contract, {'config': cfg, 'mode': args.mode, 'world': D.world,
                                   'created': dt.datetime.now(dt.timezone.utc).isoformat()})
    D.barrier()
    meta, data, stats = load_cache(cfg, names, D.device, cfg.get('feature_norm', {}).get('stats', 'task'))
    for mode in (['lp', 'nollm'] if args.mode == 'controls' else [args.mode]):
        rp = out / f'result_{mode}.json'
        if rp.exists():
            log(D, mode, 'already complete')
            continue
        tm = time.time()
        if mode == 'llm':
            r = run_llm(cfg, D, data, meta, names, stats, out)
        elif D.rank == 0:
            r = run_lp(data, meta, names, stats) if mode == 'lp' else run_nollm(cfg, data, meta, names, stats, out)
        else:
            continue
        if D.rank == 0:
            r.update({'minutes': (time.time() - tm) / 60, 'feature_cache': {t: meta['tasks'][t]['cache'] for t in names},
                      'n': {t: {s: int(len(data[t][s]['labels'])) for s in SPLITS} for t in names}})
            atomic_json(rp, r)
            if r.get('per_task'):
                log(D, 'DONE', mode, json.dumps({t: (round(v['val']['bacc'], 4), round(v['test']['bacc'], 4))
                                                 for t, v in r['per_task'].items()}), f'{r["minutes"]:.1f} min')
    D.barrier()
    if D.on:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
