#!/usr/bin/env python3
"""Evaluate a saved best_joint.pt on all validation and test samples, without training."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='new JSON path; default: RUN_DIR/reevaluation.json')
    parser.add_argument('--config', help='optional relocated config; only data/cache/model paths and hardware settings may change')
    args = parser.parse_args()
    out = args.output or args.run_dir / 'reevaluation.json'
    if out.exists():
        parser.error(f'Output already exists: {out}')
    contract = json.loads((args.run_dir / 'run_contract.json').read_text())
    cfg = contract['config']
    from runtime import validate_config
    validate_config(cfg)
    if args.config:
        from eegllm.config import load_config
        relocated = load_config(args.config)
        for key in ('tasks', 'projector', 'prompts', 'sampling', 'feature_norm', 'implementation'):
            if relocated[key] != cfg[key]:
                parser.error(f'Relocated config changes {key}')
        for key in ('checkpoint_sha256', 'model', 'input_scale', 'zscore', 'amp'):
            if relocated['encoder'][key] != cfg['encoder'][key]:
                parser.error(f'Relocated config changes encoder.{key}')
        if any(relocated['llm'].get(k) != cfg['llm'].get(k) for k in ('hf_id', 'revision', 'dtype', 'attn')):
            parser.error('Relocated config changes the LLM identity')
        if any(relocated['train'][k] != cfg['train'][k] for k in ('joint', 'stages')):
            parser.error('Relocated config changes the joint/LoRA recipe')
        cfg = relocated
    cfg['seed'] = contract['seed']
    from cache_identity import training_data_identity, require_run_data_identity
    require_run_data_identity(contract, training_data_identity(cfg), args.run_dir / 'run_contract.json')
    from runtime import backend, load_data
    backend(cfg)
    import numpy as np
    import torch
    import train_1019b as training
    from train_1019 import capped_positions
    from train_1020d_r3 import joint_param_groups
    from eegllm.prompts_1020 import num_templates
    distributed = training.Dist(cfg)
    names = cfg['tasks']
    meta, data, stats = load_data(cfg, distributed)
    model, llm_path = training.build_llm(cfg, names, stats, distributed)
    joint_param_groups(model, cfg['train']['joint'])
    checkpoint = args.run_dir / 'state/best_joint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    training.set_state(model, state['trainable'])
    ec = cfg['eval']
    scores = {}
    for split in ('val', 'test'):
        positions = {t: list(range(len(data[t][split]['labels']))) for t in names}
        scores[split] = training.llm_eval(model, distributed, data, meta, names, split, positions, ec['batch_size'],
            gen_cap=int(ec.get('final_gen_cap_per_task', 500)) if split == 'test' else 0,
            gen_bs=ec.get('gen_batch_size'))
    final = {'per_task': {t: {s: scores[s][t] for s in scores} for t in names},
             **{s+'_mean': float(np.mean([scores[s][t]['bacc'] for t in names])) for s in scores}}
    if cfg.get('prompts', {}).get('multi_template', False):
        positions = {t: capped_positions(len(data[t]['test']['labels']), ec.get('template_eval_cap', 1000), f'{t}-tpl') for t in names}
        templates = {i: training.llm_eval(model, distributed, data, meta, names, 'test', positions, ec['batch_size'], template=i)
                     for i in range(min(num_templates(t) for t in names))}
        final['template_test'] = {t: {'per_template': [templates[i][t]['bacc'] for i in templates],
            'mean': float(np.mean([templates[i][t]['bacc'] for i in templates])),
            'std': float(np.std([templates[i][t]['bacc'] for i in templates]))} for t in names}
    if distributed.rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        training.atomic_json(out, {'mode': 'llm', 'design': '1020d-joint', 'seed': cfg['seed'],
            'checkpoint': str(checkpoint), 'llm_path': llm_path, 'encoder': cfg['encoder'],
            'evaluation_only': True, 'final': {'joint': final}})
        print(out)
    distributed.barrier()
    if distributed.on:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
