#!/usr/bin/env python3
"""Summarize completed runs by exact encoder, LLM, recipe, task and seed."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path, help='recursively find result_llm.json and adjacent run_contract.json')
    parser.add_argument('--output', required=True, type=Path, help='new output directory')
    args = parser.parse_args()
    groups, details, identities = {}, [], {}
    paths = sorted(args.results.rglob('result_llm.json'))
    if not paths:
        parser.error('No completed result_llm.json files found')
    for path in paths:
        raw = json.loads(path.read_text())
        contract = json.loads(path.with_name('run_contract.json').read_text())
        cfg = contract['config']
        seed = raw['seed']
        if seed not in (42, 10, 5) or seed != contract['seed'] or seed != cfg['seed']:
            raise ValueError(f'Seed mismatch: {path}')
        if raw['design'] != '1020d-joint' or raw['mode'] != 'llm':
            raise ValueError(f'Unexpected result design: {path}')
        tasks = raw['final']['joint']['per_task']
        if set(tasks) != set(cfg['tasks']):
            raise ValueError(f'Incomplete task suite: {path}')
        recipe = {k: cfg[k] for k in ('tasks', 'sampling', 'projector', 'prompts', 'train', 'feature_norm')}
        recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        identity = (cfg['encoder']['checkpoint_sha256'], cfg['llm']['hf_id'], cfg['llm']['revision'], recipe_sha)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (identity, seed) in identities:
            if identities[identity, seed] != digest:
                raise ValueError(f'Conflicting completed results for the same identity/seed: {path}')
            continue
        identities[identity, seed] = digest
        base = dict(checkpoint=cfg['encoder']['name'], checkpoint_sha256=identity[0],
                    llm=identity[1], llm_revision=identity[2], recipe_sha256=recipe_sha)
        for task, scores in tasks.items():
            row = dict(base, task=task, seed=seed, source=str(path), source_sha256=digest)
            for split in ('val', 'test'):
                value = scores[split]['bacc']
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid BAcc in {path}/{task}/{split}')
                row[split+'_bacc_pct'] = value * 100
            details.append(row)
        for task in [*cfg['tasks'], '__macro__']:
            group = groups.setdefault((identity, task), {'base': dict(base, task=task), 'seeds': {}})
            group['seeds'][seed] = {split: (tasks[task][split]['bacc'] if task != '__macro__' else
                statistics.mean(scores[split]['bacc'] for scores in tasks.values())) * 100 for split in ('val', 'test')}
        for split in ('val', 'test'):
            value = statistics.mean(scores[split]['bacc'] for scores in tasks.values())
            if not math.isclose(value, raw['final']['joint'][split+'_mean'], abs_tol=1e-9):
                raise ValueError(f'Macro mismatch: {path}')
    summaries = []
    for key in sorted(groups):
        group = groups[key];seeds = group['seeds']
        row = dict(group['base'], completed_seeds=len(seeds), complete_three_seed=len(seeds) == 3)
        for split in ('val', 'test'):
            values = [v[split] for v in seeds.values()]
            row[split+'_mean_pct'] = statistics.mean(values) if len(values) == 3 else ''
            row[split+'_sd_pct'] = statistics.stdev(values) if len(values) == 3 else ''
            for seed in (42, 10, 5):
                row[f'{split}_seed{seed}_pct'] = seeds.get(seed, {}).get(split, '')
        summaries.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('per_task_per_seed.csv', details), ('summary.csv', summaries)]:
        with (args.output/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(f'{len(identities)} unique complete runs; {len(details)} task/seed rows. SD: ddof=1; incomplete means blank.')


if __name__ == '__main__':
    main()
