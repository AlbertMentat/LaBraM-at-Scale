"""Bind union feature caches to their extraction inputs and data index files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEEDS = (42, 10, 5)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cache_identity(cfg, task, spec) -> dict:
    """Use file contents, not index-directory names, as the data identity.

    Prompt wording and LLM size do not affect EEG features and stay outside the key.
    EEG file contents must remain immutable for a given sample index/path mapping.
    """
    root = Path(cfg['data']['index_root']) / spec.output_id
    ec = cfg['encoder']
    identity = {
        'schema_version': 1,
        'sample_index_sha256': file_sha256(root / 'sample_index.json'),
        'split_manifest_sha256': {
            str(seed): file_sha256(root / 'splits' / f'{spec.split_mode}_seed{seed}' / 'split_manifest.json')
            for seed in cfg.get('seeds', SEEDS)
        },
        'task': {
            'name': task, 'output_id': spec.output_id, 'split_mode': spec.split_mode,
            'indexed': spec.indexed, 'source_dataset_type': spec.source_dataset_type,
            'truncate_patches': spec.truncate_patches, 'num_classes': spec.num_classes,
        },
        'path_map': cfg['data'].get('path_map') or {},
        'encoder': {
            'checkpoint_sha256': ec['checkpoint_sha256'], 'model': ec.get('model'),
            'input_scale': float(ec['input_scale']), 'zscore': bool(ec.get('zscore', False)),
            'amp': ec.get('amp', 'fp16'),
        },
    }
    identity['sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    ).hexdigest()
    return identity


def require_cache_identity(record, expected, location) -> None:
    if not isinstance(record, dict) or record.get('cache_identity') != expected:
        raise ValueError(f'{location}: cache data identity differs or is missing; re-extract features for the current index and splits')


def require_unchanged_data(cfg, task, spec, expected) -> None:
    if cache_identity(cfg, task, spec) != expected:
        raise ValueError(f'{task}: index, splits or extraction settings changed while using the cache; retry extraction/loading')


def training_data_identity(cfg) -> dict:
    from eegllm.config import resolve_path
    from eegllm.data import load_task_specs
    specs = load_task_specs(resolve_path(cfg['tasks_config']), cfg['tasks'])
    return {task: cache_identity(cfg, task, specs[task]) for task in cfg['tasks']}


def require_run_data_identity(record, expected, location) -> None:
    if record.get('feature_cache_identities') != expected:
        raise ValueError(f'{location}: training data identity differs or is missing; use a new output directory for the current index and splits')
