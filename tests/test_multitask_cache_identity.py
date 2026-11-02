"""CPU cache integration tests with synthetic EEG and a tiny synthetic encoder."""
import contextlib
import copy
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

MULTITASK = Path(__file__).resolve().parents[1] / 'multitask'
sys.path.insert(0, str(MULTITASK))
import extract_1020 as original_extract
import extract_1020b as grid_extract
import train_1020 as original_train
import train_1020b as grid_train
from cache_identity import cache_identity, training_data_identity, require_run_data_identity
from eegllm.data import load_task_specs


class TinyEncoder(torch.nn.Module):
    embed_dim = 2

    def forward_features(self, x, **kwargs):
        return torch.stack((x.mean(-1), x.square().mean(-1)), dim=-1).flatten(1, 2)


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = load_task_specs(str(MULTITASK / 'configs/tasks_omni23.yaml'), ['PD31'])['PD31']
        self.index = self.root / 'indices' / self.spec.output_id
        self.index.mkdir(parents=True)
        entries = []
        for i in range(6):
            path = self.root / f'raw{i}.pkl'
            with path.open('wb') as stream:
                pickle.dump({'X': np.full((1, 200), i + 1, dtype=np.float32),
                             'ch_names': ['Cz'], 'y': i % 2}, stream)
            entries.append({'path': str(path), 'label': i % 2, 'subject_id': f's{i}', 'ch_names': ['Cz']})
        (self.index / 'sample_index.json').write_text(json.dumps({'entries': entries}))
        for seed in (42, 10, 5):
            directory = self.index / 'splits' / f'{self.spec.split_mode}_seed{seed}'
            directory.mkdir(parents=True)
            (directory / 'split_manifest.json').write_text(json.dumps({
                'task_id': 'PD31', 'seed': seed,
                'splits': {'train': [0, 1], 'val': [2, 3], 'test': [4, 5]}}))
        self.cfg = {
            'encoder': {'model': 'labram_large_patch200_200', 'checkpoint': 'synthetic.safetensors',
                        'checkpoint_sha256': 'a' * 64, 'input_scale': 1., 'zscore': False, 'amp': 'fp16'},
            'data': {'index_root': str(self.root / 'indices'), 'path_map': None, 'num_workers': 0},
            'seeds': [42, 10, 5], 'tasks': ['PD31'],
            'tasks_config': str(MULTITASK / 'configs/tasks_omni23.yaml'),
            'feature_cache': {'root': str(self.root / 'features'), 'local_dir': None,
                              'batch_size': 2, 'wait_hours': 0},
        }

    def extract(self, module=grid_extract, shards=1):
        synthetic = types.ModuleType('eegllm.encoder')
        synthetic.build_encoder = lambda cfg: (TinyEncoder(), types.SimpleNamespace(get_input_chans=lambda ch: [1]),
                                               {'synthetic_encoder': True})
        with patch.dict(sys.modules, {'eegllm.encoder': synthetic}), \
             patch('torch.autocast', side_effect=lambda *a, **k: contextlib.nullcontext()):
            for shard in range(shards):
                module.extract_shard(self.cfg, ['PD31'], shard, shards, torch.device('cpu'))
        module.merge(self.cfg, ['PD31'], shards)
        return module.task_dir(self.cfg, 'PD31', self.spec)

    def test_extract_merge_and_both_training_loaders(self):
        for extractor in (original_extract, grid_extract):
            with self.subTest(extractor=extractor.__name__):
                self.cfg['feature_cache']['root'] = str(self.root / extractor.__name__)
                cache = self.extract(extractor, shards=2)
                self.assertTrue(extractor.cache_complete(self.cfg, 'PD31', self.spec))
                marker = json.loads((cache / 'COMPLETE.json').read_text())
                self.assertEqual(marker['cache_identity'], cache_identity(self.cfg, 'PD31', self.spec))
                a = original_train.load_seed(self.cfg, ['PD31'], 42, torch.device('cpu'))
                b = grid_train.load_seed_views(self.cfg, ['PD31'], 42)
                for split in ('train', 'val', 'test'):
                    self.assertTrue(torch.equal(a[1]['PD31'][split]['labels'], b[1]['PD31'][split]['labels']))
                    self.assertTrue(torch.equal(a[1]['PD31'][split]['feats'],
                        b[1]['PD31'][split]['feats'][torch.arange(2)]))
                for x, y in zip(a[2]['PD31'], b[2]['PD31']):
                    self.assertTrue(torch.equal(x, y))

    def test_index_replacement_at_the_same_path_invalidates_cache(self):
        old = self.extract()
        path = self.index / 'sample_index.json'
        doc = json.loads(path.read_text())
        doc['entries'].reverse()
        path.write_text(json.dumps(doc))
        self.assertNotEqual(old, grid_extract.task_dir(self.cfg, 'PD31', self.spec))
        self.assertFalse(grid_extract.cache_complete(self.cfg, 'PD31', self.spec))
        for loader in (lambda: grid_train.load_seed_views(self.cfg, ['PD31'], 42),
                       lambda: original_train.load_seed(self.cfg, ['PD31'], 42, torch.device('cpu'))):
            with self.assertRaises(SystemExit):
                loader()

    def test_any_seed_split_change_invalidates_cache(self):
        old = self.extract()
        path = self.index / 'splits' / f'{self.spec.split_mode}_seed10' / 'split_manifest.json'
        doc = json.loads(path.read_text())
        doc['splits']['train'].reverse()
        path.write_text(json.dumps(doc))
        self.assertNotEqual(old, grid_extract.task_dir(self.cfg, 'PD31', self.spec))
        self.assertFalse(original_extract.cache_complete(self.cfg, 'PD31', self.spec))

    def test_task_shape_and_data_mapping_are_part_of_identity(self):
        old = grid_extract.task_dir(self.cfg, 'PD31', self.spec)
        changed = copy.deepcopy(self.spec)
        changed.truncate_patches = 2
        self.assertNotEqual(old, grid_extract.task_dir(self.cfg, 'PD31', changed))
        cfg = copy.deepcopy(self.cfg)
        cfg['data']['path_map'] = {'/old': '/new'}
        self.assertNotEqual(old, grid_extract.task_dir(cfg, 'PD31', self.spec))

    def test_identical_index_relocation_and_llm_changes_reuse_cache(self):
        old = self.extract()
        relocated = self.root / 'relocated'
        shutil.copytree(self.root / 'indices', relocated)
        cfg = copy.deepcopy(self.cfg)
        cfg['data']['index_root'] = str(relocated)
        cfg['llm'] = {'model': 'another-Qwen-size'}
        self.assertEqual(old, grid_extract.task_dir(cfg, 'PD31', self.spec))
        self.assertTrue(original_extract.cache_complete(cfg, 'PD31', self.spec))
        changed = copy.deepcopy(self.spec)
        changed.instruction = 'Different prompt wording'
        self.assertEqual(old, grid_extract.task_dir(cfg, 'PD31', changed))

    def test_copied_stale_cache_is_rejected_by_both_loaders_and_merge(self):
        old = self.extract()
        path = self.index / 'sample_index.json'
        doc = json.loads(path.read_text())
        doc['entries'][0]['ch_names'] = ['Fz']
        path.write_text(json.dumps(doc))
        new = grid_extract.task_dir(self.cfg, 'PD31', self.spec)
        shutil.copytree(old, new)
        self.assertFalse(grid_extract.cache_complete(self.cfg, 'PD31', self.spec))
        for loader in (lambda: grid_train.load_seed_views(self.cfg, ['PD31'], 42),
                       lambda: original_train.load_seed(self.cfg, ['PD31'], 42, torch.device('cpu'))):
            with self.assertRaisesRegex(ValueError, 'cache data identity'):
                loader()
        for module in (original_extract, grid_extract):
            with self.assertRaisesRegex(SystemExit, 'shards not done'):
                module.merge(self.cfg, ['PD31'], 1)

    def test_unbound_legacy_markers_do_not_count_as_complete(self):
        cache = self.extract()
        for marker in (cache / 'COMPLETE.json', cache / 'shards/1-0/DONE.json'):
            doc = json.loads(marker.read_text())
            doc.pop('cache_identity')
            marker.write_text(json.dumps(doc))
        self.assertFalse(original_extract.cache_complete(self.cfg, 'PD31', self.spec))
        self.assertFalse(grid_extract.shard_complete(self.cfg, 'PD31', self.spec, 1, 0))
        with self.assertRaisesRegex(ValueError, 'cache data identity'):
            grid_train.load_seed_views(self.cfg, ['PD31'], 42)

    def test_mid_extraction_index_change_does_not_publish_done(self):
        real_atomic_save = grid_extract.atomic_save
        calls = []

        def mutate_on_validation(cfg, task, spec, expected):
            calls.append(None)
            if len(calls) == 2:
                path = self.index / 'sample_index.json'
                doc = json.loads(path.read_text())
                doc['entries'][0]['subject_id'] = 'updated'
                path.write_text(json.dumps(doc))
            from cache_identity import require_unchanged_data
            require_unchanged_data(cfg, task, spec, expected)

        synthetic = types.ModuleType('eegllm.encoder')
        synthetic.build_encoder = lambda cfg: (TinyEncoder(), types.SimpleNamespace(get_input_chans=lambda ch: [1]), {})
        with patch.dict(sys.modules, {'eegllm.encoder': synthetic}), \
             patch('torch.autocast', side_effect=lambda *a, **k: contextlib.nullcontext()), \
             patch.object(grid_extract, 'require_unchanged_data', side_effect=mutate_on_validation), \
             patch.object(grid_extract, 'atomic_save', wraps=real_atomic_save) as save:
            with self.assertRaisesRegex(ValueError, 'changed while using the cache'):
                grid_extract.extract_shard(self.cfg, ['PD31'], 0, 1, torch.device('cpu'))
            save.assert_not_called()

    def test_training_contract_rejects_replaced_indices(self):
        identities = training_data_identity(self.cfg)
        record = {'feature_cache_identities': identities}
        require_run_data_identity(record, identities, 'fixture')
        path = self.index / 'sample_index.json'
        doc = json.loads(path.read_text())
        doc['entries'][0]['label'] = 1
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'training data identity'):
            require_run_data_identity(record, training_data_identity(self.cfg), 'fixture')
        with self.assertRaisesRegex(ValueError, 'training data identity'):
            require_run_data_identity({}, identities, 'legacy fixture')

    def test_completed_public_training_runs_validate_data_before_skipping(self):
        import train_1020d_grid
        import train_1020d_r3
        for module in (train_1020d_r3, train_1020d_grid):
            with self.subTest(module=module.__name__):
                cfg = copy.deepcopy(self.cfg)
                cfg.update(output_root=str(self.root / module.__name__), run_name='fixture', seed=42)
                config = self.root / (module.__name__ + '.yaml')
                config.write_text(yaml.safe_dump(cfg))
                output = Path(cfg['output_root']) / 'fixture__seed42'
                output.mkdir(parents=True)
                record = {'config': cfg, 'seed': 42, 'feature_cache_identities': training_data_identity(cfg)}
                (output / 'run_contract.json').write_text(json.dumps(record))
                (output / 'result_llm.json').write_text('{}')
                local_rank = types.SimpleNamespace(rank=0, world=1, on=False,
                    device=torch.device('cpu'), barrier=lambda: None)
                with patch.object(module.TB, 'Dist', return_value=local_rank), \
                     patch.object(sys, 'argv', [module.__file__, '--config', str(config), '--seed', '42']):
                    module.main()
                    # The output remains present, but its recorded data identity is stale.
                    record['feature_cache_identities'] = {}
                    (output / 'run_contract.json').write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, 'training data identity'):
                        module.main()


if __name__ == '__main__':
    unittest.main()
