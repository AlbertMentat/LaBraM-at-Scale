import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from downstream_state import best_completed_epoch, training_class_counts
from run_linear_probe import load_probe, predict, save_probe


ROOT = Path(__file__).resolve().parents[1]


def finetuning_helpers():
    # Load the real storage helpers and dataset wrapper without optional GPU/logging imports.
    names = {'_json_safe', 'write_analysis_checkpoint', 'UnifiedFinetuneWrapper'}
    nodes = [node for node in ast.parse((ROOT / 'run_unified_finetuning.py').read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = dict(np=np, torch=torch, os=os, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'finetuning_helpers', 'exec'), namespace)
    return namespace


class DownstreamStateTests(unittest.TestCase):
    def test_counts_use_selected_remapped_labels(self):
        class Source(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, index):
                return {'eeg': torch.zeros(2, 400), 'label': [7, 2, 7, 2, 7][index]}

        wrapper = finetuning_helpers()['UnifiedFinetuneWrapper'](
            torch.utils.data.Subset(Source(), [0, 1, 2]), label_remap={7: 1, 2: 0}, num_classes=2)
        self.assertEqual(training_class_counts(wrapper, 2), [1, 2])
        with self.assertRaises(ValueError):
            training_class_counts([(None, 3)], 2)
        with self.assertRaises(ValueError):
            training_class_counts([], 2)

    def test_resume_uses_durable_history_ties_and_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'epoch_records').mkdir()
            rows = [dict(epoch=0, val_balanced_accuracy=0.8, test_accuracy=0.75),
                    dict(epoch=1, val_balanced_accuracy=0.8, test_accuracy=0.7),
                    dict(epoch=2, val_balanced_accuracy=0.99, test_accuracy=0.9)]
            (root / 'log.txt').write_text('\n'.join(json.dumps(dict(row, val_balanced_accuracy=0.1) if row['epoch'] == 0 else row) for row in rows) + '\n{"epoch":')
            # The atomic epoch record takes precedence over a conflicting log entry.
            (root / 'epoch_records/epoch-0000.json').write_text(json.dumps(rows[0]))
            best = best_completed_epoch(root, 2, 'balanced_accuracy')
            self.assertEqual(best['epoch'], 0)
            self.assertEqual(best['selection_value'], 0.8)
            self.assertEqual(best['test']['accuracy'], 0.75)
            self.assertIsNone(best_completed_epoch(root, 0, 'balanced_accuracy'))
            (root / 'log.txt').write_text('')
            (root / 'epoch_records/epoch-0000.json').unlink()
            (root / 'results.json').write_text(json.dumps(dict(
                best_epoch=1, selection_metric='balanced_accuracy',
                best_val={'balanced_accuracy': 0.85}, best_test={'accuracy': 0.81})))
            self.assertEqual(best_completed_epoch(root, 2, 'balanced_accuracy')['epoch'], 1)
            self.assertIsNone(best_completed_epoch(root, 2, 'accuracy'))

    def test_ft_best_checkpoint_reloads_the_selected_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(3, 2)
            features = torch.tensor([[1., 2., 3.], [-2., 1., 0.]])
            expected = model(features).detach().clone()
            finetuning_helpers()['write_analysis_checkpoint'](
                tmp, model, 4, 'balanced_accuracy', 0.8,
                {'balanced_accuracy': np.float64(0.8)}, {'accuracy': 0.7})
            with torch.no_grad():
                model.weight.add_(1)
            state = torch.load(Path(tmp) / 'checkpoint-best-analysis.pth', weights_only=True)
            model.load_state_dict(state['model'])
            self.assertTrue(torch.equal(model(features).detach(), expected))
            self.assertEqual(state['epoch'], 4)
            self.assertEqual(state['selection_value'], 0.8)

    def test_lp_head_reload_preserves_predictions_and_rejects_wrong_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = root / 'encoder.safetensors'
            encoder.write_bytes(b'encoder identity fixture')
            head = torch.nn.Linear(6, 2)
            x = np.random.default_rng(2).normal(size=(10, 6)).astype('float32')
            expected = predict(head, x, torch.device('cpu'))
            save_probe(root / 'probe-best.pth', head, model='fixture', checkpoint=encoder,
                       readout='no_pool', input_norm='scale100', layout=(('C3', 'C4'), 3),
                       label_map={'negative': 2, 'positive': 7},
                       best={'epoch': 9, 'val': {'balanced_accuracy': 0.8}})
            payload, restored = load_probe(root / 'probe-best.pth', encoder, torch.device('cpu'))
            np.testing.assert_array_equal(predict(restored, x, torch.device('cpu')), expected)
            self.assertEqual(payload['layout'], [['C3', 'C4'], 3])
            self.assertEqual(payload['label_values'], [2, 7])
            encoder.write_bytes(b'different encoder')
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_probe(root / 'probe-best.pth', encoder, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
