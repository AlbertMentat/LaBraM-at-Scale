"""Exercise the real launch scripts with an argv recorder instead of torchrun."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ('25k_base.sh', '25k_large.sh', '25k_huge.sh', '25k_tokenizer.sh',
           '25k_giant.sh', '25k_base_large.sh')


class QuarkPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='quark paths ', dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        recorder = self.bin / 'torchrun'
        recorder.write_text('#!' + sys.executable + '\n' +
                            'import json, os, sys\nfrom pathlib import Path\n'
                            'Path(os.environ["QUARK_TEST_RECORD"]).write_text('
                            'json.dumps({"cwd": os.getcwd(), "argv": sys.argv[1:]}))\n')
        recorder.chmod(0o755)
        self.record = self.root / 'argv.json'
        self.relative = str(self.root.relative_to(ROOT))

    def run_recipe(self, recipe, args=(), values=None):
        env = os.environ.copy()
        for key in ('DATASET_CONFIGS', 'TOKENIZER_WEIGHT', 'OUTPUT_DIR', 'RESUME',
                    'LOG_DIR', 'WORLD_SIZE', 'MODEL', 'HIP_VISIBLE_DEVICES'):
            env.pop(key, None)
        env.update(PATH=str(self.bin) + os.pathsep + env['PATH'], CUDA_VISIBLE_DEVICES='0',
                   NPROC='1', QUARK_TEST_RECORD=str(self.record),
                   OUTPUT_DIR=str(self.root / 'fallback output'))
        env.update(values or {})
        result = subprocess.run(['bash', str(ROOT / 'configs/recipes' / recipe), *args],
                                cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(self.record.read_text())

    @staticmethod
    def option(argv, name):
        result = None
        for i, arg in enumerate(argv):
            if arg == name:
                result = argv[i + 1]
            elif arg.startswith(name + '='):
                result = arg.partition('=')[2]
        return result

    def test_environment_paths_from_another_working_directory(self):
        paths = {key: self.relative + '/' + name for key, name in (
            ('DATASET_CONFIGS', 'data settings.json'), ('TOKENIZER_WEIGHT', 'tokenizer weights.safetensors'),
            ('OUTPUT_DIR', 'training output'), ('RESUME', 'resume weights.pth'))}
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                record = self.run_recipe(recipe, values=paths)
                for env, option in (('DATASET_CONFIGS', '--dataset_configs'),
                                    ('OUTPUT_DIR', '--output_dir'), ('RESUME', '--resume')):
                    self.assertEqual(self.option(record['argv'], option), str(ROOT / paths[env]))
                if recipe != '25k_tokenizer.sh':
                    self.assertEqual(self.option(record['argv'], '--tokenizer_weight'), str(ROOT / paths['TOKENIZER_WEIGHT']))
                expected_cwd = ROOT if recipe == '25k_giant.sh' else ROOT / 'legacy_25k'
                self.assertEqual(Path(record['cwd']), expected_cwd)
                self.assertTrue((ROOT / paths['OUTPUT_DIR']).is_dir())
                beta = record['argv'].index('--opt_betas')
                self.assertEqual(len(record['argv'][beta+1:beta+3]), 2)
                self.assertNotIn(' ', record['argv'][beta+1])

    def test_cli_paths_and_equals_syntax_preserve_spaces(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                dataset = self.relative + '/cli data.json'
                output = self.relative + '/cli output'
                resume = self.relative + '/cli checkpoint.pth'
                logs = self.relative + '/cli logs'
                record = self.run_recipe(recipe, ('--dataset_configs', dataset,
                    '--output_dir=' + output, '--resume', resume, '--log_dir=' + logs))
                for flag, path in (('--dataset_configs', dataset), ('--output_dir', output),
                                   ('--resume', resume), ('--log_dir', logs)):
                    self.assertEqual(self.option(record['argv'], flag), str(ROOT / path))
                self.assertTrue((ROOT / output).is_dir())

    def test_absolute_paths_are_not_rebased(self):
        path = str(self.root / 'absolute tokenizer.safetensors')
        record = self.run_recipe('25k_base.sh', values={'TOKENIZER_WEIGHT': path})
        self.assertEqual(self.option(record['argv'], '--tokenizer_weight'), path)
        self.assertEqual(self.option(record['argv'], '--dataset_configs'), 'configs/dataset_configs_example.json')

    def test_missing_path_value_fails_before_launch(self):
        result = subprocess.run(['bash', str(ROOT / 'configs/recipes/25k_base.sh'), '--output_dir'],
                                cwd=self.root, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing path', result.stderr)
        self.assertFalse(self.record.exists())


if __name__ == '__main__':
    unittest.main()
