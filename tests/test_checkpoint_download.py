"""Exercise checkpoint downloads without contacting the Hub or loading tensors."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch

from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'download_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('checkpoint_download_under_test', SCRIPT)
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = 'example/private-release'
        self.revision = 'pinned-release-commit'
        self.payload = b'small synthetic checkpoint fixture\n'
        self.weight = self.root / 'cached-weight.safetensors'
        self.weight.write_bytes(self.payload)
        digest = hashlib.sha256(self.payload).hexdigest()

        def entry(model_id, filename):
            return dict(id=model_id, file=filename, export_sha256=digest)

        self.model = entry('molecule-base', 'molecule/molecule-base.safetensors')
        self.selected = entry('atom-large', 'best-full-ft/atom/atom-large.safetensors')
        self.tokenizer = entry('quark-tokenizer', 'quark/quark-tokenizer.safetensors')
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({
            'schema_version': 1,
            'variants': {
                'default': {'models': [self.model], 'tokenizers': [self.tokenizer]},
                'best-full-ft': {'models': [self.selected], 'tokenizers': []},
            },
        }))
        self.paths = {'config.json': self.config}
        self.paths.update({entry['file']: self.weight
                           for entry in (self.model, self.selected, self.tokenizer)})
        self.failures = {}
        hub_patch = patch('huggingface_hub.hf_hub_download', side_effect=self.fetch)
        self.hub = hub_patch.start()
        self.addCleanup(hub_patch.stop)

    def fetch(self, repo, filename, revision):
        if filename in self.failures:
            raise self.failures[filename]
        return str(self.paths[filename])

    def run_cli(self, *args, output_dir=None):
        output_dir = output_dir or self.root / 'output'
        argv = [str(SCRIPT), *args, '--repo-id', self.repo, '--revision', self.revision,
                '--output-dir', str(output_dir)]
        stdout = io.StringIO()
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stdout):
            DOWNLOADER.main()
        return stdout.getvalue().strip()

    def assert_requests(self, *filenames):
        self.assertEqual(self.hub.call_args_list, [
            call(self.repo, filename, revision=self.revision) for filename in filenames
        ])

    def assert_download(self, args, entry, relative_target, *requests):
        target = self.root / 'output' / relative_target
        self.assertEqual(self.run_cli(*args), str(target))
        self.assertEqual(target.read_bytes(), self.payload)
        self.assert_requests(*requests, entry['file'])

    def test_default_model_reads_root_config_then_weight(self):
        self.assert_download(['--model', 'molecule-base'], self.model,
                             'molecule-base.safetensors', 'config.json')

    def test_selected_model_uses_repository_relative_weight_path(self):
        self.assert_download(['--model', 'atom-large', '--variant', 'best-full-ft'],
                             self.selected, 'best-full-ft/atom-large.safetensors',
                             'config.json')

    def test_tokenizer_reads_root_config_then_weight(self):
        self.assert_download(['--tokenizer', 'quark'], self.tokenizer,
                             'quark-tokenizer.safetensors', 'config.json')

    def test_remote_missing_config_falls_back_to_matching_old_manifest(self):
        self.failures['config.json'] = EntryNotFoundError('No config at this revision')
        cases = [
            (['--model', 'molecule-base'], self.model, 'manifest.json',
             'models', 'molecule-base.safetensors'),
            (['--model', 'atom-large', '--variant', 'best-full-ft'], self.selected,
             'best-full-ft/manifest.json', 'models', 'best-full-ft/atom-large.safetensors'),
            (['--tokenizer', 'quark'], self.tokenizer, 'manifest.json',
             'tokenizers', 'quark-tokenizer.safetensors'),
        ]
        for args, entry, manifest_name, key, target in cases:
            with self.subTest(args=args):
                self.hub.reset_mock()
                old_entry = dict(entry, file=entry['file'].removeprefix('best-full-ft/'))
                manifest = self.root / 'legacy-manifest.json'
                manifest.write_text(json.dumps({key: [old_entry]}))
                self.paths[manifest_name] = manifest
                self.assert_download(args, entry, target, 'config.json', manifest_name)

    def test_local_network_repository_and_revision_errors_do_not_fall_back(self):
        errors = [
            LocalEntryNotFoundError('Offline or uncached config'),
            RequestsConnectionError('Connection failed'),
            RepositoryNotFoundError('Repository inaccessible'),
            RevisionNotFoundError('Revision does not exist'),
            HfHubHTTPError('Server or authorization error'),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.hub.reset_mock()
                self.failures['config.json'] = error
                with self.assertRaises(type(error)) as raised:
                    self.run_cli('--model', 'molecule-base')
                self.assertIs(raised.exception, error)
                self.assert_requests('config.json')
                self.assertFalse((self.root / 'output').exists())

    def test_invalid_config_does_not_fall_back(self):
        cases = [
            ('{invalid json', json.JSONDecodeError),
            (json.dumps({'schema_version': 2}), ValueError),
            (json.dumps({'schema_version': 1, 'variants': {}}), KeyError),
        ]
        for content, error_type in cases:
            with self.subTest(content=content):
                self.hub.reset_mock()
                self.config.write_text(content)
                with self.assertRaises(error_type):
                    self.run_cli('--model', 'molecule-base')
                self.assert_requests('config.json')

    def test_missing_weight_does_not_trigger_manifest_fallback(self):
        error = EntryNotFoundError('Weight is missing')
        self.failures[self.model['file']] = error
        with self.assertRaises(EntryNotFoundError) as raised:
            self.run_cli('--model', 'molecule-base')
        self.assertIs(raised.exception, error)
        self.assert_requests('config.json', self.model['file'])

    def test_corrupted_download_neither_creates_nor_overwrites_target(self):
        self.weight.write_bytes(b'corrupted downloaded bytes')
        for already_exists in (False, True):
            with self.subTest(already_exists=already_exists):
                self.hub.reset_mock()
                output_dir = self.root / ('existing-output' if already_exists else 'new-output')
                target = output_dir / 'molecule-base.safetensors'
                if already_exists:
                    output_dir.mkdir()
                    target.write_bytes(b'previous verified checkpoint')
                with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                    self.run_cli('--model', 'molecule-base', output_dir=output_dir)
                self.assert_requests('config.json', self.model['file'])
                if already_exists:
                    self.assertEqual(target.read_bytes(), b'previous verified checkpoint')
                else:
                    self.assertFalse(output_dir.exists())


if __name__ == '__main__':
    unittest.main()
