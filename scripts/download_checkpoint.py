#!/usr/bin/env python3
"""Download a released encoder or tokenizer and verify its release SHA-256."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def load_entry(repo, revision, variant, model_id, tokenizer=False):
    """Read the release configuration, with support for older repository revisions."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    key = 'tokenizers' if tokenizer else 'models'
    try:
        config_path = hf_hub_download(repo, 'config.json', revision=revision)
    except LocalEntryNotFoundError:
        raise
    except EntryNotFoundError:
        prefix = '' if variant == 'default' else 'best-full-ft/'
        manifest_path = hf_hub_download(repo, prefix+'manifest.json', revision=revision)
        collection = json.loads(Path(manifest_path).read_text())
        entries = [dict(entry, file=prefix+entry['file'])
                   for entry in collection.get(key, [])]
    else:
        config = json.loads(Path(config_path).read_text())
        if config.get('schema_version') != 1:
            raise ValueError('Unsupported release configuration schema')
        entries = config['variants'][variant][key]
    for entry in entries:
        if entry['id'] == model_id:
            return entry
    raise ValueError(f'{model_id} is not available in the {variant} collection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--model', choices=[
        'quark-base', 'quark-large', 'quark-huge', 'quark-giant',
        'atom-base', 'atom-large', 'atom-huge',
        'molecule-base', 'molecule-large', 'molecule-huge'])
    selection.add_argument('--tokenizer', choices=['quark', 'atom', 'molecule'])
    parser.add_argument('--variant', choices=['default', 'best-full-ft'], default='default')
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--revision', default='main', help='Hugging Face branch or commit')
    parser.add_argument('--repo-id', default='Pan-Junyu/LaBraM-at-Scale',
                        help='Hugging Face model repository; override to inspect a private archive')
    args = parser.parse_args()
    if args.tokenizer and args.variant != 'default':
        parser.error('Tokenizers are available in the default collection only')
    from huggingface_hub import hf_hub_download
    repo = args.repo_id
    prefix = '' if args.variant == 'default' else 'best-full-ft/'
    model_id = args.tokenizer+'-tokenizer' if args.tokenizer else args.model
    entry = load_entry(repo, args.revision, args.variant, model_id, bool(args.tokenizer))
    source = Path(hf_hub_download(repo, entry['file'], revision=args.revision))
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != entry['export_sha256']:
        raise RuntimeError('Downloaded weight does not match the release SHA-256')
    target = args.output_dir / prefix / (model_id+'.safetensors')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    print(target)

if __name__ == '__main__':
    main()
