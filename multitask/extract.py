#!/usr/bin/env python3
"""Build the channel-mean union feature cache for the selected EEG checkpoint."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--wait-merge', action='store_true')
    parser.add_argument('--wait-hours', type=float, default=36)
    parser.add_argument('--check-only', action='store_true', help='check indexed data and split manifests without extracting features')
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
        parser.error('require num-shards > 0 and 0 <= shard < num-shards')
    from eegllm.config import load_config
    from runtime import validate_config
    cfg = load_config(args.config, args.set)
    validate_config(cfg)
    name = 'extract_1020' if cfg['implementation'] == 'original' else 'extract_1020b'
    module = importlib.import_module(name)
    sys.argv = [module.__file__, '--config', args.config, '--shard', str(args.shard),
                '--num-shards', str(args.num_shards), '--wait-hours', str(args.wait_hours)]
    for override in args.set:
        sys.argv.extend(['--set', override])
    for option in ('merge', 'wait_merge', 'check_only'):
        if getattr(args, option):
            sys.argv.append('--' + option.replace('_', '-'))
    module.main()


if __name__ == '__main__':
    main()
