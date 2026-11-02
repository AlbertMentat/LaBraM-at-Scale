#!/usr/bin/env python3
"""Joint EEG projector + Qwen3 LoRA training and final full-split evaluation."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', required=True, type=int, choices=(42, 10, 5))
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--check-config', action='store_true', help='validate and print config without loading data or models')
    args = parser.parse_args()
    from eegllm.config import load_config
    from runtime import backend, validate_config
    cfg = load_config(args.config, args.set)
    cfg['seed'] = args.seed
    validate_config(cfg)
    if args.check_config:
        print(json.dumps(cfg, indent=2))
        return
    module = backend(cfg)
    sys.argv = [module.__file__, '--config', args.config, '--seed', str(args.seed)]
    for override in args.set:
        sys.argv.extend(['--set', override])
    module.main()


if __name__ == '__main__':
    main()
