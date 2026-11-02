"""Evaluate a saved LP head with its original encoder and feature layout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_linear_probe import extract, load_encoder, load_probe, load_split, metrics, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feature_batch_size', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    config, head = load_probe(args.probe, args.checkpoint, device)
    label_map = json.loads((args.data_dir / 'label_map.json').read_text())
    if label_map != config['label_map']:
        raise ValueError('Task label map differs from the saved probe')
    samples = load_split(args.data_dir, args.split, config['label_values'])
    encoder = load_encoder(config['model'], args.checkpoint, device)
    features = extract(encoder, samples, config['readout'], config['input_norm'], device,
                       args.feature_batch_size, config['layout'])
    predictions = predict(head, features, device)
    labels = np.asarray([label for _, _, label in samples], dtype=np.int64)
    result = {'split': args.split, 'samples': len(samples), 'metrics': metrics(labels, predictions),
              'predictions': [config['label_values'][int(i)] for i in predictions]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
