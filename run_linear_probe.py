"""Linear probing of a pretrained LaBraM encoder on one downstream task.

Data layout (same as run_unified_finetuning.py, dataset type "downstream"):
    <data_dir>/train/*.pkl, <data_dir>/val/*.pkl, <data_dir>/test/*.pkl, <data_dir>/label_map.json
Each pkl holds {"X": float array [channels, time] at 200 Hz, "y": int label, "ch_names": [...]}.
label_map.json maps class names to the integer labels used in "y", e.g. {"class_0": 0, "class_1": 1}.

Readouts:
    pooled  - mean of all patch tokens (channels kept: those in the LaBraM 10-20 vocabulary)
    no_pool - all patch tokens flattened; channels aligned to the union of task channels in 10-20 order
Head: one linear layer, AdamW, 30 epochs, batch 128, gradient clip 0.8, evaluated on val every 5 epochs;
the epoch with the best validation balanced accuracy is reported on val and test.

Input normalisation must match the pretraining corpus:
    --input_norm zscore    per-channel z-score, clipped at +-15 (79k / LaBraM-Molecule)
    --input_norm scale100  signal / 100 (25k LaBraM-Quark and 49k LaBraM-Atom)
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch import nn

PATCH_SIZE = 200
EPOCHS = 30
BATCH_SIZE = 128
CLIP_GRAD = 0.8
EVAL_FREQ = 5
SPLITS = ('train', 'val', 'test')


def clean_channel(name):
    channel = str(name).strip().upper()
    if channel.startswith('EEG '):
        channel = channel[4:]
    for suffix in ('-REF', '-LE'):
        if channel.endswith(suffix):
            channel = channel[: -len(suffix)]
    return channel.split('-')[0].strip()


def load_split(data_dir, split, label_values):
    samples = []
    for path in sorted((data_dir / split).glob('*.pkl')):
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        signal = np.asarray(payload.get('X', payload.get('eeg')), dtype=np.float32)
        if signal.ndim == 3:
            signal = signal.reshape(signal.shape[0], -1)
        if signal.ndim != 2 or len(payload['ch_names']) != signal.shape[0]:
            raise ValueError(f'{path}: expected [channels, time] with matching ch_names')
        signal = signal[:, : (signal.shape[1] // PATCH_SIZE) * PATCH_SIZE]
        if not signal.shape[1]:
            raise ValueError(f'{path}: window is shorter than one patch')
        label = int(payload['y'] if 'y' in payload else payload['label'])
        if label not in label_values:
            raise ValueError(f'{path}: label {label} not in label_map.json')
        samples.append((signal, [clean_channel(c) for c in payload['ch_names']], label_values.index(label)))
    if not samples:
        raise ValueError(f'no samples in {data_dir / split}')
    return samples


def load_encoder(model_name, checkpoint, device):
    from timm.models import create_model
    import models.labram  # noqa: F401  (registers the LaBraM models)

    model = create_model(model_name, pretrained=False, num_classes=0, drop_rate=0.0, drop_path_rate=0.0,
                         attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                         use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    if Path(checkpoint).suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device='cpu')
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for key in ('model', 'module', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if any(key.startswith('student.') for key in state):
        state = {key[len('student.'):]: value for key, value in state.items() if key.startswith('student.')}
    target = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in target and v.shape == target[k].shape}
    coverage = sum(target[k].numel() for k in compatible) / sum(v.numel() for v in target.values())
    if coverage < 0.8:
        raise RuntimeError(f'checkpoint covers only {coverage:.1%} of {model_name}')
    model.load_state_dict(compatible, strict=False)
    return model.eval().to(device)


def normalise(tensor, mode):
    if mode == 'zscore':
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True) + 1e-7
        return ((tensor - mean) / std).clamp(-15, 15)
    return tensor / 100.0


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_probe(path, head, *, model, checkpoint, readout, input_norm, layout,
               label_map, best):
    """Save the selected head and the exact feature and label contract."""
    payload = {
        'format_version': 1,
        'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
        'model': model, 'encoder_sha256': checkpoint_sha256(checkpoint),
        'readout': readout, 'input_norm': input_norm, 'patch_size': PATCH_SIZE,
        'layout': None if layout is None else [list(layout[0]), int(layout[1])],
        'label_map': label_map, 'label_values': sorted(int(v) for v in label_map.values()),
        'best_epoch': best['epoch'], 'val': best['val'],
    }
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_probe(path, checkpoint, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1 or payload.get('patch_size') != PATCH_SIZE:
        raise ValueError('Unsupported linear-probe format')
    if checkpoint_sha256(checkpoint) != payload['encoder_sha256']:
        raise ValueError('Encoder checkpoint differs from the one used to train this probe')
    weight = payload['head']['weight']
    head = nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    head.load_state_dict(payload['head'], strict=True)
    return payload, head.eval()


@torch.inference_mode()
def extract(model, samples, readout, input_norm, device, batch_size, layout=None):
    from models.labram import get_input_chans, standard_1020

    vocab = {name.upper() for name in standard_1020}
    features = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        rows = []
        for signal, channels, _ in batch:
            if readout == 'pooled':
                keep = [i for i, name in enumerate(channels) if name in vocab]
                if not keep:
                    raise ValueError('No task channels match the LaBraM channel vocabulary')
                names = [channels[i] for i in keep]
                patches = signal.shape[-1] // PATCH_SIZE
                x = normalise(torch.from_numpy(signal[keep]).to(device).float(), input_norm)
                x = x.reshape(1, len(keep), patches, PATCH_SIZE)
                tokens = model.forward_features(x, input_chans=get_input_chans(names), return_all_tokens=True)
                rows.append(tokens[:, 1:].mean(dim=1).float().cpu())
            else:
                canonical, patches = layout
                if signal.shape[-1] != patches * PATCH_SIZE:
                    raise ValueError('Window length differs from the no-pool probe layout')
                aligned = np.zeros((len(canonical), patches * PATCH_SIZE), dtype=np.float32)
                position = {name: i for i, name in enumerate(canonical)}
                seen = set()
                for i, name in enumerate(channels):
                    if name in position and name not in seen:
                        aligned[position[name]] = signal[i, : patches * PATCH_SIZE]
                        seen.add(name)
                x = normalise(torch.from_numpy(aligned).to(device).float(), input_norm)
                x = x.reshape(1, len(canonical), patches, PATCH_SIZE)
                tokens = model.forward_features(x, input_chans=get_input_chans(list(canonical)),
                                                return_all_tokens=True)
                rows.append(tokens[:, 1:].flatten(1).float().cpu())
        features.append(torch.cat(rows).numpy())
        print(f'features {min(start + batch_size, len(samples))}/{len(samples)}', flush=True)
    return np.concatenate(features)


def metrics(labels, predictions):
    return {'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
            'accuracy': float(accuracy_score(labels, predictions)),
            'cohen_kappa': float(cohen_kappa_score(labels, predictions))}


def predict(head, features, device):
    head.eval()
    with torch.inference_mode():
        out = [head(torch.from_numpy(features[i:i + 4096]).float().to(device)).argmax(1).cpu().numpy()
               for i in range(0, len(features), 4096)]
    return np.concatenate(out)


def fit_probe(train, val, n_classes, lr, weight_decay, seed, device):
    (x_train, y_train), (x_val, y_val) = train, val
    torch.manual_seed(seed)
    head = nn.Linear(x_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = np.random.default_rng(seed)
    best, best_state = {'epoch': -1, 'val': {'balanced_accuracy': -1.0}}, None
    for epoch in range(EPOCHS):
        head.train()
        order = generator.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            x = torch.from_numpy(x_train[index]).float().to(device)
            y = torch.from_numpy(y_train[index]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(head(x), y).backward()
            nn.utils.clip_grad_norm_(head.parameters(), CLIP_GRAD)
            optimizer.step()
        if (epoch + 1) % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
            val_metrics = metrics(y_val, predict(head, x_val, device))
            print(f'epoch {epoch}: val {val_metrics}', flush=True)
            if val_metrics['balanced_accuracy'] > best['val']['balanced_accuracy']:
                best = {'epoch': epoch, 'val': val_metrics}
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return best, head


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='e.g. labram_base_patch200_200, labram_large_patch200_200, '
                        'labram_huge_patch200_200')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--readout', choices=('pooled', 'no_pool'), default='pooled')
    parser.add_argument('--input_norm', choices=('zscore', 'scale100'), required=True)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature_batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    label_map = json.loads((args.data_dir / 'label_map.json').read_text())
    label_values = sorted(int(v) for v in label_map.values())
    if not label_values or len(label_values) != len(set(label_values)):
        raise ValueError('label_map.json must contain distinct integer labels')
    data = {split: load_split(args.data_dir, split, label_values) for split in SPLITS}
    layout = None
    if args.readout == 'no_pool':
        from models.labram import standard_1020

        rank = {name.upper(): i for i, name in enumerate(standard_1020)}
        names = {c for split in SPLITS for _, channels, _ in data[split] for c in channels if c in rank}
        lengths = {s.shape[-1] for split in SPLITS for s, _, _ in data[split]}
        if len(lengths) != 1:
            raise ValueError(f'no_pool needs equal-length windows, found {sorted(lengths)}')
        layout = (tuple(sorted(names, key=rank.get)), lengths.pop() // PATCH_SIZE)
        if not layout[0]:
            raise ValueError('No task channels match the LaBraM channel vocabulary')
    encoder = load_encoder(args.model, args.checkpoint, device)
    arrays = {}
    for split in SPLITS:
        x = extract(encoder, data[split], args.readout, args.input_norm, device, args.feature_batch_size, layout)
        arrays[split] = (x, np.asarray([label for _, _, label in data[split]], dtype=np.int64))
    del encoder
    best, head = fit_probe(arrays['train'], arrays['val'], len(label_values), args.lr, args.weight_decay,
                           args.seed, device)
    result = {'model': args.model, 'checkpoint': str(args.checkpoint), 'data_dir': str(args.data_dir),
              'readout': args.readout, 'input_norm': args.input_norm, 'lr': args.lr,
              'weight_decay': args.weight_decay, 'seed': args.seed, 'best_epoch': best['epoch'],
              'val': best['val'], 'test': metrics(arrays['test'][1], predict(head, arrays['test'][0], device)),
              'samples': {split: len(data[split]) for split in SPLITS}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_probe(args.output_dir / 'probe-best.pth', head, model=args.model,
               checkpoint=args.checkpoint, readout=args.readout, input_norm=args.input_norm,
               layout=layout, label_map=label_map, best=best)
    result['probe_file'] = 'probe-best.pth'
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

