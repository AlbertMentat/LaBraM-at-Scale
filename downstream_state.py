"""Label statistics and durable validation selection for downstream training."""
import json
import math
from pathlib import Path


def training_class_counts(dataset, n_classes):
    """Count the labels actually returned by the selected training dataset."""
    counts = [0] * n_classes
    for index in range(len(dataset)):
        sample = dataset[index]
        label = sample['label'] if isinstance(sample, dict) else sample[1]
        if hasattr(label, 'item'):
            label = label.item()
        label = int(label)
        if not 0 <= label < n_classes:
            raise ValueError(f'Training label {label} is outside [0, {n_classes})')
        counts[label] += 1
    if not sum(counts):
        raise ValueError('Cannot compute class weights for an empty training split')
    return counts


def best_completed_epoch(output_dir, start_epoch, selection_metric):
    """Restore selection only from epochs included in the resumed checkpoint."""
    if not output_dir or start_epoch <= 0:
        return None
    root = Path(output_dir)
    records = {}
    log = root / 'log.txt'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A process interruption can leave the last log line incomplete.
                continue
            if isinstance(row, dict) and isinstance(row.get('epoch'), int):
                records[row['epoch']] = row
    for path in sorted((root / 'epoch_records').glob('epoch-*.json')):
        row = json.loads(path.read_text())
        if not isinstance(row.get('epoch'), int):
            raise ValueError(f'Invalid durable epoch record: {path}')
        records[row['epoch']] = row
    key = 'val_' + selection_metric
    best = None
    for epoch, row in sorted(records.items()):
        if not 0 <= epoch < start_epoch or key not in row:
            continue
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f'Invalid {key} at epoch {epoch}')
        if best is None or value > best['selection_value']:
            best = dict(epoch=epoch, selection_value=value,
                        validation={k[4:]: v for k, v in row.items()
                                    if k.startswith('val_') and k != 'val_loss'},
                        test={k[5:]: v for k, v in row.items()
                              if k.startswith('test_') and k != 'test_loss'})
    if best is None:
        summary = root / 'results.json'
        if summary.exists():
            row = json.loads(summary.read_text())
            epoch = row.get('best_epoch', -1)
            val = row.get('best_val', {})
            if (0 <= epoch < start_epoch and row.get('selection_metric') == selection_metric
                    and selection_metric in val):
                value = float(val[selection_metric])
                if math.isfinite(value):
                    best = dict(epoch=epoch, selection_value=value,
                                validation=val, test=row.get('best_test', {}))
    return best
