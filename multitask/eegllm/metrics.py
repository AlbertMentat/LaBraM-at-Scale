"""Balanced accuracy and validation-only checkpoint selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float('nan')
    # sklearn ignores classes absent from y_true; unparseable generations (-1) count as wrong
    return float(balanced_accuracy_score(y_true, y_pred))


def closed_set_predictions(scores) -> list[int]:
    return [int(np.argmax(np.asarray(s))) for s in scores]


def mean_over_tasks(per_task: dict[str, float], tasks: list[str]) -> float:
    vals = [per_task[t] for t in tasks]
    return float(np.mean(vals))


def select_best(history: list[dict], key: str = 'val_mean_bacc') -> dict | None:
    """Pick the eval record with the highest mean validation BAcc; ties -> earliest step.
    Test metrics are never consulted."""
    best = None
    for rec in history:
        v = rec.get(key)
        if v is None or v != v:
            continue
        if best is None or v > best[key]:
            best = rec
    return best
