"""Deterministic cross-epoch sharding helpers for persistent WDS workers."""

import torch


class SharedEpoch:
    """An epoch counter visible to persistent DataLoader worker processes."""

    def __init__(self, value=0):
        self._value = torch.tensor(int(value), dtype=torch.int64).share_memory_()

    def get(self):
        return int(self._value.item())

    def set(self, value):
        self._value.fill_(int(value))


def rotating_partition(items, part_count, part_index, epoch):
    """Partition disjointly while cycling every item through all parts."""
    part_count = int(part_count)
    part_index = int(part_index)
    epoch = int(epoch)
    if part_count <= 0:
        raise ValueError("part_count must be positive")
    if not 0 <= part_index < part_count:
        raise ValueError("part_index must be in [0, part_count)")
    values = list(items)
    start = (part_index - epoch) % part_count
    return values[start::part_count]
