"""Portable optimizer state and per-rank RNG for joint EEG–LLM training."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist


def optimizer_format(model):
    return 'fsdp-full-v1' if getattr(model, 'fsdp_layers', None) else 'torch-v1'


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


@contextmanager
def _full_optimizer_state(model, rank0_only):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
    ):
        yield FSDP


def collect_optimizer_state(model, optimizer):
    """All ranks enter; rank 0 receives full CPU state, including the projector.

    Passing the whole model lets FSDP map both the replicated projector and
    the LoRA parameters inside separately wrapped decoder layers.
    """
    if optimizer_format(model) == 'fsdp-full-v1':
        with _full_optimizer_state(model, rank0_only=True) as fsdp:
            return fsdp.optim_state_dict(model, optimizer)
    if dist.is_initialized() and dist.get_rank() != 0:
        return {}
    return _cpu_copy(optimizer.state_dict())


def restore_optimizer_state(model, optimizer, state, saved_format=None, *, allow_empty=False):
    """Load full state on every rank and let FSDP reconstruct its local shards."""
    if not state:
        if allow_empty:
            return False
        raise ValueError(
            'This checkpoint has no optimizer state. Resume from a checkpoint that contains '
            'AdamW state, or set EEGLLM_ALLOW_OPTIMIZER_RESET=1 to explicitly restart its moments.'
        )
    current_format = optimizer_format(model)
    # Earlier non-FSDP checkpoints stored the ordinary PyTorch optimizer dict.
    saved_format = saved_format or 'torch-v1'
    if saved_format != current_format:
        raise ValueError(f'Optimizer checkpoint format {saved_format} does not match {current_format}')
    if current_format == 'fsdp-full-v1':
        # Every rank loads last.pt on CPU. rank0_only=False therefore consumes
        # a complete named state on each rank before flattening/sharding it.
        with _full_optimizer_state(model, rank0_only=False) as fsdp:
            state = fsdp.optim_state_dict_to_load(model, optimizer, state)
    optimizer.load_state_dict(state)
    return True


def collect_rng_states(device):
    """Capture dropout RNG after the completed update/evaluation on each rank."""
    state = {'cpu': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None}
    if dist.is_initialized() and dist.get_world_size() > 1:
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, state)
        return states
    return [state]


def restore_rng_states(states, device):
    if states is None:
        return False  # Older checkpoints contain only the NumPy samplers.
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(states) != world:
        raise ValueError(f'Checkpoint contains RNG states for {len(states)} ranks; resume with that world size')
    state = states[rank]
    if device.type == 'cuda' and state['cuda'] is None:
        raise ValueError('Checkpoint has no CUDA RNG state for the active device')
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)
    return True
