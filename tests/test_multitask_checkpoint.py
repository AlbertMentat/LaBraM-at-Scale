"""Resume equivalence for projector/LoRA AdamW state and per-rank dropout RNG.

Run the optional two-rank test with EEGLLM_TEST_DISTRIBUTED=1. Set
EEGLLM_TEST_DEVICE=cuda for two CUDA GPUs; the default uses CPU/Gloo.
"""
from contextlib import contextmanager
import ast
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multitask'))
from checkpoint_state import (
    collect_optimizer_state, collect_rng_states, optimizer_format,
    restore_optimizer_state, restore_rng_states,
)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(6, 6)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(6, 2, bias=False)
        self.lora_B = nn.Linear(2, 6, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x)))


class ToyModel(nn.Module):
    def __init__(self, device, sharded=False):
        super().__init__()
        torch.manual_seed(17)
        self.projector = nn.Linear(4, 6)
        self.layers = nn.ModuleList([Decoder(), Decoder()])
        self.to(device)
        self.fsdp_layers = []
        if sharded:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            for i in range(len(self.layers)):
                self.layers[i] = FSDP(self.layers[i], device_id=device, use_orig_params=True)
                self.fsdp_layers.append(self.layers[i])

    def forward(self, x):
        x = self.projector(x)
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class DelayedLoRAModel(nn.Module):
    """Exercise the production stage loop's lazy add_lora() reconstruction."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.projector = nn.Linear(4, 6)
        self.causal_lm = nn.Module()
        self.causal_lm.base = nn.Linear(6, 6)
        self.dropout = nn.Dropout(0.25)
        self.has_lora = False
        self.fsdp_layers = []

    def add_lora(self, config):
        self.causal_lm.lora_A = nn.Linear(6, 2, bias=False)
        self.causal_lm.lora_B = nn.Linear(2, 6, bias=False)
        nn.init.zeros_(self.causal_lm.lora_B.weight)
        self.has_lora = True

    def forward(self, x):
        x = self.projector(x)
        result = self.causal_lm.base(x)
        if self.has_lora:
            result = result + self.causal_lm.lora_B(self.causal_lm.lora_A(self.dropout(x)))
        return result

    def trainable_state(self):
        return {name: p.detach().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, tensor in state.items():
                parameters[name].copy_(tensor)


def run_real_stage_loop_until(output, stop_at):
    """Run the actual stage/checkpoint loop with a tiny CPU model and task."""
    class CheckpointWritten(Exception):
        pass

    class LocalRank:
        world, rank, on, device = 1, 0, False, torch.device('cpu')
        fsdp = False

        def barrier(self):
            path = output / 'state/last.pt'
            if path.exists() and torch.load(path, weights_only=True)['gstep'] == stop_at:
                raise CheckpointWritten

    class TaskBatcher:
        def __init__(self, *args):
            self.cursor = 0

        def next(self, count):
            self.cursor += count
            return [('task', i % 2) for i in range(count)]

        def state(self):
            return {'cursor': self.cursor}

        def load(self, state):
            self.cursor = state['cursor']

    def train_step(model, *args):
        x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
        loss = (model(x) - 0.4).square().mean()
        loss.backward()
        return float(loss.detach())

    def evaluate(*args, **kwargs):
        torch.rand(3)  # checkpoint RNG must follow all evaluation work
        return {'task': {'bacc': 0.6, 'pred_hist': [1, 1]}}

    cfg = {
        'train': {'batch_size': 2, 'eval_every': 1, 'log_every': 1000,
                  'stages': [{'name': 'align', 'lr': 1e-2, 'steps': 2, 'warmup': 0},
                             {'name': 'lora', 'lr': 1e-2, 'steps': 4, 'warmup': 0, 'lora': {}}]},
        'eval': {'batch_size': 2, 'val_cap_per_task': 2, 'test_cap_per_task': 2, 'skip_final': True},
        'sampling': {'alpha': 0.3}, 'seed': 42,
    }
    data = {'task': {split: {'labels': torch.tensor([0, 1])} for split in ('train', 'val', 'test')}}
    namespace = dict(torch=torch, np=np, os=os, Path=Path, json=json, math=math, time=time,
                     contextlib=contextlib, dist=dist, Batcher=TaskBatcher,
                     build_llm=lambda *args: (DelayedLoRAModel(), 'fixture'),
                     log=lambda *args: None, train_step=train_step, llm_eval=evaluate,
                     class_weights=lambda *args: {'task': torch.ones(2)},
                     capped_positions=lambda n, *args: list(range(n)),
                     prompt_keys=lambda pairs, meta, rng, tpl: [int(rng.integers(100)) for _ in pairs],
                     collect_optimizer_state=collect_optimizer_state, collect_rng_states=collect_rng_states,
                     optimizer_format=optimizer_format, restore_optimizer_state=restore_optimizer_state,
                     restore_rng_states=restore_rng_states)
    root = Path(__file__).resolve().parents[1] / 'multitask'
    needed = {'run_llm', 'configure_stage', 'set_state', 'get_state', 'full_params',
              'atomic_save', 'sync_and_clip'}
    nodes = [node for node in ast.parse((root / 'train_1019b.py').read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in needed]
    nodes += [node for node in ast.parse((root / 'train_1019.py').read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'lr_at']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'real_stage_loop', 'exec'), namespace)
    try:
        namespace['run_llm'](cfg, LocalRank(), data, {}, ['task'], {}, output)
    except CheckpointWritten:
        return torch.load(output / 'state/last.pt', weights_only=True, map_location='cpu')
    raise AssertionError('Training did not reach the requested checkpoint')


def make_optimizer(model):
    return torch.optim.AdamW([
        {'name': 'projector', 'base_lr': 3e-3, 'lr': 3e-3,
         'params': list(model.projector.parameters()), 'weight_decay': 0.0},
        {'name': 'lora', 'base_lr': 2e-3, 'lr': 2e-3,
         'params': [p for p in model.layers.parameters() if p.requires_grad], 'weight_decay': 0.01},
    ], betas=(0.9, 0.98))


def update(model, optimizer, device, rank=0, world=1):
    x = torch.arange(20, device=device, dtype=torch.float32).reshape(5, 4) / 20 + rank * 0.1
    y = torch.full((5, 6), 0.4, device=device)
    optimizer.zero_grad(set_to_none=True)
    (model(x) - y).square().mean().backward()
    # The production model also averages replicated projector gradients manually.
    if world > 1:
        for p in model.projector.parameters():
            dist.all_reduce(p.grad)
            p.grad.div_(world)
    optimizer.step()


def assert_nested_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            assert_nested_equal(x, y)
    else:
        assert actual == expected


@contextmanager
def full_model_state(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        yield


def distributed_worker(rank, root, device_type):
    device = torch.device(device_type, rank) if device_type == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                            init_method='file://' + str(Path(root) / 'rendezvous'), rank=rank, world_size=2)
    torch.set_num_threads(1)
    try:
        model = ToyModel(device, sharded=True)
        optimizer = make_optimizer(model)
        torch.manual_seed(91 + rank)
        for _ in range(3):
            update(model, optimizer, device, rank, 2)
        with full_model_state(model):
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state = collect_optimizer_state(model, optimizer)
        rng = collect_rng_states(device)
        if rank == 0:
            assert any(k.startswith('projector.') for k in state['state'])
            assert any('lora_A' in k for k in state['state'])
            assert [g['name'] for g in state['param_groups']] == ['projector', 'lora']
            torch.save({'weights': weights, 'optimizer': state, 'torch_rng': rng}, Path(root) / 'last.pt')
        else:
            assert state == {}
        dist.barrier()
        for _ in range(2):
            update(model, optimizer, device, rank, 2)
        with full_model_state(model):
            expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expected_opt = collect_optimizer_state(model, optimizer)
        del model, optimizer
        checkpoint = torch.load(Path(root) / 'last.pt', weights_only=True, map_location='cpu')
        restored = ToyModel(device, sharded=True)
        restored_opt = make_optimizer(restored)
        with full_model_state(restored):
            restored.load_state_dict(checkpoint['weights'])
        restore_optimizer_state(restored, restored_opt, checkpoint['optimizer'], 'fsdp-full-v1')
        restore_rng_states(checkpoint['torch_rng'], device)
        for _ in range(2):
            update(restored, restored_opt, device, rank, 2)
        with full_model_state(restored):
            actual = {k: v.detach().cpu() for k, v in restored.state_dict().items()}
        assert_nested_equal(actual, expected)
        actual_opt = collect_optimizer_state(restored, restored_opt)
        if rank == 0:
            assert_nested_equal(actual_opt, expected_opt)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class MultitaskCheckpointTests(unittest.TestCase):
    def test_real_stage_loop_mid_lora_resume_preserves_dropout_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = run_real_stage_loop_until(root / 'continuous', 5)
            interrupted = run_real_stage_loop_until(root / 'resumed', 4)
            self.assertEqual(interrupted['stage_step'], 2)
            actual = run_real_stage_loop_until(root / 'resumed', 5)
            for key in ('trainable', 'optimizer', 'torch_rng', 'batcher', 'prng'):
                assert_nested_equal(actual[key], expected[key])

    def test_real_stage_loop_boundary_resume_preserves_new_lora_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = run_real_stage_loop_until(root / 'continuous', 5)
            interrupted = run_real_stage_loop_until(root / 'resumed', 2)
            self.assertEqual(interrupted['stage_step'], 0)
            self.assertEqual(interrupted['optimizer'], {})
            actual = run_real_stage_loop_until(root / 'resumed', 5)
            for key in ('trainable', 'optimizer', 'torch_rng', 'batcher', 'prng'):
                assert_nested_equal(actual[key], expected[key])

    def test_cpu_resume_matches_uninterrupted_training_with_dropout(self):
        torch.set_num_threads(1)
        device = torch.device('cpu')
        model = ToyModel(device)
        optimizer = make_optimizer(model)
        for _ in range(3):
            update(model, optimizer, device)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            checkpoint = {'model': copy.deepcopy(model.state_dict()),
                          'optimizer': collect_optimizer_state(model, optimizer),
                          'optimizer_format': optimizer_format(model),
                          'torch_rng': collect_rng_states(device)}
            torch.save(checkpoint, path)
            for _ in range(2):
                update(model, optimizer, device)
            expected = copy.deepcopy(model.state_dict())
            expected_opt = collect_optimizer_state(model, optimizer)
            # A collected state must not change when the live optimizer advances.
            assert_nested_equal(checkpoint, torch.load(path, weights_only=True))
            restored = ToyModel(device)
            restored_opt = make_optimizer(restored)
            state = torch.load(path, weights_only=True, map_location='cpu')
            restored.load_state_dict(state['model'])
            self.assertTrue(restore_optimizer_state(restored, restored_opt, state['optimizer'], state['optimizer_format']))
            self.assertTrue(restore_rng_states(state['torch_rng'], device))
            for _ in range(2):
                update(restored, restored_opt, device)
            assert_nested_equal(restored.state_dict(), expected)
            assert_nested_equal(collect_optimizer_state(restored, restored_opt), expected_opt)

    def test_legacy_non_fsdp_state_is_accepted(self):
        device = torch.device('cpu')
        model = ToyModel(device)
        optimizer = make_optimizer(model)
        update(model, optimizer, device)
        saved = copy.deepcopy(optimizer.state_dict())
        self.assertTrue(restore_optimizer_state(model, make_optimizer(model), saved))
        self.assertFalse(restore_rng_states(None, device))

    def test_missing_or_incompatible_optimizer_history_is_explicit(self):
        model = ToyModel(torch.device('cpu'))
        optimizer = make_optimizer(model)
        with self.assertRaisesRegex(ValueError, 'no optimizer state'):
            restore_optimizer_state(model, optimizer, {})
        self.assertFalse(restore_optimizer_state(model, optimizer, {}, allow_empty=True))
        with self.assertRaisesRegex(ValueError, 'format'):
            restore_optimizer_state(model, optimizer, optimizer.state_dict(), 'fsdp-full-v1')

    def test_rank_rng_contract(self):
        state = collect_rng_states(torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'world size'):
            restore_rng_states(state + state, torch.device('cpu'))

    @unittest.skipUnless(os.environ.get('EEGLLM_TEST_DISTRIBUTED') == '1',
                         'set EEGLLM_TEST_DISTRIBUTED=1 for the real two-rank FSDP test')
    def test_two_rank_fsdp_resume_matches_uninterrupted_training(self):
        device_type = os.environ.get('EEGLLM_TEST_DEVICE', 'cpu')
        if device_type not in ('cpu', 'cuda'):
            self.fail('EEGLLM_TEST_DEVICE must be cpu or cuda')
        if device_type == 'cuda' and torch.cuda.device_count() < 2:
            self.skipTest('two CUDA GPUs are required')
        try:
            with socket.socket() as server:
                server.bind(('127.0.0.1', 0))
        except OSError as error:
            self.skipTest(f'local distributed sockets unavailable: {error}')
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker, args=(tmp, device_type), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
