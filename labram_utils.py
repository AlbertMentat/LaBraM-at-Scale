import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf

from tensorboardX import SummaryWriter
import pickle
from scipy.signal import resample
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
            
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", log_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_file = log_file
        
    def write_log(self, message):
        """Write message to both file and stdout"""
        print(message)  # Keep console output for monitoring
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.write_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.write_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        _n_it = len(iterable) if hasattr(iterable, '__len__') and len(iterable) > 0 else 1  # 空loader护栏
        self.write_log('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / _n_it))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
    
    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)
            
    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    from datetime import timedelta
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=timedelta(minutes=60))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
        
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_nan_model(args, epoch, model, optimizer, loss_scaler=None, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-nan.pth'
    
    model_without_ddp = model.module
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        # 'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()

    if model_ema is not None:
        to_save['model_ema'] = get_state_dict(model_ema)
        
    if optimizer_disc is not None:
        to_save['optimizer_disc'] = optimizer_disc.state_dict()

    save_on_master(to_save, checkpoint_path)


def prune_numbered_checkpoints(output_dir, max_keep):
    if max_keep <= 0:
        return []
    numbered = []
    for path in Path(output_dir).glob("checkpoint-*.pth"):
        suffix = path.stem.removeprefix("checkpoint-")
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    removed = []
    for _epoch, path in sorted(numbered, reverse=True)[max_keep:]:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
        max_keep = int(getattr(args, 'max_numbered_checkpoints', 0) or 0)
        if is_main_process() and max_keep > 0:
            removed = prune_numbered_checkpoints(output_dir, max_keep)
            if removed:
                print(f"Pruned numbered checkpoints: {removed}")
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def save_model_step(args, epoch, step_in_epoch, model_without_ddp, optimizer, loss_scaler):
    """Save a mid-epoch checkpoint, ping-pong 交替写 checkpoint.pth / checkpoint_alt.pth (抗抢占写坏)。
    Stores 'step_in_epoch' so training can resume within the same epoch after preemption.
    """
    output_dir = Path(args.output_dir)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step_in_epoch': step_in_epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    # ping-pong 双缓冲: 交替写 checkpoint.pth / checkpoint_alt.pth。
    # 抢占中断只写坏当前这一份, 另一份仍是上次有效存档 (blobfuse rename 非原子, 不依赖 rename)。
    # auto_load_model 按 mtime 选最新且可加载的一份, giant 抢占后至多回退 save_ckpt_steps 步。
    n = step_in_epoch // max(1, getattr(args, 'save_ckpt_steps', 1) or 1)
    fname = 'checkpoint.pth' if (n % 2 == 0) else 'checkpoint_alt.pth'
    save_on_master(to_save, output_dir / fname)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    
    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        # 构建候选 checkpoint 列表(优先 checkpoint.pth, 再 checkpoint-N.pth 按 N 降序).
        # 抢占/中断可能留下 0 字节损坏 checkpoint, 故逐个尝试加载, 损坏则回退下一个.
        resume_candidates = []
        if args.auto_resume and len(args.resume) == 0:
            # step级双缓冲 checkpoint.pth / checkpoint_alt.pth: 按 mtime 新->旧优先,
            # 抢占写坏其一时回退另一份 (下方逐个尝试加载, 跳过 0 字节/损坏)。
            step_files = [os.path.join(output_dir, f)
                          for f in ('checkpoint.pth', 'checkpoint_alt.pth')]
            step_files = [f for f in step_files if os.path.exists(f)]
            step_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            resume_candidates.extend(step_files)
            epoch_ckpts = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            numbered = []
            for ckpt in epoch_ckpts:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    numbered.append((int(t), ckpt))
            for _, ckpt in sorted(numbered, key=lambda x: -x[0]):
                resume_candidates.append(ckpt)
            if resume_candidates:
                print("Auto resume candidates (in order): %s" % resume_candidates)
        elif args.resume:
            resume_candidates = [args.resume]

        checkpoint = None
        if resume_candidates:
            for cand in resume_candidates:
                try:
                    if str(cand).startswith('https'):
                        checkpoint = torch.hub.load_state_dict_from_url(
                            cand, map_location='cpu', check_hash=True)
                    else:
                        if os.path.getsize(cand) == 0:
                            # blobfuse大文件(如huge 4.23G)stat常误报0字节但read正常(base/large torch.load成功已证);
                            # copyfile走open+read绕过stat做本地化, 再load; 真损坏则本地也0字节→skip
                            print("getsize=0 for %s (疑blobfuse大文件stat缓存), copyfile本地化验证..." % cand)
                            import shutil as _sh, tempfile as _tf
                            _loc = os.path.join(_tf.gettempdir(), 'resume_' + os.path.basename(cand))
                            try:
                                _sh.copyfile(cand, _loc)
                            except Exception as _e:
                                print("  copyfile失败(%s), skip" % _e); continue
                            if os.path.getsize(_loc) == 0:
                                print("  本地化后仍0字节, 确为损坏, skip"); continue
                            print("  本地化成功 %d bytes (blobfuse stat误报) → %s" % (os.path.getsize(_loc), _loc))
                            cand = _loc
                        checkpoint = torch.load(cand, map_location='cpu', weights_only=False)
                    if 'model' not in checkpoint:
                        print("Skip checkpoint without 'model' key: %s" % cand)
                        checkpoint = None
                        continue
                    args.resume = cand
                    break
                except Exception as e:
                    print("Failed to load checkpoint %s (%s: %s), trying next candidate..." % (
                        cand, type(e).__name__, str(e)[:120]))
                    checkpoint = None
                    continue
            if checkpoint is None:
                print("No valid checkpoint found among candidates; training from scratch.")
                args.resume = ''

        if checkpoint is not None:
            model_without_ddp.load_state_dict(checkpoint['model']) # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                step_in_epoch = checkpoint.get('step_in_epoch', 0)
                if step_in_epoch and step_in_epoch > 0 and not getattr(args, 'no_mid_epoch_resume', False):
                    # mid-epoch resume: continue the SAME epoch from this step (fast-forward)
                    args.start_epoch = checkpoint['epoch']
                    args.start_step_in_epoch = step_in_epoch
                    print(f"Resume checkpoint at epoch {checkpoint['epoch']} step {step_in_epoch} (mid-epoch)")
                elif step_in_epoch and step_in_epoch > 0:
                    # no_mid_epoch_resume + mid-epoch ckpt: 从当前epoch开头重训(step归零)。
                    # 避免fast-forward重下shard空转, 同时不跳过未完成的epoch。
                    # (原bug: start_epoch=epoch+1把正在训的epoch误当成已完成,
                    #  retry在epoch中途时直接跳到下一epoch; 若是最后一epoch则range(N,N)空转直接pass)
                    args.start_epoch = checkpoint['epoch']
                    args.start_step_in_epoch = 0
                    print(f"Resume at epoch {checkpoint['epoch']} FROM START (retrain full epoch, no fast-forward, --no_mid_epoch_resume)")
                else:
                    # 完整epoch边界(save_model末尾写的checkpoint.pth无step_in_epoch): 从下一epoch开始
                    args.start_epoch = checkpoint['epoch'] + 1
                    args.start_step_in_epoch = 0
                    print(f"Resume checkpoint at epoch {checkpoint['epoch']} (completed), start next epoch")
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


# 大写 -> standard_1020 索引 的映射, 使通道查找对大小写不敏感
# (standard_1020 含 FTT9h/TTP7h 等小写h高密电极, 上游会 .upper() 成 FTT9H 导致 .index 失败)
_STD1020_UPPER2IDX = {name.upper(): i for i, name in enumerate(standard_1020)}

# TUH 双极导联 (standard_1020[128:136]) 越界 pos_embed(仅128通道槽).
# remap 到双极首电极位置(保留空间信息, 不丢数据). 如 'F3-C3' → 'F3' 的位置索引.
# 仅 CHBMIT/ZEN_10259996 等用双极蒙太奇的集受益; 单极通道不受影响.
_BIPOLAR_REMAP = {}
for _bip in standard_1020[128:]:
    _first = _bip.split('-')[0].upper()
    _fi = _STD1020_UPPER2IDX.get(_first)
    if _fi is not None and _fi < 128:
        _BIPOLAR_REMAP[_bip.upper()] = _fi

def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        cu = ch_name.upper().replace('EEG-','').replace('EEG ','').replace('-REF','').replace('-LE','').strip()
        idx = _STD1020_UPPER2IDX.get(cu)
        if idx is None:
            idx = 0  # 未知通道(数字索引/EOG)→cls位置 fallback
        if idx >= 128:
            # 越界 pos_embed: TUH 双极 remap 到首电极位置 (F3-C3 → F3)
            idx = _BIPOLAR_REMAP.get(cu, idx % 128)
        input_chans.append(idx + 1)
    return input_chans
