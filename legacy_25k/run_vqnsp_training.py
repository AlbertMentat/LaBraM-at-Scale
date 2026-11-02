"""
Stage 1: VQ-NSP Training — trains the discrete EEG tokenizer.

Uses UnifiedEEGDataset_v2 for data loading.
Adapted from LaBraM/run_vqnsp_training.py + avde data pipeline.
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
import logging

from pathlib import Path
from einops import rearrange

from timm.models import create_model
from optim_factory import create_optimizer

from labram_utils import NativeScalerWithGradNormCount as NativeScaler
import labram_utils as utils

# trigger model registration
import models.labram_vqnsp

from eeg_data.unified_dataset import (
    UnifiedEEGDataset_v2,
    GroupedBatchSampler,
    unified_collate_fn_v2,
)


# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
#  Helpers
# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------

def get_input_chans_safe(ch_names_list):
    """Compute input_chans from channel names with fallback to None."""
    try:
        ch_upper = [ch.upper() for ch in ch_names_list]
        return utils.get_input_chans(ch_upper)
    except (ValueError, KeyError):
        return None


# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
#  Arguments
# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('LaBraM VQ-NSP training script', add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='vqnsp_encoder_base_decoder_3x200x12', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--codebook_n_emd', default=8192, type=int, help='number of codebook entries')
    parser.add_argument('--codebook_emd_dim', default=32, type=int, help='codebook embedding dim')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='ema decay for quantizer')
    parser.add_argument('--quantize_kmeans_init', action='store_true', help='enable kmeans_init for quantizer')
    parser.add_argument('--input_size', default=1600, type=int, help='EEG input size for backbone')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    # Dataset parameters
    parser.add_argument('--dataset_configs', type=str, required=True,
                        help='Path to dataset configs JSON file')
    parser.add_argument('--force_rebuild_cache', action='store_true', default=False)
    parser.add_argument('--balanced_sampling', action='store_true', default=False)

    # Output / logging
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path for tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--disable_eval', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser.parse_args()


# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
#  Model creation
# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------

def get_model(args):
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        EEG_size=args.input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
    )
    return model


# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
#  Training engine (inlined from LaBraM/engine_for_vqnsp.py)
# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------

def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    clip_grad=0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Reset codebook stats at the start of each epoch
    model_unwrap = model.module if hasattr(model, 'module') else model
    if hasattr(model_unwrap, 'quantize'):
        try:
            model_unwrap.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except Exception:
            pass

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

        # Unpack batch from UnifiedEEGDataset_v2
        eeg = batch['eeg'].float().to(device, non_blocking=True) / 100  # [B, C, N, 200]
        ch_names = batch['meta']['ch_names'][0]  # same within batch
        input_chans = get_input_chans_safe(ch_names)

        # VQNSP expects [B, N_ch, Total_T] — flatten temporal dimension
        eeg_flat = rearrange(eeg, 'B C N T -> B C (N T)')

        with torch.amp.autocast('cuda', enabled=True):
            loss, log_loss = model(eeg_flat, input_chans=input_chans)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(**new_log_loss, head="train/loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Stat codebook usage
    model_unwrap = model.module if hasattr(model, 'module') else model
    if hasattr(model_unwrap, 'quantize'):
        try:
            codebook_cluster_size = model_unwrap.quantize.cluster_size
        except Exception:
            codebook_cluster_size = model_unwrap.quantize._codebook.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['Unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, log_writer=None, epoch=None, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    model.eval()

    model_unwrap = model.module if hasattr(model, 'module') else model
    if hasattr(model_unwrap, 'quantize'):
        try:
            model_unwrap.quantize.reset_cluster_size(device)
        except Exception:
            pass

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg = batch['eeg'].float().to(device, non_blocking=True) / 100
        ch_names = batch['meta']['ch_names'][0]
        input_chans = get_input_chans_safe(ch_names)
        eeg_flat = rearrange(eeg, 'B C N T -> B C (N T)')

        loss, log_loss = model(eeg_flat, input_chans=input_chans)
        metric_logger.update(loss=loss.item())
        new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    model_unwrap = model.module if hasattr(model, 'module') else model
    if hasattr(model_unwrap, 'quantize'):
        try:
            codebook_cluster_size = model_unwrap.quantize.cluster_size
        except Exception:
            codebook_cluster_size = model_unwrap.quantize._codebook.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
#  Main
# Configure logging so that INFO messages from dataset loading are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)

    # --- Dataset ---
    with open(args.dataset_configs, 'r') as f:
        dataset_configs = json.load(f)

    dataset_train = UnifiedEEGDataset_v2(
        dataset_configs, mode='train',
        balanced_sampling=args.balanced_sampling,
        force_rebuild_cache=args.force_rebuild_cache,
    )

    if not args.disable_eval:
        dataset_val = UnifiedEEGDataset_v2(
            dataset_configs, mode='test',
            balanced_sampling=False,
            force_rebuild_cache=args.force_rebuild_cache,
        )
    else:
        dataset_val = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = GroupedBatchSampler(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_replicas=num_tasks, rank=global_rank, seed=args.seed,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=unified_collate_fn_v2,
    )

    num_training_steps_per_epoch = len(data_loader_train)

    if dataset_val is not None:
        sampler_val = GroupedBatchSampler(
            dataset_val, batch_size=int(1.5 * args.batch_size), shuffle=False, drop_last=False,
            num_replicas=num_tasks, rank=global_rank, seed=args.seed,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_sampler=sampler_val,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=unified_collate_fn_v2,
        )
    else:
        data_loader_val = None

    # --- Logging ---
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # --- Model setup ---
    model.to(device)
    model_without_ddp = model

    if not getattr(args, 'eval', False):
        print("Model = %s" % str(model_without_ddp))
    for part in ['encoder', 'decoder']:
        model_part = getattr(model, part)
        n_learnable = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fix = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f'number of learnable params in model.{part}: {n_learnable / 1e6} M')
        print(f'number of fixed params in model.{part}: {n_fix / 1e6} M')

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = total_batch_size / 128 * args.lr
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weight Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            args=args,
        )

        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, log_writer, epoch, args=args)
            print(f"Validation loss: {test_stats['loss']:.4f}")

            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
