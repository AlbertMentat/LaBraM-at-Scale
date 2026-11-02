"""
Stage 2: LaBraM Pre-training — Masked EEG Modeling with frozen VQ-NSP tokenizer.

Uses UnifiedEEGDataset_v2 for data loading.
Adapted from LaBraM/run_labram_pretraining.py + avde data pipeline.
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math
import sys

from pathlib import Path
from einops import rearrange
from contextlib import nullcontext

from timm.models import create_model
from optim_factory import create_optimizer

from labram_utils import NativeScalerWithGradNormCount as NativeScaler
import labram_utils as utils

# trigger model registration
import models.labram_pretrain
import models.labram_vqnsp

from eeg_data.unified_dataset import (
    UnifiedEEGDataset_v2,
    GroupedBatchSampler,
    unified_collate_fn_v2,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_input_chans_safe(ch_names_list):
    """Compute input_chans from channel names with fallback to None."""
    try:
        ch_upper = [ch.upper() for ch in ch_names_list]
        return utils.get_input_chans(ch_upper)
    except (ValueError, KeyError):
        return None


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    x: [N, L, D] or [N, L] — only shape[:2] is used.
    Returns: bool mask of shape [N, L] (True = masked).
    """
    N, L = x.shape[:2]
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.to(torch.bool)


# ---------------------------------------------------------------------------
#  Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script', add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Tokenizer settings
    parser.add_argument('--tokenizer_weight', type=str, required=True)
    parser.add_argument('--tokenizer_model', type=str, default='vqnsp_encoder_base_decoder_3x200x12')

    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_1600_8k_vocab', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--input_size', default=1600, type=int, help='EEG input size for backbone')
    parser.add_argument('--drop_path', type=float, default=0.1)

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int)
    parser.add_argument('--codebook_dim', default=32, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR')
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

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--torch_compile', action='store_true', default=False,
                        help='Use torch.compile for speedup')

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Model creation
# ---------------------------------------------------------------------------

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size,
        use_grad_checkpoint=getattr(args, 'gradient_checkpointing', False),
    )
    return model


def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,
        as_tokenzer=True,
        n_code=args.codebook_size,
        code_dim=args.codebook_dim,
    ).eval()
    return model


# ---------------------------------------------------------------------------
#  Training engine (inlined from LaBraM/engine_for_pretraining.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, vqnsp, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Unpack batch from UnifiedEEGDataset_v2
        eeg = batch['eeg'].float().to(device, non_blocking=True) / 100  # [B, C, N, 200]
        ch_names = batch['meta']['ch_names'][0]
        input_chans = get_input_chans_safe(ch_names)

        # Pretraining model expects [B, N_ch, A, T=200]
        samples = eeg  # already [B, C, N, 200]

        bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                input_ids = vqnsp.get_codebook_indices(samples, input_chans)

            labels = input_ids[bool_masked_pos]
            labels_sym = input_ids[~bool_masked_pos]

        my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
        with my_context():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos)
                x_rec, x_rec_sym = outputs
                loss_rec = loss_fn(x_rec, labels)
                loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                loss = loss_rec + loss_rec_sym

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, zeroing loss at rank {utils.get_rank()}", force=True)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= args.gradient_accumulation_steps
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # NOTE: torch.cuda.synchronize() removed — it blocked CPU/GPU overlap
        # .item() calls below already sync as needed

        mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
        mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
        metric_logger.update(mlm_acc=mlm_acc)
        metric_logger.update(mlm_acc_sym=mlm_acc_sym)
        metric_logger.update(loss_rec=loss_rec.item() / 2)

        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")
            log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
            log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
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
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
#  Main
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
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # --- Dataset ---
    with open(args.dataset_configs, 'r') as f:
        dataset_configs = json.load(f)

    dataset_train = UnifiedEEGDataset_v2(
        dataset_configs, mode='train',
        balanced_sampling=args.balanced_sampling,
        force_rebuild_cache=args.force_rebuild_cache,
    )

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
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=unified_collate_fn_v2,
    )

    num_training_steps_per_epoch = len(data_loader_train)

    # --- Tokenizer ---
    vqnsp = get_visual_tokenizer(args).to(device)

    # --- Logging ---
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # --- Model setup ---
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print("Tokenizer = %s" % str(vqnsp))

    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if getattr(args, 'torch_compile', False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    # Barrier: ensure all ranks finished data loading / model init before training
    if args.distributed:
        torch.distributed.barrier()
        print(f"All ranks synchronized, starting training")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
        )

        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

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
