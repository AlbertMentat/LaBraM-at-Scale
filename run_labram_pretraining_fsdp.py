"""
Stage 2: LaBraM Pre-training with FSDP — Masked EEG Modeling.

Standalone FSDP version for training Giant-scale models (~2B params) on 40GB A100s.
Differences from run_labram_pretraining.py:
  - Uses FSDP (FULL_SHARD) instead of DDP → shards params/grads/optimizer across GPUs
  - bf16 MixedPrecision handled by FSDP (no GradScaler)
  - FSDP-aware checkpoint save/load (FullStateDictConfig)
  - FSDP clip_grad_norm_ for correct sharded gradient norms

Uses UnifiedEEGDataset_v2 for data loading.
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
import glob

from pathlib import Path
from einops import rearrange
from contextlib import nullcontext
from functools import partial

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from timm.models import create_model
from optim_factory import create_optimizer

import labram_utils as utils

# trigger model registration
import models.labram_pretrain
import models.labram_vqnsp
from models.labram import Block
from eeg_data.wds_dataset import WebDatasetEEGIterable
from eeg_data.wds_549_training_dataset import (
    TRAIN_VIEWS, WDS549TrainingIterable,
)
from run_labram_pretraining import (
    random_masking, token_accuracy, token_training_contract,
    weighted_token_cross_entropy,
)

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


def get_grad_norm_(parameters, norm_type: float = 2.0):
    """Compute gradient norm for non-sharded parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return total_norm


# ---------------------------------------------------------------------------
#  Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('LaBraM FSDP pre-training script', add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt_steps', default=0, type=int,
                        help='Save a mid-epoch checkpoint every N optimizer steps (0=disabled, preemption-safe).')
    parser.add_argument('--start_step_in_epoch', default=0, type=int,
                        help='Internal: resume within start_epoch from this step (set by auto_load).')

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
    parser.add_argument('--dataset_configs', type=str, default='',
                        help='Path to dataset configs JSON file')
    parser.add_argument('--force_rebuild_cache', action='store_true', default=False)
    parser.add_argument('--balanced_sampling', action='store_true', default=False)
    parser.add_argument('--use_wds', action='store_true', default=False)
    parser.add_argument('--wds_root', type=str, default='')
    parser.add_argument('--wds_format', choices=('legacy-v2', 'wds-549'),
                        default='legacy-v2')
    parser.add_argument('--wds_catalog_root', type=str, default='')
    parser.add_argument('--wds_train_views', nargs='+', default=list(TRAIN_VIEWS),
                        choices=TRAIN_VIEWS)
    parser.add_argument('--wds_strict_v2', action='store_true', default=False)
    parser.add_argument('--wds_require_catalog', action='store_true', default=False)
    parser.add_argument('--wds_cache_dir', type=str, default='')
    parser.add_argument('--wds_cache_max_gb', type=float,
                        default=float(os.environ.get('WDS_CACHE_MAX_GB', '0')))
    parser.add_argument('--wds_cache_min_free_gb', type=float,
                        default=float(os.environ.get('WDS_CACHE_MIN_FREE_GB', '20')))
    parser.add_argument('--wds_549_allow_unreleased', action='store_true')
    parser.add_argument('--wds_549_verify_tar_sha256', action='store_true')
    parser.add_argument('--pretrain_data_fraction', type=float, default=1.0)
    parser.add_argument('--input_scale', type=float, default=0.01)

    # Output / logging
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path for tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--init_from', default='',
                        help='Path to checkpoint to load model weights ONLY from '
                             '(optimizer/epoch/scaler are ignored). Used only when '
                             'no resume checkpoint is found in output_dir.')

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
    _assert_tokenizer_loaded(model, args.tokenizer_weight)
    return model


def _assert_tokenizer_loaded(model, weight_path):
    """A silently mis-loaded tokenizer still runs and emits garbage tokens, so
    check the codebook carries trained statistics rather than fresh init."""
    try:
        cluster_size = model.quantize.cluster_size
    except AttributeError:
        cluster_size = model.quantize._codebook.cluster_size
    active = int((cluster_size != 0).sum().item())
    total = cluster_size.numel()
    print(f"Tokenizer weight: {weight_path}", flush=True)
    print(f"Tokenizer codebook: {active}/{total} entries active", flush=True)
    if active == 0:
        raise ValueError(
            f"tokenizer codebook is entirely unused -- weights were probably not "
            f"loaded from {weight_path}")


# ---------------------------------------------------------------------------
#  FSDP checkpoint save / load
# ---------------------------------------------------------------------------

def save_model_fsdp(args, epoch, model, optimizer, save_ckpt_freq=1):
    """Save FSDP model checkpoint — gathers full state dict to rank 0."""
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_paths = [output_dir / 'checkpoint.pth']
    if epoch == 'best':
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    elif (epoch + 1) % save_ckpt_freq == 0:
        checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

    # Gather full state dicts to rank 0 (offload to CPU to save GPU memory)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_save_policy):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_state,
            'optimizer': optim_state,
            'epoch': epoch,
            'args': args,
        }
        utils.save_on_master(to_save, checkpoint_path)


def _fsdp_step_checkpoint_path(args, output_dir, step_in_epoch):
    interval = max(1, getattr(args, 'save_ckpt_steps', 1) or 1)
    save_index = step_in_epoch // interval
    filename = 'checkpoint.pth' if save_index % 2 == 0 else 'checkpoint_alt.pth'
    return Path(output_dir) / filename


def save_model_step_fsdp(args, epoch, step_in_epoch, model, optimizer):
    """Save a mid-epoch FSDP checkpoint using two alternating files.

    BlobFuse overwrites expose a zero-byte window. Alternating files ensures a
    preemption can damage only the new write while the prior checkpoint remains
    available for resume.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = _fsdp_step_checkpoint_path(args, output_dir, step_in_epoch)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_save_policy):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    to_save = {
        'model': model_state,
        'optimizer': optim_state,
        'epoch': epoch,
        'step_in_epoch': step_in_epoch,
        'args': args,
    }
    utils.save_on_master(to_save, checkpoint_path)


def _fsdp_resume_candidates(output_dir):
    output_dir = Path(output_dir)
    step_files = [output_dir / name for name in ('checkpoint.pth', 'checkpoint_alt.pth')]
    step_files = [path for path in step_files if path.exists()]
    step_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    numbered = []
    for path in output_dir.glob('checkpoint-*.pth'):
        suffix = path.stem.removeprefix('checkpoint-')
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    return step_files + [path for _, path in sorted(numbered, reverse=True)]


def _load_valid_fsdp_checkpoint(candidates):
    for candidate in candidates:
        try:
            checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
            if 'model' not in checkpoint:
                print("Skip checkpoint without 'model' key: %s" % candidate)
                continue
            return str(candidate), checkpoint
        except Exception as error:
            print(
                "Failed to load checkpoint %s (%s: %s), trying next candidate..."
                % (candidate, type(error).__name__, str(error)[:120])
            )
    return '', None


def auto_load_model_fsdp(args, model, optimizer):
    """Resume from the newest loadable FSDP checkpoint."""
    output_dir = Path(args.output_dir)
    resume_candidates = []

    if args.auto_resume and len(args.resume) == 0:
        resume_candidates = _fsdp_resume_candidates(output_dir)
        if resume_candidates:
            print("Auto resume candidates (in order): %s" % resume_candidates)
    elif args.resume:
        resume_candidates = [Path(args.resume)]

    checkpoint = None
    if resume_candidates:
        args.resume, checkpoint = _load_valid_fsdp_checkpoint(resume_candidates)
        if checkpoint is None:
            print("No valid FSDP checkpoint found among candidates; training from scratch.")

    if checkpoint is not None:
        # Load model state dict via FSDP context
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            # Convert saved full optimizer state dict to FSDP sharded format
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, checkpoint['optimizer'])
            optimizer.load_state_dict(optim_state)
            step_in_epoch = checkpoint.get('step_in_epoch', 0)
            if step_in_epoch and step_in_epoch > 0:
                # mid-epoch resume: continue the SAME epoch from this step
                args.start_epoch = checkpoint['epoch']
                args.start_step_in_epoch = step_in_epoch
                print(f"Resume checkpoint at epoch {checkpoint['epoch']} step {step_in_epoch} (mid-epoch)")
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                args.start_step_in_epoch = 0
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
            print("With optim & sched!")
    elif getattr(args, "init_from", ""):
        # All-ranks load (validated working in attempts 1-3). Use default FULL_STATE_DICT
        # so FSDP automatically reshapes & matches wrapper prefixes during load_state_dict.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print("[init_from] loading model weights ONLY from %s" % args.init_from, flush=True)
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model"])
        del checkpoint
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print("[init_from] loaded; optimizer/epoch IGNORED (fresh start_epoch=0)", flush=True)


# ---------------------------------------------------------------------------
#  Training engine
# ---------------------------------------------------------------------------

def train_one_epoch(model, vqnsp, data_loader, optimizer, device, epoch,
                    max_norm=0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None,
                    start_step=0, save_ckpt_steps=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
        if step < start_step:
            # mid-epoch resume: fast-forward over already-completed steps
            continue
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        eeg = (
            batch['eeg'].float().to(device, non_blocking=True)
            * float(args.input_scale))
        ch_names = batch['meta']['ch_names'][0]
        input_chans = get_input_chans_safe(ch_names)
        if args.use_wds and args.wds_format == 'wds-549' and input_chans is None:
            raise ValueError(f"WDS 549 contains unmapped channels: {ch_names}")
        samples = torch.clamp(
            torch.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0),
            -20.0, 20.0)
        token_padding_mask, valid_token_mask, token_weights = (
            token_training_contract(batch, samples))
        bool_masked_pos = random_masking(
            samples.flatten(1, 2), mask_ratio=0.5,
            valid_token_mask=valid_token_mask)
        symmetric_mask = ~bool_masked_pos & valid_token_mask

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                input_ids = vqnsp.get_codebook_indices(samples, input_chans)
            if input_ids.shape != bool_masked_pos.shape:
                raise ValueError("tokenizer output shape does not match training mask")
            labels = input_ids[bool_masked_pos]
            labels_sym = input_ids[symmetric_mask]
            weights = token_weights[bool_masked_pos]
            weights_sym = token_weights[symmetric_mask]

        # FSDP no_sync for gradient accumulation
        my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
        with my_context():
            # FSDP handles mixed precision via its MixedPrecision policy, no manual autocast needed
            outputs = model(
                samples, input_chans, bool_masked_pos=bool_masked_pos,
                token_padding_mask=token_padding_mask)
            x_rec, x_rec_sym = outputs
            loss_rec = weighted_token_cross_entropy(x_rec, labels, weights)
            loss_rec_sym = weighted_token_cross_entropy(
                x_rec_sym, labels_sym, weights_sym)
            loss = loss_rec + loss_rec_sym

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, zeroing loss at rank {utils.get_rank()}", force=True)
            loss = 0.0 * sum(
                parameter.sum() for parameter in model.parameters()
                if parameter.requires_grad)

        loss /= args.gradient_accumulation_steps
        # FSDP + bf16: no GradScaler needed, just plain backward + step
        loss.backward()

        grad_norm = None
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if max_norm is not None and max_norm > 0:
                grad_norm = model.clip_grad_norm_(max_norm)
            else:
                grad_norm = get_grad_norm_(model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- step-level checkpoint (mid-epoch, overwrites checkpoint.pth) ----
        if (save_ckpt_steps and args.output_dir
                and (step + 1) % args.gradient_accumulation_steps == 0
                and (step + 1) % save_ckpt_steps == 0):
            save_model_step_fsdp(
                args=args, epoch=epoch, step_in_epoch=step + 1,
                model=model, optimizer=optimizer)

        # NOTE: torch.cuda.synchronize() removed — it blocked CPU/GPU overlap
        # .item() calls below already sync as needed

        mlm_acc = token_accuracy(x_rec, labels)
        mlm_acc_sym = token_accuracy(x_rec_sym, labels_sym)
        metric_logger.update(mlm_acc=mlm_acc)
        metric_logger.update(mlm_acc_sym=mlm_acc_sym)
        metric_logger.update(loss_rec=loss_rec.item() / 2)

        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")
            log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
            log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")

        metric_logger.update(loss=loss_value)
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
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if grad_norm is not None:
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
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.use_wds:
        if args.wds_cache_max_gb < 0 or args.wds_cache_min_free_gb < 0:
            raise ValueError("WDS cache size arguments must be nonnegative")
        cache_max_bytes = (
            int(args.wds_cache_max_gb * 1024 ** 3)
            if args.wds_cache_max_gb > 0 else None)
        cache_min_free_bytes = int(args.wds_cache_min_free_gb * 1024 ** 3)
        if not args.wds_root:
            raise ValueError("--use_wds requires --wds_root")
        if args.wds_format == 'wds-549':
            if not args.wds_catalog_root:
                raise ValueError("--wds_format wds-549 requires --wds_catalog_root")
            if not math.isclose(args.input_scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("WDS 549 requires --input_scale 1.0")
            if args.rel_pos_bias:
                raise ValueError("WDS 549 variable windows require absolute position embeddings")
            dataset_train = WDS549TrainingIterable(
                args.wds_root, args.wds_catalog_root,
                batch_size=args.batch_size, views=args.wds_train_views,
                num_replicas=num_tasks, rank=global_rank, seed=args.seed,
                shuffle=True, drop_last=True,
                data_fraction=args.pretrain_data_fraction,
                require_release_ready=not args.wds_549_allow_unreleased,
                verify_tar_sha256=args.wds_549_verify_tar_sha256,
                cache_dir=args.wds_cache_dir,
                cache_max_bytes=cache_max_bytes,
                cache_min_free_bytes=cache_min_free_bytes,
            )
        else:
            dataset_train = WebDatasetEEGIterable(
                args.wds_root, batch_size=args.batch_size, mode='train',
                num_replicas=num_tasks, rank=global_rank, seed=args.seed,
                shuffle=True, drop_last=True,
                cache_dir=args.wds_cache_dir,
                cache_max_bytes=cache_max_bytes,
                cache_min_free_bytes=cache_min_free_bytes,
                data_fraction=args.pretrain_data_fraction,
                strict_v2=args.wds_strict_v2,
                require_catalog=args.wds_require_catalog,
            )
        sampler_train = dataset_train
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=None, num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
    else:
        if not args.dataset_configs:
            raise ValueError("non-WDS training requires --dataset_configs")
        with open(args.dataset_configs, 'r') as f:
            dataset_configs = json.load(f)
        dataset_train = UnifiedEEGDataset_v2(
            dataset_configs, mode='train',
            balanced_sampling=args.balanced_sampling,
            force_rebuild_cache=args.force_rebuild_cache,
        )
        sampler_train = GroupedBatchSampler(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_replicas=num_tasks, rank=global_rank, seed=args.seed,
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=sampler_train,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
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

    # --- Model setup with FSDP ---
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)
    print("Tokenizer = %s" % str(vqnsp))

    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # --- FSDP wrapping ---
    fsdp_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=args.gpu,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    print(f"FSDP enabled: FULL_SHARD, bf16 mixed precision, {num_tasks} GPUs")

    # model_without_ddp = model for FSDP (use_orig_params=True allows named_parameters)
    optimizer = create_optimizer(args, model)
    # Enable fused kernel for AdamW (~20% faster optimizer step on CUDA)
    if hasattr(optimizer, 'defaults') and isinstance(optimizer, torch.optim.AdamW):
        for group in optimizer.param_groups:
            group.setdefault('fused', True)

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

    auto_load_model_fsdp(args=args, model=model, optimizer=optimizer)

    # Barrier: ensure all ranks finished data loading / model init before training
    if args.distributed:
        torch.distributed.barrier()
        print(f"All ranks synchronized, starting training")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        skip_steps = args.start_step_in_epoch if epoch == args.start_epoch else 0
        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train, optimizer, device, epoch,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
            start_step=skip_steps, save_ckpt_steps=args.save_ckpt_steps,
        )
        # mid-epoch resume only applies to the first resumed epoch
        args.start_step_in_epoch = 0

        if args.output_dir:
            save_model_fsdp(
                args=args, model=model,
                optimizer=optimizer, epoch=epoch,
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
