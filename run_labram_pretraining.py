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
from pretraining_continuation import restore_continuation

from labram_utils import NativeScalerWithGradNormCount as NativeScaler
import labram_utils as utils

# trigger model registration
import models.labram_pretrain
import models.labram_vqnsp
from eeg_data.wds_dataset import WebDatasetEEGIterable
from eeg_data.wds_549_training_dataset import (
    TRAIN_VIEWS, WDS549TrainingIterable,
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


def random_masking(x, mask_ratio, valid_token_mask=None):
    """Return a per-sample random mask, excluding invalid tokens when supplied."""
    batch_size, token_count = x.shape[:2]
    if valid_token_mask is None:
        len_keep = int(token_count * (1 - mask_ratio))
        noise = torch.rand(batch_size, token_count, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([batch_size, token_count], device=x.device)
        mask[:, :len_keep] = 0
        return torch.gather(mask, dim=1, index=ids_restore).to(torch.bool)

    if (valid_token_mask.dtype != torch.bool
            or valid_token_mask.shape != (batch_size, token_count)):
        raise ValueError("valid_token_mask must be bool with shape [batch, tokens]")
    valid_counts = valid_token_mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("every sample must contain at least one valid token")
    masked_counts = valid_counts - torch.floor(
        valid_counts.float() * (1 - mask_ratio)).to(torch.long)
    noise = torch.rand(batch_size, token_count, device=x.device)
    noise = noise.masked_fill(~valid_token_mask, 2.0)
    ranks = torch.argsort(torch.argsort(noise, dim=1), dim=1)
    return (ranks < masked_counts.unsqueeze(1)) & valid_token_mask


def token_training_contract(batch, eeg):
    """Build token padding and loss weights from sample-level WDS metadata."""
    batch_size, n_channels, n_patches, patch_size = eeg.shape
    if patch_size != 200:
        raise ValueError(f"expected 200 samples per patch, got {patch_size}")
    token_count = n_channels * n_patches
    sample_weight = batch.get("sample_weight")
    if sample_weight is None:
        sample_weight = torch.ones(batch_size, device=eeg.device)
    else:
        sample_weight = sample_weight.to(
            device=eeg.device, dtype=torch.float32, non_blocking=True)
        if sample_weight.shape != (batch_size,):
            raise ValueError("sample_weight must have shape [batch]")
        if not torch.isfinite(sample_weight).all() or torch.any(sample_weight <= 0):
            raise ValueError("sample_weight must be finite and positive")

    padding_mask = batch.get("padding_mask")
    if padding_mask is None:
        token_valid_fractions = torch.ones(
            (batch_size, token_count), device=eeg.device)
        token_padding_mask = None
    else:
        padding_mask = padding_mask.to(
            device=eeg.device, dtype=torch.bool, non_blocking=True)
        if padding_mask.shape != (batch_size, n_patches * patch_size):
            raise ValueError("padding_mask must have shape [batch, patches * 200]")
        patch_valid_fractions = (~padding_mask).reshape(
            batch_size, n_patches, patch_size).float().mean(dim=-1)
        token_valid_fractions = patch_valid_fractions[:, None, :].expand(
            batch_size, n_channels, n_patches).reshape(batch_size, token_count)
        token_padding_mask = token_valid_fractions <= 0
    token_weights = token_valid_fractions * sample_weight.unsqueeze(1)
    return token_padding_mask, token_valid_fractions > 0, token_weights


def weighted_token_cross_entropy(logits, labels, weights):
    if logits.shape[0] != labels.shape[0] or labels.shape != weights.shape:
        raise ValueError("token logits, labels, and weights are misaligned")
    if labels.numel() == 0:
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, labels, reduction="none")
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("token loss has no positive finite weight")
    return (losses * weights).sum() / denominator


def token_accuracy(logits, labels):
    if labels.numel() == 0:
        return 0.0
    return (logits.argmax(dim=-1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
#  Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script', add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt_steps', default=0, type=int,
                        help='Save checkpoint.pth every N optimizer steps within an epoch (0=disabled). '
                             'Enables mid-epoch resume under frequent preemption.')
    parser.add_argument('--start_step_in_epoch', default=0, type=int,
                        help='Internal: step to resume from within the first (resumed) epoch. Set by auto_load_model.')

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
    parser.add_argument('--use_wds', action='store_true', default=False, help='Use WebDatasetEEGIterable tar shards for fast IO')
    parser.add_argument('--wds_root', type=str, default='', help='Root dir of WebDataset .tar shards')
    parser.add_argument('--wds_format', choices=('legacy-v2', 'wds-549'),
                        default='legacy-v2')
    parser.add_argument('--wds_catalog_root', type=str, default='',
                        help='WDS 549 global catalog root containing CATALOGS_COMPLETE.json')
    parser.add_argument('--wds_train_views', nargs='+', default=list(TRAIN_VIEWS),
                        choices=TRAIN_VIEWS)
    parser.add_argument('--wds_549_allow_unreleased', action='store_true',
                        help='Canary only: allow WDS 549 before RELEASE_READY.json exists')
    parser.add_argument('--wds_549_verify_tar_sha256', action='store_true',
                        help='Verify each WDS 549 tar SHA before reading it')
    parser.add_argument('--input_scale', type=float, default=0.01,
                        help='Scale applied to EEG before tokenizer/model; WDS 549 requires 1.0')
    parser.add_argument('--wds_strict_v2', action='store_true', default=False,
                        help='Fail fast unless WDS v2 schema/signatures/shapes are valid')
    parser.add_argument('--wds_require_catalog', action='store_true', default=False,
                        help='Read only tar files listed by a checksummed WDS v2 catalog')
    parser.add_argument('--wds_cache_dir', type=str, default='',
                        help='本地磁盘分片缓存目录(命中先读本地, 未命中从blob拷到本地). '
                             '空=关闭; 也可用 env WDS_CACHE_DIR. 建议指向节点本地SSD.')
    parser.add_argument('--wds_cache_max_gb', type=float,
                        default=float(os.environ.get('WDS_CACHE_MAX_GB', '0')),
                        help='Maximum node-local cache GiB; 0 uses 80%% of usable free space')
    parser.add_argument('--wds_cache_min_free_gb', type=float,
                        default=float(os.environ.get('WDS_CACHE_MIN_FREE_GB', '20')),
                        help='Disk GiB that the cache must leave free')
    parser.add_argument('--pretrain_data_fraction', type=float, default=1.0,
                        help='样本级间隔采样比例(0<f<=1): 每个数据集每隔round(1/f)取1个样本, 保持数据集间比例, 总数据量x f. 用于data scaling. 1.0=全量')

    # Output / logging
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path for tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--continuation_plan', default='',
                        help='Frozen JSON plan for a separate continuation phase')
    parser.add_argument('--continuation_arm', default='',
                        help='Arm name in the continuation plan')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--no_mid_epoch_resume', action='store_true',
                        help='ignore step_in_epoch, resume from next epoch boundary (avoid slow fast-forward that idles GPU)')

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
#  Training engine (inlined from LaBraM/engine_for_pretraining.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, vqnsp, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None,
                    start_step=0, save_ckpt_steps=0, model_without_ddp=None):
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

        my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
        with my_context():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    samples, input_chans, bool_masked_pos=bool_masked_pos,
                    token_padding_mask=token_padding_mask)
                x_rec, x_rec_sym = outputs
                loss_rec = weighted_token_cross_entropy(
                    x_rec, labels, weights)
                loss_rec_sym = weighted_token_cross_entropy(
                    x_rec_sym, labels_sym, weights_sym)
                loss = loss_rec + loss_rec_sym

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, zero-grad proxy at rank {utils.get_rank()}", force=True)
            # NaN/inf loss: nan_to_num(loss)会因 0*NaN 反传仍得NaN梯度污染权重;
            # 改用基于(此刻仍有限的)参数的零梯度代理, 反传得纯0梯度, 不污染. DDP安全(各rank都backward).
            loss = 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= args.gradient_accumulation_steps
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # ---- step-level checkpoint (mid-epoch, overwrites checkpoint.pth) ----
        if (save_ckpt_steps and args.output_dir and model_without_ddp is not None
                and (step + 1) % args.gradient_accumulation_steps == 0
                and (step + 1) % save_ckpt_steps == 0):
            utils.save_model_step(
                args=args, epoch=epoch, step_in_epoch=step + 1,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler)

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
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.use_wds:
        if args.wds_cache_max_gb < 0 or args.wds_cache_min_free_gb < 0:
            raise ValueError("WDS cache size arguments must be nonnegative")
        cache_max_bytes = (
            int(args.wds_cache_max_gb * 1024 ** 3)
            if args.wds_cache_max_gb > 0 else None)
        cache_min_free_bytes = int(args.wds_cache_min_free_gb * 1024 ** 3)
        assert args.wds_root, "--use_wds requires --wds_root"
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
        sampler_train = dataset_train  # IterableDataset 自带 set_epoch
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        num_training_steps_per_epoch = len(dataset_train)
    else:
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
    loss_scaler = NativeScaler(enabled=False)  # bf16 autocast: GradScaler disabled

    print("Use step level LR & WD scheduler!")
    schedule_start_epoch = 0
    if args.continuation_plan:
        lr_schedule_values, wd_schedule_values, schedule_start_epoch = restore_continuation(
            args, model_without_ddp, optimizer, loss_scaler,
            num_training_steps_per_epoch, utils.get_world_size())
    else:
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

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
            model, vqnsp, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=(epoch - schedule_start_epoch) * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
            start_step=skip_steps, save_ckpt_steps=args.save_ckpt_steps,
            model_without_ddp=model_without_ddp,
        )
        # mid-epoch resume only applies to the first resumed epoch
        args.start_step_in_epoch = 0

        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        if args.continuation_plan:
            log_stats.update({
                'continuation_arm': args.continuation_arm,
                'phase_epoch': epoch - schedule_start_epoch,
                'phase_steps': (epoch + 1 - schedule_start_epoch) * num_training_steps_per_epoch,
            })

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
