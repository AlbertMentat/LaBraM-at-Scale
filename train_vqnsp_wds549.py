#!/usr/bin/env python3
"""VQ-NSP tokenizer training on the WDS-549 full-R1 release.

Reuses the repository VQ-NSP model and training utilities; only the data path
differs from run_vqnsp_training.py:

  * input is consumed as-is (no /100) because the release stores z-scores
  * padding metadata is logged; reconstruction retains the original tokenizer loss
  * batches come from the release catalog via WDS549TrainingIterable
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import labram_utils as utils
import models.labram_vqnsp  # noqa: F401  (registers the timm model names)
from labram_utils import NativeScalerWithGradNormCount as NativeScaler
from optim_factory import create_optimizer
from wds549_data import (
    TRAIN_VIEWS,
    build_train_loader,
    prepare_batch,
    read_release_scale,
)
from run_vqnsp_training import get_input_chans_safe


def get_args():
    p = argparse.ArgumentParser("LaBraM extension v2 VQ-NSP tokenizer on WDS-549")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--save_ckpt_freq", default=1, type=int)
    p.add_argument("--save_ckpt_steps", default=0, type=int,
                   help="mid-epoch checkpoint interval (0 disables)")
    p.add_argument("--start_step_in_epoch", default=0, type=int)
    p.add_argument("--max_steps_per_epoch", default=0, type=int,
                   help="stop the epoch early; 0 runs the full epoch")

    p.add_argument("--model", default="vqnsp_encoder_base_decoder_3x200x12", type=str)
    p.add_argument("--codebook_n_emd", default=8192, type=int)
    p.add_argument("--codebook_emd_dim", default=32, type=int)
    p.add_argument("--ema_decay", default=0.99, type=float)
    p.add_argument("--quantize_kmeans_init", action="store_true", default=True)
    p.add_argument("--input_size", default=1600, type=int)

    p.add_argument("--opt", default="adamw", type=str)
    p.add_argument("--opt_betas", default=[0.9, 0.99], type=float, nargs="+")
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--lr", default=1.5e-5, type=float,
                   help="base rate per 128 global batch; lowered from 5e-5 to avoid codebook collapse")
    p.add_argument("--warmup_lr", default=1e-6, type=float)
    p.add_argument("--min_lr", default=1e-5, type=float)
    p.add_argument("--warmup_epochs", default=15, type=int)
    p.add_argument("--warmup_steps", default=-1, type=int)
    p.add_argument("--clip_grad", default=1.0, type=float)
    p.add_argument("--use_bf16", action="store_true", default=False)

    # WDS-549 release
    p.add_argument("--wds549_root", required=True,
                   help="release root holding corpus_builds/ and replacement_builds/")
    p.add_argument("--wds549_catalog_root", required=True,
                   help="catalogs/phase5-global-20260817")
    p.add_argument("--wds549_policy", default=str(HERE / "configs/wds_549_full_rebuild_policy_20260815.json"))
    p.add_argument("--wds549_views", nargs="+", default=list(TRAIN_VIEWS))
    p.add_argument("--wds549_data_fraction", default=1.0, type=float)
    p.add_argument("--wds549_cache_dir", default=None)
    p.add_argument("--wds549_allow_unreleased", action="store_true", default=False,
                   help="skip the RELEASE_READY.json requirement (smoke tests only)")

    p.add_argument("--output_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--log_dir", default=str(HERE / "logs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--resume", default="")
    p.add_argument("--auto_resume", action="store_true", default=True)
    p.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    p.add_argument("--start_epoch", default=0, type=int)
    p.add_argument("--num_workers", default=10, type=int)
    p.add_argument("--pin_mem", action="store_true", default=True)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url", default="env://")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch, loss_scaler,
                    log_writer=None, lr_schedule_values=None, start_steps=0,
                    args=None, scale=1.0):
    model.train()
    logger = utils.MetricLogger(delimiter="  ")
    logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    start_step = int(getattr(args, "start_step_in_epoch", 0) or 0)

    limit = int(getattr(args, "max_steps_per_epoch", 0) or 0)
    for step, batch in enumerate(logger.log_every(loader, 10, header)):
        if step < start_step:
            continue
        if limit and step >= limit:
            break
        it = start_steps + step
        if lr_schedule_values is not None and it < len(lr_schedule_values):
            for group in optimizer.param_groups:
                group["lr"] = lr_schedule_values[it] * group.get("lr_scale", 1.0)

        signal, patch_keep = prepare_batch(batch, device, scale)
        input_chans = get_input_chans_safe(batch["meta"]["ch_names"][0])
        dtype = torch.bfloat16 if getattr(args, "use_bf16", False) else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            loss, log_loss = model(signal, input_chans=input_chans)

        value = loss.item()
        if not math.isfinite(value):
            print(f"Loss is {value}, stopping training", force=True)
            sys.exit(1)

        optimizer.zero_grad()
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=False)
        torch.cuda.synchronize()

        logger.update(loss=value)
        logger.update(grad_norm=grad_norm)
        logger.update(lr=optimizer.param_groups[0]["lr"])
        logger.update(padded_patch_frac=1.0 - patch_keep.float().mean().item())
        for key, val in log_loss.items():
            logger.update(**{key: val})
        if log_writer is not None:
            log_writer.update(loss=value, head="train")
            log_writer.update(grad_norm=grad_norm, head="train")
            log_writer.update(lr=optimizer.param_groups[0]["lr"], head="opt")
            log_writer.update(**log_loss, head="train/loss")
            log_writer.set_step()

        if (args is not None and args.save_ckpt_steps
                and (step + 1) % args.save_ckpt_steps == 0):
            utils.save_model_step(args=args, epoch=epoch, step_in_epoch=step + 1,
                                  model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)

    logger.synchronize_between_processes()
    print("Averaged stats:", logger)
    stats = {k: meter.global_avg for k, meter in logger.meters.items()}

    unwrapped = model.module if hasattr(model, "module") else model
    if hasattr(unwrapped, "quantize"):
        try:
            cluster_size = unwrapped.quantize.cluster_size
        except AttributeError:
            cluster_size = unwrapped.quantize._codebook.cluster_size
        zero_cnt = (cluster_size == 0).sum().item()
        stats["Unused_code"] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt} / {cluster_size.numel()}",
              flush=True)
        if log_writer is not None:
            log_writer.update(unused_code=zero_cnt, head="codebook")
    return stats


def main():
    args = get_args()
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    scale = read_release_scale(args.wds549_policy)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    cudnn.benchmark = True

    global model_without_ddp
    model = create_model(
        args.model, pretrained=False, as_tokenzer=False,
        n_code=args.codebook_n_emd, code_dim=args.codebook_emd_dim,
        EEG_size=args.input_size, decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
    )
    model.to(device)
    model_without_ddp = model

    dataset, loader = build_train_loader(
        args, utils.get_world_size(), utils.get_rank())
    steps_per_epoch = len(dataset)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=True, broadcast_buffers=False)
        model_without_ddp = model.module

    # same linear-scaling rule as run_vqnsp_training.py: --lr is the per-128 base rate
    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = total_batch_size / 128 * args.lr
    print(f"total batch size = {total_batch_size}  scaled lr = {args.lr:.3e}")

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        start_warmup_value=args.warmup_lr)

    log_writer = (utils.TensorboardLogger(log_dir=args.log_dir)
                  if utils.get_rank() == 0 else None)
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                          optimizer=optimizer, loss_scaler=loss_scaler)

    if utils.get_rank() == 0:
        print(f"WDS-549 tokenizer training | views={args.wds549_views} "
              f"| input_scale={scale} | steps/epoch={steps_per_epoch} "
              f"| output={args.output_dir}", flush=True)

    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        stats = train_one_epoch(
            model, loader, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, lr_schedule_values=lr_schedule_values,
            start_steps=epoch * steps_per_epoch, args=args, scale=scale)
        args.start_step_in_epoch = 0

        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            utils.save_model(args=args, model=model,
                             model_without_ddp=model_without_ddp,
                             optimizer=optimizer, loss_scaler=loss_scaler,
                             epoch=epoch)
        if utils.get_rank() == 0:
            with (Path(args.output_dir) / "log.txt").open("a") as fh:
                fh.write(json.dumps({**{f"train_{k}": v for k, v in stats.items()},
                                     "epoch": epoch}) + "\n")

    print("Training time", str(datetime.timedelta(seconds=int(time.time() - start))))


if __name__ == "__main__":
    main()
