#!/usr/bin/env python3
"""
LaBraM at Scale: classification fine-tuning

Generic finetuning script that uses UnifiedEEGDataset_v2 to support all 8
downstream datasets via a single dataset config JSON:
  - tuab, tuev, hmc (clinical/sleep)
  - eegimage sub-datasets: SEED, SEED-IV, SEED-V, SEED-VII, Game

Usage:
  python run_unified_finetuning.py \
      --dataset_config experiments/configs/single_tuab.json \
      --finetune /path/to/pretrain/checkpoint.pth \
      --output_dir /path/to/output

For EEGImage sub-datasets that lack a validation split, the test set is
used as both val and test (stated explicitly in logs).
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import OrderedDict, Counter
from pathlib import Path

from downstream_state import training_class_counts, best_completed_epoch

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from einops import rearrange
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

import labram_utils as utils
import models.labram as modeling_finetune  # noqa: F401  -- registers models
from labram_utils import NativeScalerWithGradNormCount as NativeScaler
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from eeg_data.unified_dataset import UnifiedEEGDataset_v2

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score,
        average_precision_score, f1_score, cohen_kappa_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def _json_safe(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def freeze_first_transformer_blocks(model, count):
    if count < 0:
        raise ValueError("freeze_layers must be non-negative")
    pattern = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    indexed = []
    block_indices = set()
    for name, parameter in model.named_parameters():
        match = pattern.search(name)
        if match is not None:
            index = int(match.group(1))
            indexed.append((index, parameter))
            block_indices.add(index)
    if count == 0:
        return {"frozen_blocks": 0, "available_blocks": len(block_indices), "frozen_parameters": 0}
    if not block_indices:
        raise ValueError("freeze_layers requested but model has no transformer blocks")
    expected = set(range(max(block_indices) + 1))
    if block_indices != expected:
        raise ValueError(f"non-contiguous transformer block indices: {sorted(block_indices)}")
    if count > len(block_indices):
        raise ValueError(
            f"freeze_layers={count} exceeds available transformer blocks={len(block_indices)}"
        )
    frozen_parameters = 0
    for index, parameter in indexed:
        if index < count:
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()
    return {
        "frozen_blocks": count,
        "available_blocks": len(block_indices),
        "frozen_parameters": frozen_parameters,
    }


def write_results_json(output_dir, payload):
    if not output_dir:
        return
    out = Path(output_dir) / "results.json"
    tmp = Path(output_dir) / "results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp, out)


def write_durable_epoch_record(output_dir: str, record: dict) -> None:
    epoch = int(record["epoch"])
    directory = Path(output_dir) / "epoch_records"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"epoch-{epoch:04d}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output.name}.", suffix=".tmp", dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(record), handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        with output.open(encoding="utf-8") as handle:
            persisted = json.load(handle)
        if int(persisted.get("epoch", -1)) != epoch:
            raise ValueError(f"durable epoch record verification failed: {output}")
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_analysis_checkpoint(
    output_dir: str,
    model_without_ddp,
    epoch: int,
    selection_metric: str,
    selection_value: float,
    val_stats: dict,
    test_stats: dict,
) -> None:
    output = Path(output_dir) / "checkpoint-best-analysis.pth"
    temporary = output.with_name(f"{output.name}.tmp")
    payload = {
        "format": "analysis-best-v1",
        "epoch": int(epoch),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "validation": _json_safe(val_stats),
        "test": _json_safe(test_stats),
        "model": model_without_ddp.state_dict(),
    }
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------ #
#  Channel name mapping per dataset (loaded from UnifiedEEGDataset_v2 rules)
# ------------------------------------------------------------------ #

def get_ch_names_from_rule(ds_type):
    """Look up channel names from UnifiedEEGDataset_v2.DATASET_RULES."""
    rules = UnifiedEEGDataset_v2.DATASET_RULES
    if ds_type not in rules:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    ch_names = rules[ds_type].get('ch_names')
    if ch_names is None:
        raise ValueError(f"Dataset {ds_type} has no fixed ch_names in rules")
    return list(ch_names)


def get_label_names_from_rule(ds_type, sub_dataset=None):
    """Look up label names from UnifiedEEGDataset_v2.DATASET_RULES or EEGIMAGE_LABEL_DICT."""
    rules = UnifiedEEGDataset_v2.DATASET_RULES
    if ds_type == 'eegimage':
        if sub_dataset is None:
            raise ValueError("eegimage requires a sub_dataset")
        labels = UnifiedEEGDataset_v2.EEGIMAGE_LABEL_DICT.get(sub_dataset)
        if labels is None:
            raise ValueError(f"Unknown eegimage sub-dataset: {sub_dataset}")
        return list(labels)
    label_names = rules[ds_type].get('label_names')
    if label_names is None:
        raise ValueError(f"Dataset {ds_type} has no fixed label_names in rules")
    return list(label_names)


# ------------------------------------------------------------------ #
#  Wrapper: UnifiedEEGDataset_v2 → (eeg_flat, label) tuples
# ------------------------------------------------------------------ #

class UnifiedFinetuneWrapper(torch.utils.data.Dataset):
    """Convert UnifiedEEGDataset_v2 dict output → (eeg, label) tuples for finetuning.

    Output:
        eeg: FloatTensor [C, N*200] — flattened along (n_patches, patch_size)
        label: int
    """

    def __init__(self, unified_dataset, canonical_ch=None, canonical_n=None, label_remap=None, num_classes=None):
        self.dataset = unified_dataset
        self.label_remap = label_remap
        self.num_classes = num_classes
        # RC4: 逐样本按通道名对齐到 canonical_ch, patch数对齐到 canonical_n (处理跨recording变通道/变时长)
        self.canonical_ch = [str(c).strip().upper() for c in canonical_ch] if canonical_ch else None
        self.canonical_n = canonical_n

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        eeg = sample['eeg']  # [C, N, 200] or [C, T]
        label = sample['label']

        # RC4 通道对齐: 按通道名选择/重排/补零到 canonical_ch
        if self.canonical_ch is not None and eeg.dim() >= 2:
            samp_ch = [str(c).strip().upper() for c in sample.get('meta', {}).get('ch_names', [])]
            if samp_ch and samp_ch != self.canonical_ch:
                pos = {c: i for i, c in enumerate(samp_ch)}
                aligned = torch.zeros((len(self.canonical_ch), *eeg.shape[1:]), dtype=eeg.dtype)
                for i, c in enumerate(self.canonical_ch):
                    j = pos.get(c)
                    if j is not None and j < eeg.shape[0]:
                        aligned[i] = eeg[j]
                eeg = aligned

        # Flatten N and T + RC4 时长对齐 (patch数裁剪/补零到 canonical_n)
        if eeg.dim() == 3:
            if self.canonical_n is not None and eeg.shape[1] != self.canonical_n:
                cn = self.canonical_n
                if eeg.shape[1] > cn:
                    eeg = eeg[:, :cn]
                else:
                    pad = torch.zeros((eeg.shape[0], cn - eeg.shape[1], eeg.shape[2]), dtype=eeg.dtype)
                    eeg = torch.cat([eeg, pad], dim=1)
            C, N, T = eeg.shape
            eeg = eeg.reshape(C, N * T)
        elif eeg.dim() == 2:
            pass  # already [C, T]
        else:
            raise ValueError(f"Unexpected eeg shape: {eeg.shape}")

        _lab = int(label)
        if self.label_remap is not None:
            if _lab not in self.label_remap:
                raise ValueError(f"label {_lab} is absent from label_remap")
            _lab = self.label_remap[_lab]
        if self.num_classes is not None and not 0 <= _lab < self.num_classes:
            raise ValueError(
                f"label {_lab} is outside [0, {self.num_classes}); "
                "check label_map.json and source labels"
            )
        return eeg.contiguous(), _lab


# ------------------------------------------------------------------ #
#  Dataset preparation
# ------------------------------------------------------------------ #

def _infer_canonical_patch_count(record, rule):
    if not isinstance(record, dict):
        return None
    eeg_key = rule.get('eeg_key')
    patch_size = rule.get('patch_size')
    if not eeg_key or not patch_size:
        return None
    eeg = record.get(eeg_key)
    if eeg is None and eeg_key != 'X':
        eeg = record.get('X')
    if eeg is None:
        return None
    shape = np.asarray(eeg).shape
    if len(shape) == 3 and shape[-1] == patch_size:
        patch_count = shape[-2]
    elif len(shape) == 2:
        patch_count = max(1, shape[1] // patch_size)
    else:
        return None
    truncate = rule.get('eeg_truncate_patches')
    if truncate:
        patch_count = min(patch_count, int(truncate))
    return int(patch_count)


def prepare_unified_dataset(dataset_config_path):
    """Load a dataset config JSON and build train/val/test datasets.

    Returns:
        train_ds, test_ds, val_ds, ch_names, nb_classes, ds_type, sub_dataset
    """
    with open(dataset_config_path, 'r') as f:
        configs = json.load(f)

    if len(configs) != 1:
        raise ValueError(
            f"Unified finetuning requires exactly 1 dataset in config, "
            f"got {len(configs)} in {dataset_config_path}"
        )

    cfg = configs[0]
    ds_type = cfg['type']
    sub_dataset = None
    if ds_type == 'eegimage':
        subs = cfg.get('datasets', [])
        if len(subs) != 1:
            raise ValueError(
                f"eegimage finetuning requires exactly 1 sub-dataset, got {subs}"
            )
        sub_dataset = subs[0]

    print(f"Dataset config: {dataset_config_path}")
    print(f"  type={ds_type}, sub_dataset={sub_dataset}, data_dir={cfg['data_dir']}")

    # Build train / test / val datasets via UnifiedEEGDataset_v2
    cache_dir = cfg.get('cache_dir')
    train_unified = UnifiedEEGDataset_v2([cfg], mode='train', balanced_sampling=False, cache_dir=cache_dir)
    test_unified = UnifiedEEGDataset_v2([cfg], mode='test', balanced_sampling=False, cache_dir=cache_dir)

    # Unit-fixed EEGImage datasets provide an explicit eval split; legacy ones do not.
    if ds_type == 'eegimage':
        has_eval = bool(sub_dataset) and os.path.isdir(os.path.join(cfg['data_dir'], sub_dataset, 'eval'))
    else:
        has_eval = True
    if has_eval:
        try:
            val_unified = UnifiedEEGDataset_v2([cfg], mode='eval', balanced_sampling=False, cache_dir=cache_dir)
            if len(val_unified) == 0:
                print(f"  WARNING: eval split is empty, falling back to test as val")
                val_unified = test_unified
        except Exception as e:
            print(f"  WARNING: failed to load eval split ({e}), falling back to test")
            val_unified = test_unified
    else:
        print(f"  NOTE: {ds_type}/{sub_dataset} has no val split — using test set as val")
        val_unified = test_unified

    # Channel names + canonical shape (先算, 供 wrapper 做 RC4 通道/时长对齐)
    import glob, pickle, json as _json
    import numpy as _np
    # 优先用已加载的 train_unified.index[0] (走缓存, 避免 glob 大目录 listdir 失败, 如 Siena 33605)
    if len(train_unified.index) > 0 and isinstance(train_unified.index[0], dict) and 'file_path' in train_unified.index[0]:
        _fp0 = train_unified.index[0]['file_path']
        if cfg.get('index_path'):
            _d0, _, _ = train_unified._read_one(train_unified.index[0])
        else:
            _d0 = (__import__('torch').load(_fp0, map_location='cpu', weights_only=False) if _fp0.endswith('.pt') else pickle.load(open(_fp0, 'rb')))
    else:
        _rule_dp = UnifiedEEGDataset_v2.DATASET_RULES.get(ds_type, {}).get('dir_pattern')
        _train_sub = _rule_dp.get('train', 'train') if isinstance(_rule_dp, dict) else 'train'
        _fs = sorted(glob.glob(os.path.join(cfg['data_dir'], _train_sub, '*.pkl')))
        _d0 = pickle.load(open(_fs[0], 'rb'))
    _rule = UnifiedEEGDataset_v2.DATASET_RULES.get(ds_type, {})
    canonical_rule = _rule
    if ds_type == 'downstream':
        source_type = str(cfg.get('source_dataset_type', '')).lower()
        canonical_rule = UnifiedEEGDataset_v2.DATASET_RULES.get(source_type, _rule)
    _canon_n = _infer_canonical_patch_count(_d0, canonical_rule)
    if ds_type == 'downstream':
        lm = os.path.join(cfg['data_dir'], 'label_map.json')
        label_values = cfg.get('label_values')
        label_names = (
            [f'class_{int(value)}' for value in label_values]
            if isinstance(label_values, list) and label_values
            else list(_json.load(open(lm)).keys()) if os.path.exists(lm) else None
        )
        ch_names = _d0['ch_names']
        nb_classes = len(label_names) if label_names else 2
    elif ds_type == 'seedimg':
        ch_names = get_ch_names_from_rule('seedimg')
        label_names = list(_d0.get('candidate_labels', []))
        nb_classes = len(label_names) if label_names else 3
        _canon_n = 4
    else:
        try:
            ch_names = get_ch_names_from_rule(ds_type)
        except ValueError:  # RC6: 规则无固定ch_names(如bcic2020_3/bcicv_2a) → 从首个train pkl读
            ch_names = _d0['ch_names']
        label_names = get_label_names_from_rule(ds_type, sub_dataset)
        nb_classes = len(label_names)

    # label remap: 原始label值(如class_4,5,6,7 或不连续class_0,1,3..) -> 0..N-1 连续
    label_remap = None
    if ds_type == 'downstream' and label_names:
        try:
            _orig = sorted(int(str(_k).replace('class_', '')) for _k in label_names)
            if _orig != list(range(len(_orig))):
                label_remap = {_v: _i for _i, _v in enumerate(_orig)}
                print(f"  [label_remap] 非连续label {_orig} -> 0..{len(_orig)-1}")
        except Exception:
            label_remap = None

    # RC4: 用 canonical 通道名+patch数 构建 wrapper, 逐样本对齐
    train_ds = UnifiedFinetuneWrapper(train_unified, ch_names, _canon_n, label_remap, nb_classes)
    val_ds = UnifiedFinetuneWrapper(val_unified, ch_names, _canon_n, label_remap, nb_classes)
    test_ds = UnifiedFinetuneWrapper(test_unified, ch_names, _canon_n, label_remap, nb_classes)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Channels: {len(ch_names)}, Classes: {nb_classes} {label_names}, canon_n={_canon_n}")

    return train_ds, test_ds, val_ds, ch_names, nb_classes, ds_type, sub_dataset, label_remap


# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required")
    results = {}
    if is_binary:
        preds = (output >= threshold).astype(int)
        has_both = (target.sum() > 0) and (target.sum() < len(target))
        for m in metrics:
            if m == 'accuracy':
                results[m] = accuracy_score(target, preds)
            elif m == 'balanced_accuracy':
                results[m] = balanced_accuracy_score(target, preds) if has_both else accuracy_score(target, preds)
            elif m == 'roc_auc':
                try:
                    results[m] = roc_auc_score(target, output) if has_both else 0.0
                except ValueError:
                    results[m] = 0.0
            elif m == 'pr_auc':
                try:
                    results[m] = average_precision_score(target, output) if has_both else 0.0
                except ValueError:
                    results[m] = 0.0
            elif m == 'f1_weighted':
                results[m] = f1_score(target, preds, average='weighted', zero_division=0)
            elif m == 'cohen_kappa':
                results[m] = cohen_kappa_score(target, preds)
    else:
        preds = np.argmax(output, axis=-1)
        target_cls = target if target.ndim == 1 else np.argmax(target, axis=-1)
        nb_classes = output.shape[-1]
        all_labels = list(range(nb_classes))
        unique = set(np.unique(target_cls)) | set(np.unique(preds))
        has_multi = len(unique) > 1
        for m in metrics:
            if m == 'accuracy':
                results[m] = accuracy_score(target_cls, preds)
            elif m == 'balanced_accuracy':
                results[m] = balanced_accuracy_score(target_cls, preds) if has_multi else accuracy_score(target_cls, preds)
            elif m == 'f1_weighted':
                results[m] = f1_score(target_cls, preds, average='weighted', zero_division=0, labels=all_labels)
            elif m == 'cohen_kappa':
                results[m] = cohen_kappa_score(target_cls, preds, labels=all_labels) if has_multi else 0.0
    return results


def get_default_metrics(nb_classes, ds_type):
    """Pick default metrics based on dataset properties."""
    is_binary = (nb_classes == 2)
    if is_binary:
        return ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "f1_weighted"]
    else:
        return ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]


# ------------------------------------------------------------------ #
#  Args
# ------------------------------------------------------------------ #

def get_args():
    parser = argparse.ArgumentParser(
        'LaBraM at Scale classification fine-tuning', add_help=True)

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument(
        '--max_numbered_checkpoints', default=0, type=int,
        help='Keep only the newest N checkpoint-<epoch>.pth files; 0 keeps all.',
    )

    # Model
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=False)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)

    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.1)

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--smoothing', type=float, default=0.1)

    # Finetuning
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--no_pool', action='store_true', default=False,
                        help='Flatten all channel-time patch tokens before the classification head')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias',
                        action='store_true', default=False)

    # Dataset (KEY CHANGE: take dataset_config JSON instead of fixed dataset name)
    parser.add_argument('--dataset_config', required=True, type=str,
                        help='path to a single-dataset config JSON')

    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--input_scale', type=float, default=1.0, help='forward输入缩放(0.2=zscore对齐/100)')
    parser.add_argument('--skip_norm', action='store_true', help='跳过forward per-sample zscore')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--save_best_analysis_checkpoint', action='store_true', default=False,
                        help='Atomically retain model weights from the validation-best epoch')

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Selection metric
    parser.add_argument('--model_selection_metric', default='balanced_accuracy', type=str,
                        choices=['accuracy', 'cohen_kappa', 'balanced_accuracy',
                                 'f1_weighted', 'roc_auc', 'pr_auc'])

    # Class-balanced loss
    parser.add_argument('--use_class_weight', action='store_true', default=False)

    # bf16
    parser.add_argument('--bf16', action='store_true', default=False)

    # Freeze layers (for frozen-probing)
    parser.add_argument('--freeze_layers', default=0, type=int,
                        help='Freeze first N transformer blocks (0 = full finetune)')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze entire backbone, only train head')
    parser.add_argument('--train_fraction', default=1.0, type=float,
                        help='Fraction of training set (data-efficiency scaling); deterministic seeded subsample')

    # TQS (Task-aware Query Selection, adapted from UniMind)
    parser.add_argument('--use_tqs', action='store_true', default=False,
                        help='Replace simple linear head with TQS module')
    parser.add_argument('--tqs_query_pool_size', default=16, type=int)
    parser.add_argument('--tqs_top_k', default=4, type=int)
    parser.add_argument('--tqs_num_heads', default=4, type=int)
    parser.add_argument('--tqs_router_dim', default=0, type=int,
                        help='If 0, use linear router; otherwise MLP hidden dim')
    parser.add_argument('--tqs_use_static_query', action='store_true', default=True)
    parser.add_argument('--tqs_no_static_query', action='store_false', dest='tqs_use_static_query')
    parser.add_argument('--tqs_dropout', default=0.1, type=float)

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Model creation
# ------------------------------------------------------------------ #

def get_models(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        use_no_pool=args.no_pool,
        no_pool_tokens=getattr(args, 'no_pool_tokens', None),
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )
    return model


# ------------------------------------------------------------------ #
#  Training engine (same logic as run_class_finetuning.py)
# ------------------------------------------------------------------ #

_INPUT_SCALE = 1.0  # forward输入缩放(对齐预训练 eeg/100 尺度)
_SKIP_NORM = False  # 跳过 forward per-sample zscore


def train_class_batch(model, samples, target, criterion, input_chans):
    outputs = model(samples, input_chans)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, target)
    return loss, outputs


def accumulation_group(
    data_iter_step: int, num_batches: int, update_freq: int
) -> tuple[int, bool]:
    if num_batches <= 0 or update_freq <= 0 or not 0 <= data_iter_step < num_batches:
        raise ValueError(
            f'invalid accumulation state: step={data_iter_step}, '
            f'batches={num_batches}, update_freq={update_freq}'
        )
    group_start = (data_iter_step // update_freq) * update_freq
    group_size = min(update_freq, num_batches - group_start)
    should_update = (data_iter_step + 1) % update_freq == 0 or data_iter_step + 1 == num_batches
    return group_size, should_update


def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, loss_scaler, max_norm=0,
                    model_ema=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    input_chans=None, is_binary=True,
                    amp_dtype=torch.float16):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    num_batches = len(data_loader)

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None:
            if data_iter_step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        if not _SKIP_NORM:
            _mu = samples.mean(dim=-1, keepdim=True)
            _sd = samples.std(dim=-1, keepdim=True) + 1e-7
            samples = ((samples - _mu) / _sd).clamp(-15, 15)
        samples = samples * _INPUT_SCALE
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss, output = train_class_batch(model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        group_size, should_update = accumulation_group(
            data_iter_step, num_batches, update_freq
        )
        loss /= group_size
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=should_update)
        if should_update:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:',
             input_chans=None, metrics=None, is_binary=True,
             amp_dtype=torch.float16):
    criterion = torch.nn.BCEWithLogitsLoss() if is_binary else torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    pred = []
    true = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0].float().to(device, non_blocking=True)
        if not _SKIP_NORM:
            _mu = samples.mean(dim=-1, keepdim=True)
            _sd = samples.std(dim=-1, keepdim=True) + 1e-7
            samples = ((samples - _mu) / _sd).clamp(-15, 15)
        samples = samples * _INPUT_SCALE
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        target = batch[1].to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            output = model(samples, input_chans)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        if is_binary:
            output_np = torch.sigmoid(output).detach().cpu().float().numpy()
            target_np = target.cpu().numpy()
        else:
            output_np = output.detach().cpu().float().numpy()
            target_np = target.cpu().numpy()

        pred.append(output_np)
        true.append(target_np)
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)

    if len(pred) == 0:
        print("  WARNING: evaluate loader为空(0样本), 返回零指标避免崩溃")
        try:
            return {k: 0.0 for k in metrics}
        except Exception:
            return {}
    pred = np.concatenate(pred, axis=0)
    true = np.concatenate(true, axis=0)

    if is_binary:
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        results = get_metrics(pred_flat, true_flat, metrics, is_binary=True)
    else:
        results = get_metrics(pred, true, metrics, is_binary=False)

    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    for k, v in results.items():
        print(f'* {k}: {v:.4f}')
        metric_logger.meters[k] = utils.SmoothedValue(window_size=1)
        metric_logger.meters[k].update(v)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main(args):
    global _INPUT_SCALE, _SKIP_NORM
    _INPUT_SCALE = args.input_scale
    _SKIP_NORM = args.skip_norm
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Dataset ----
    (dataset_train, dataset_test, dataset_val,
     ch_names, nb_classes, ds_type, sub_dataset, _label_remap) = prepare_unified_dataset(args.dataset_config)
    args._label_remap = _label_remap
    args.nb_classes = 1 if nb_classes == 2 else nb_classes
    if args.no_pool:
        canonical_n = getattr(dataset_train, 'canonical_n', None)
        canonical_ch = getattr(dataset_train, 'canonical_ch', None)
        if canonical_n is None or canonical_ch is None:
            raise ValueError('No-Pool requires canonical channel and patch dimensions')
        args.no_pool_tokens = len(canonical_ch) * int(canonical_n)
        print(f"No-Pool readout: {len(canonical_ch)} channels x {canonical_n} patches "
              f"= {args.no_pool_tokens} tokens")
    is_binary = (args.nb_classes == 1)
    metrics = get_default_metrics(nb_classes, ds_type)

    # ---- Data-efficiency: deterministic subsample of training set ----
    if getattr(args, 'train_fraction', 1.0) < 1.0:
        import numpy as _np
        _n = len(dataset_train); _k = max(1, int(round(_n * args.train_fraction)))
        _idx = sorted(_np.random.RandomState(args.seed).choice(_n, size=_k, replace=False).tolist())
        dataset_train = torch.utils.data.Subset(dataset_train, _idx)
        print(f"[train_fraction={args.train_fraction}] subsample train {_n} -> {_k} (seed={args.seed})")

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    log_writer = None  # Skip tensorboard for simplicity

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # 小数据集(<batch)也保留>=1个batch, 避免空loader除零
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None
        data_loader_test = None

    # ---- Input channels ----
    input_chans = utils.get_input_chans(ch_names)

    # ---- Model ----
    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # ---- Load pretrained checkpoint ----
    if args.finetune:
        if str(args.finetune).endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = {'model': load_file(args.finetune, device='cpu')}
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load ckpt from %s" % args.finetune)

        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if checkpoint_model is not None and args.model_filter_name != '':
            all_keys = list(checkpoint_model.keys())
            has_student_keys = any(k.startswith('student.') for k in all_keys)
            if has_student_keys:
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                checkpoint_model = new_dict
                print(f'Filtered student. keys: {len(checkpoint_model)} params')

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and (
                    k not in state_dict or checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # ---- Optionally wrap with TQS head ----
    if args.use_tqs:
        from models.tqs_module import TQSModelWrapper
        print(f"Wrapping with TQS: pool={args.tqs_query_pool_size}, top_k={args.tqs_top_k}, "
              f"heads={args.tqs_num_heads}, static={args.tqs_use_static_query}")
        model = TQSModelWrapper(
            base_encoder=model,
            num_classes=args.nb_classes,
            query_pool_size=args.tqs_query_pool_size,
            top_k=args.tqs_top_k,
            num_heads=args.tqs_num_heads,
            router_dim=args.tqs_router_dim,
            use_static_query=args.tqs_use_static_query,
            dropout=args.tqs_dropout,
        )

    if args.freeze_layers and args.freeze_backbone:
        raise ValueError("freeze_layers and freeze_backbone are mutually exclusive")
    if args.freeze_layers:
        freeze_report = freeze_first_transformer_blocks(model, args.freeze_layers)
        print(
            f"Frozen first {freeze_report['frozen_blocks']} / "
            f"{freeze_report['available_blocks']} transformer blocks "
            f"({freeze_report['frozen_parameters']:,} parameters)"
        )

    model.to(device)

    # ---- Freeze backbone if requested ----
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            # When using TQS, keep the TQS head trainable; everything else frozen
            if args.use_tqs:
                trainable_prefix = ('tqs.',)
            else:
                trainable_prefix = ('head.',)
            if not any(name.startswith(p) for p in trainable_prefix):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen backbone — trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model, decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '', resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(
        1, math.ceil(len(data_loader_train) / args.update_freq)
    )
    print(f"LR = {args.lr:.8f}")
    print(f"Total batch size = {total_batch_size}")
    print(f"Update frequency = {args.update_freq}")
    print(f"Training examples = {len(dataset_train)}")
    print(f"Training steps per epoch = {num_training_steps_per_epoch}")

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    loss_scaler = NativeScaler(enabled=(amp_dtype == torch.float16))

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    # ---- Auto resume ----
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    run_meta = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_type": ds_type,
        "sub_dataset": sub_dataset or "",
        "dataset_config": args.dataset_config,
        "finetune_ckpt": args.finetune,
        "selection_metric": args.model_selection_metric,
        "seed": args.seed,
        "epochs": args.epochs,
        "no_pool": bool(args.no_pool),
        "no_pool_tokens": getattr(args, 'no_pool_tokens', None),
        "auto_resume": bool(args.auto_resume),
        "resume_path": args.resume,
        "start_epoch": int(args.start_epoch),
    }

    # ---- Loss ----
    class_weight = None
    if args.use_class_weight and not args.eval:
        counts = np.asarray(training_class_counts(dataset_train, nb_classes), dtype='float64')
        safe_counts = np.maximum(counts, 1.0)
        weights = safe_counts.sum() / (nb_classes * safe_counts)
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print('Training class counts:', counts.astype(int).tolist(), 'weights:', weights.tolist())

    if is_binary:
        if class_weight is not None and len(class_weight) >= 2:
            _pw = (class_weight[1] / class_weight[0]).detach().reshape(1)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_pw)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    else:
        if class_weight is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight,
                                                  label_smoothing=(args.smoothing if args.smoothing > 0 else 0.0))
        elif args.smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        test_stats = evaluate(data_loader_test, model, device,
                              input_chans=input_chans, metrics=metrics, is_binary=is_binary,
                              amp_dtype=amp_dtype)
        if args.output_dir and utils.is_main_process():
            eval_payload = {
                **run_meta,
                "status": "eval_only",
                "test": {k: v for k, v in test_stats.items() if k != 'loss'},
                "test_loss": test_stats.get('loss', None),
            }
            write_results_json(args.output_dir, eval_payload)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    best_metric_val = -float('inf')
    best_epoch = -1
    best_val_stats = {}
    best_test_stats = {}
    last_val_stats = {}
    last_test_stats = {}
    sel_metric = args.model_selection_metric
    previous_best = best_completed_epoch(args.output_dir, args.start_epoch, sel_metric)
    if previous_best is not None:
        best_epoch = previous_best['epoch']
        best_metric_val = previous_best['selection_value']
        best_val_stats = previous_best['validation']
        best_test_stats = previous_best['test']
        max_accuracy = best_val_stats.get('accuracy', 0) * 100
        max_accuracy_test = best_test_stats.get('accuracy', 0) * 100
        print(f'Restored validation-best epoch {best_epoch}: {sel_metric}={best_metric_val:.6f}')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            input_chans=input_chans, is_binary=is_binary,
            amp_dtype=amp_dtype,
        )

        is_new_best = False
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:',
                                 input_chans=input_chans, metrics=metrics, is_binary=is_binary,
                                 amp_dtype=amp_dtype)
            test_stats = evaluate(data_loader_test, model, device, header='Test:',
                                   input_chans=input_chans, metrics=metrics, is_binary=is_binary,
                                   amp_dtype=amp_dtype)
            last_val_stats = {k: v for k, v in val_stats.items()}
            last_test_stats = {k: v for k, v in test_stats.items()}

            curr_val_metric = val_stats.get(sel_metric, -float('inf'))
            if curr_val_metric > best_metric_val:
                is_new_best = True
                best_metric_val = curr_val_metric
                best_epoch = epoch
                best_val_stats = {k: v for k, v in val_stats.items() if k != 'loss'}
                best_test_stats = {k: v for k, v in test_stats.items() if k != 'loss'}
                max_accuracy = val_stats.get('accuracy', 0) * 100
                max_accuracy_test = test_stats.get('accuracy', 0) * 100

            print(f'Best val {sel_metric}: {best_metric_val:.4f} (ep{best_epoch}), '
                  f'val_acc: {max_accuracy:.2f}%, test_acc: {max_accuracy_test:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            write_durable_epoch_record(args.output_dir, log_stats)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.flush()
                os.fsync(f.fileno())

        if is_new_best and args.output_dir and args.save_best_analysis_checkpoint \
                and utils.is_main_process():
            write_analysis_checkpoint(
                args.output_dir, model_without_ddp, epoch, sel_metric, curr_val_metric,
                val_stats, test_stats,
            )

        # Advance the durable checkpoint only after validation/test are logged.
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # ====== Final summary ======
    if best_epoch >= 0:
        print("\n" + "=" * 60)
        print("  BEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Dataset: {ds_type}" + (f"/{sub_dataset}" if sub_dataset else ""))
        print(f"  Best epoch: {best_epoch}")
        print(f"  --- Val ---")
        for k, v in best_val_stats.items():
            print(f"    {k}: {v:.4f}")
        print(f"  --- Test (at best val epoch) ---")
        for k, v in best_test_stats.items():
            print(f"    {k}: {v:.4f}")
        print("=" * 60 + "\n")

        if args.output_dir and utils.is_main_process():
            best_summary = {
                "dataset_type": ds_type,
                "sub_dataset": sub_dataset or "",
                "best_epoch": best_epoch,
                **{f"best_val_{k}": v for k, v in best_val_stats.items()},
                **{f"best_test_{k}": v for k, v in best_test_stats.items()},
            }
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(best_summary) + "\n")


    # Resume edge case: if start_epoch >= epochs, loop is skipped and we still
    # need one deterministic eval artifact for reliable aggregation.
    if data_loader_val is not None and best_epoch < 0:
        print("No training epoch executed; running one final val/test evaluation for stable outputs.")
        val_stats = evaluate(data_loader_val, model, device, header='Val(final):',
                             input_chans=input_chans, metrics=metrics, is_binary=is_binary,
                             amp_dtype=amp_dtype)
        test_stats = evaluate(data_loader_test, model, device, header='Test(final):',
                              input_chans=input_chans, metrics=metrics, is_binary=is_binary,
                              amp_dtype=amp_dtype)
        best_epoch = int(args.start_epoch) - 1
        best_metric_val = val_stats.get(sel_metric, -float('inf'))
        best_val_stats = {k: v for k, v in val_stats.items() if k != 'loss'}
        best_test_stats = {k: v for k, v in test_stats.items() if k != 'loss'}
        last_val_stats = {k: v for k, v in val_stats.items()}
        last_test_stats = {k: v for k, v in test_stats.items()}

    if args.output_dir and utils.is_main_process():
        final_payload = {
            **run_meta,
            "status": "finished",
            "best_epoch": int(best_epoch),
            "best_val_selection_metric": float(best_metric_val) if best_metric_val != -float('inf') else None,
            "best_val": best_val_stats,
            "best_test": best_test_stats,
            "last_val": {k: v for k, v in last_val_stats.items() if k != 'loss'},
            "last_test": {k: v for k, v in last_test_stats.items() if k != 'loss'},
            "training_time_sec": float(total_time),
        }
        write_results_json(args.output_dir, final_payload)

if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

