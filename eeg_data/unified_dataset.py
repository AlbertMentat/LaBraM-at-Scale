"""
Unified EEG dataset classes for multi-dataset joint training.

Contains:
  - UnifiedEEGDataset_v2: Multi-dataset loader with rules-table-driven design
  - GroupedBatchSampler: Batch sampler that groups by EEG shape
  - unified_collate_fn_v2: Collate function for UnifiedEEGDataset_v2
  - ThingsEEGDataset_2: Single ThingsEEG dataset loader (per-sample .pt files)
  - custom_collate_fn_things_2: Collate function for ThingsEEGDataset_2

Extracted from AVDE_v1/utils.py to avoid cross-directory imports.
"""

import json
import logging
import os
import pickle
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Helpers
# ====================================================================== #

def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ====================================================================== #
#  UnifiedEEGDataset_v2
# ====================================================================== #

def _robust_listdir(d, max_tries=8, sleep=2.0):
    """blobfuse 列大目录(万级文件)可能返回不全/空; 反复列取并集直到稳定, 缓解 RC3 大目录扫描不全。"""
    import time as _t
    seen = set(); prev = -1
    for _i in range(max_tries):
        try:
            cur = os.listdir(d)
        except OSError:
            cur = []
        seen.update(cur)
        if len(seen) > 0 and len(seen) == prev:
            break
        prev = len(seen)
        if _i < max_tries - 1:
            _t.sleep(sleep)
    return sorted(seen)


class UnifiedEEGDataset_v2(Dataset):
    """
    Unified EEG Dataset V2 — 高性能版本

    相比 V1 的改进:
      1. **不再实例化子 Dataset 类** — 直接扫描目录 + 规则表驱动加载，初始化从分钟级降到秒级
      2. **索引缓存** — 首次扫描结果自动缓存为 .pkl，第二次启动 <1s
      3. **所有数据集都有图像** — TUEV/TUAB/HMC 使用 AVDE 生成的图像，统一输出 (eeg, img, label, meta)
      4. **规则表驱动** — 一个 __getitem__ 处理所有格式，无中间包装层
      5. **配合 GroupedBatchSampler** — 按 EEG shape 分组 batch，零 padding 开销

    支持的数据集:
      - tuev:      [23, 5, 200]   6-class event classification  (.pkl, _with_images 版本)
      - tuab:      [23, 10, 200]  2-class normal/abnormal        (.pkl, _with_images 版本)
      - hmc:       [4, 16, 200]   5-class sleep stage            (.pkl, _with_images 版本)
      - eegimage:  [62, 4, 200]   emotion classification         (.pt)
      - things:    [17, 1, T]     object recognition             (.pt, ThingsEEG v2 后验区)
      - things_v1: [63, 1, 200]   object recognition             (.pt, ThingsEEG v1 全头皮)
      - alljoined: [C, 1, T]      object recognition             (.pt)
      - eegpt_pkl: [C, N, 200]   unlabeled pretraining          (.pkl, EEGPT TUH 数据)

    输出格式 (统一):
        {
            'eeg':   Tensor [C, N, 200],
            'img':   Tensor [3, 256, 256],
            'label': int,
            'meta':  { dataset, subject, sample_id, true_label, ch_names, prompt, ... }
        }

    用法:
        dataset_configs = [
            {'type': 'tuev',  'data_dir': '/path/to/tuev_with_images',  'weight': 1.0},
            {'type': 'tuab',  'data_dir': '/path/to/tuab_with_images',  'weight': 1.0},
            {'type': 'hmc',   'data_dir': '/path/to/hmc_with_images',   'weight': 0.5},
            {'type': 'eegimage', 'data_dir': '/path/to/eegimage',       'weight': 1.0},
            {'type': 'things', 'data_dir': '/path/to/things',           'weight': 1.0},
            {'type': 'things_v1', 'data_dir': '/path/to/things_v1',     'weight': 1.0, 'subjects': ['sub-01']},
            {'type': 'alljoined', 'data_dir': '/path/to/alljoined',     'weight': 1.0},
        ]
        dataset = UnifiedEEGDataset_v2(dataset_configs, mode='train')
        sampler = GroupedBatchSampler(dataset, batch_size=32)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=unified_collate_fn_v2)
    """

    # ------------------------------------------------------------------ #
    #  数据集规则表
    # ------------------------------------------------------------------ #
    DATASET_RULES = {
        'downstream': {
            'file_format': 'pkl', 'eeg_key': 'X', 'image_key': 'generated_image',
            'label_key': 'y', 'label_is_array': False, 'label_offset': 0,
            'eeg_reshape': 'auto', 'eeg_truncate_patches': None, 'patch_size': 200,
            'ch_names': None, 'label_names': None, 'prompt': None,
            'dir_pattern': {'train': 'train', 'eval': 'val', 'test': 'test'},
            'eeg_shape_group': None,
        },
        'tuev': {
            'file_format': 'pkl',
            'eeg_key': 'signal',
            'image_key': 'generated_image',
            'label_key': 'label',
            'label_is_array': True,     # label 字段是数组，需要取 [0]
            'label_offset': -1,         # 原始 1-6 → 0-5
            'eeg_reshape': 'auto',      # reshape to [C, -1, 200]
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ',
                         'PZ', 'T1', 'T2'],
            'label_names': [
                'Spike and Sharp Wave',
                'Generalized Periodic Epileptiform Discharge',
                'Periodic Lateralized Epileptiform Discharge',
                'Eye Movement',
                'Artifact',
                'Background',
            ],
            'prompt': (
                "You should classify the event type of this EEG segment.\n"
                "The possible event types are strictly limited to:\n"
                "    • Spike and Sharp Wave — abnormal transient waveforms indicating cortical irritability\n"
                "    • Generalized Periodic Epileptiform Discharge — repetitive sharp discharges seen bilaterally\n"
                "    • Periodic Lateralized Epileptiform Discharge — repetitive sharp discharges lateralized to one hemisphere\n"
                "    • Eye Movement — ocular artifact from saccades or blinks\n"
                "    • Artifact — non-cerebral signal contamination\n"
                "    • Background — normal ongoing cerebral activity\n"
                "Respond with exactly one phrase from the options above."
            ),
            'dir_pattern': {'train': 'train/processed_train', 'eval': 'eval/processed_eval', 'test': 'test/processed_test'},
            'eeg_shape_group': (23, 5),   # (ch_count, n_patches) — 用于分组 batch
        },
        'tuab': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': 'generated_image',
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ',
                         'PZ', 'T1', 'T2'],
            'label_names': ['Normal', 'Abnormal'],
            'prompt': (
                "You should classify whether this EEG recording is normal or abnormal.\n"
                "The possible classifications are strictly limited to:\n"
                "    • Normal — EEG recording within normal physiological range\n"
                "    • Abnormal — EEG recording with pathological patterns such as slowing, epileptiform discharges, or asymmetry\n"
                "Respond with exactly one word from: Normal, Abnormal."
            ),
            'dir_pattern': {'train': 'train/train', 'eval': 'test/test', 'test': 'test/test'},
            'eeg_shape_group': (23, 10),
        },
        'hmc': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': 'generated_image',
            'label_key': 'y',
            'label_is_array': True,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': 16,   # 只保留前 16 个 patch
            'patch_size': 200,
            'ch_names': ['F4', 'C4', 'O2', 'C3'],
            'label_names': ['Wake', 'Light Sleep N1', 'Light Sleep N2', 'Deep Sleep N3', 'REM Sleep'],
            'prompt': (
                "You should classify the sleep stage of the subject.\n"
                "The possible sleep stages are strictly limited to:\n"
                "    • Wake — alert state with alpha or beta rhythms and high muscle tone\n"
                "    • Light Sleep N1 — drowsy transitional stage with theta waves and slow eye movements\n"
                "    • Light Sleep N2 — stable light sleep marked by sleep spindles and K-complexes\n"
                "    • Deep Sleep N3 — slow-wave sleep dominated by high-amplitude delta waves\n"
                "    • REM Sleep — rapid eye movement sleep with low-amplitude mixed-frequency activity\n"
                "Respond with exactly one phrase from the options above."
            ),
            'dir_pattern': {'train': 'train/train', 'eval': 'eval/eval', 'test': 'test/test'},
            'eeg_shape_group': (4, 16),
        },
        'eegimage': {
            'file_format': 'pt',
            'eeg_key': 'eeg',
            'image_key': 'image',
            'label_key': '__eegimage__',  # 特殊处理
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': None,          # 已经是 [62, 4, 200]，无需 reshape
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': [
                'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                'CB1', 'O1', 'OZ', 'O2', 'CB2',
            ],  # 62 channels
            'label_names': None,  # 由子数据集决定
            'prompt': None,       # 由子数据集决定
            'dir_pattern': '__eegimage__',  # 特殊目录结构
            'eeg_shape_group': (62, 4),
        },
        'seedimg': {
            'file_format': 'pt',
            'eeg_key': 'eeg',
            'image_key': 'image',
            'label_key': '__eegimage__',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': None,
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': [
                'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                'CB1', 'O1', 'OZ', 'O2', 'CB2',
            ],
            'label_names': None,
            'prompt': None,
            'dir_pattern': {'train': 'train/train', 'eval': 'test/test', 'test': 'test/test'},
            'eeg_shape_group': (62, 4),
        },
        'things': {
            'file_format': 'pt',
            'eeg_key': 'eeg',
            'image_key': 'image',
            'label_key': 'label',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': None,  # 保持原样 [C, 1, T] or [C, T]→unsqueeze
            'eeg_truncate_patches': None,
            'patch_size': None,
            'ch_names': [
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8',
                'P1', 'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2',
            ],  # 17 channels, posterior region
            'label_names': None,  # 由 metadata.json 动态构建
            'prompt': None,       # 由 metadata 构建
            'dir_pattern': '__things__',
            'eeg_shape_group': (17, 1),   # ThingsEEG v2 后验区 17 ch
        },
        'things_v1': {
            'file_format': 'pt',          # 与 things 相同的 per-sample .pt 格式
            'eeg_key': 'eeg',
            'image_key': 'image',
            'label_key': 'label',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': None,
            'eeg_truncate_patches': None,
            'patch_size': None,
            'ch_names': [
                'FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
                'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8',
                'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
                'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
                'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                'O1', 'OZ', 'O2',
            ],  # 63 channels, full scalp
            'label_names': None,
            'prompt': None,
            'dir_pattern': '__things_v1__',
            'eeg_shape_group': (63, 1),   # ThingsEEG v1 全头皮 63 通道
        },
        'alljoined': {
            'file_format': 'pt',
            'eeg_key': 'eeg',
            'image_key': 'image',
            'label_key': '__alljoined__',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': None,
            'eeg_truncate_patches': None,
            'patch_size': None,
            'ch_names': [
                'FCZ', 'AFZ', 'FP1', 'F5', 'F1',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
                'PO7', 'PO3', 'O1', 'PZ', 'POZ', 'OZ',
                'CZ',
                'O2', 'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2',
                'CP2', 'CP4', 'CP6', 'F2', 'F6', 'FP2',
            ],  # 32 channels
            'label_names': None,
            'prompt': None,
            'dir_pattern': '__alljoined__',
            'eeg_shape_group': (32, 1),   # Alljoined 实际 32 ch
        },
        'eegpt_pkl': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,           # 无图像
            'label_key': None,           # 无标签（纯预训练数据）
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',       # (C, T) → (C, N, 200)
            'eeg_truncate_patches': 16,    # 截断到 16 patches，匹配 time_embed 大小
            'patch_size': 200,
            'ch_names': None,            # 从文件读取
            'label_names': None,
            'prompt': None,
            'dir_pattern': None,         # flat directory
            # 注意：不设置 eeg_shape_group，使用 entry 级别的 shape_group
        },
        # ---- Added: TUSZ (seizure, 23ch, binary) ----
        'tusz': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ',
                         'PZ', 'T1', 'T2'],
            'label_names': ['Non-seizure', 'Seizure'],
            'prompt': "Determine whether this EEG segment contains a seizure event.\nRespond with exactly one word from: Non-seizure, Seizure.",
            'dir_pattern': {'train': 'train', 'eval': 'val', 'test': 'test'},
            'eeg_shape_group': (23, 12),
        },
        # ---- Added: TUSL (slow-event, 23ch, 3-class) ----
        'tusl': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': True,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ',
                         'PZ', 'T1', 'T2'],
            'label_names': ['Background', 'Seizure', 'Slow'],
            'prompt': "Classify this EEG segment as Background, Seizure, or Slow.\nRespond with one word.",
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
            'eeg_shape_group': (23, 10),
        },
        # ---- Added: EEGWorkload (cognitive workload, 19ch, binary) ----
        'workload': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
                         'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ', 'PZ'],
            'label_names': ['Low workload', 'High workload'],
            'prompt': "Classify the cognitive workload of this EEG segment.\nRespond with 'Low workload' or 'High workload'.",
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
            'eeg_shape_group': (19, 4),
        },
        # ---- Added: PhysioNet-MI (motor imagery, 64ch, 5-class) ----
        'physionet': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'],
            'label_names': ['Rest', 'Left fist', 'Right fist', 'Both fists', 'Both feet'],
            'prompt': "Classify the motor imagery task.\nRespond with one of: Rest, Left fist, Right fist, Both fists, Both feet.",
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
            'eeg_shape_group': (64, 3),
        },
        # ---- Added: FACED (emotion, 32ch, 9-class) ----
        'faced': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
                         'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
                         'P7', 'P3', 'PZ', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'OZ', 'O2',
                         'AF3', 'AF4'],
            'label_names': ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral',
                            'Amusement', 'Inspiration', 'Joy', 'Tenderness'],
            'prompt': "Classify the emotion evoked by the video.\nRespond with one word.",
            'dir_pattern': {'train': 'train/train', 'eval': 'eval/eval', 'test': 'test/test'},
            'eeg_shape_group': (32, 10),
        },
        # ---- Added: SHU-MI (motor imagery, 32ch, binary) ----
        'shumi': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC3', 'FCZ', 'FC4',
                         'T7', 'C3', 'CZ', 'C4', 'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3',
                         'PZ', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'OZ', 'O2',
                         'FC1', 'FC2', 'CP1', 'CP2'],
            'label_names': ['Left hand', 'Right hand'],
            'prompt': "Identify which hand the subject is imagining moving.\nRespond with one of: Left hand, Right hand.",
            'dir_pattern': {'train': 'train/train', 'eval': 'eval/eval', 'test': 'test/test'},
            'eeg_shape_group': (32, 4),
        },
        # ---- Added: Siena (seizure, 29ch, binary) ----
        'siena': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': 16,
            'patch_size': 200,
            'ch_names': ['FP1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'FC1', 'FC5',
                         'CP1', 'CP5', 'F9', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4',
                         'O2', 'F8', 'T4', 'T6', 'FC2', 'FC6', 'CP2', 'CP6', 'F10'],
            'label_names': ['Non-seizure', 'Seizure'],
            'prompt': "Determine whether this EEG segment contains a seizure event.\nRespond with one word.",
            'dir_pattern': {'train': 'train/train', 'eval': 'eval/eval', 'test': 'test/test'},
            'eeg_shape_group': (29, 10),
        },
        # ---- Added: BCIC2020-3 (imagined speech, 64ch, 5-class) ----
        'bcic2020_3': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            # Using minimal 10-20 subset; actual data has 64 ch — rule is overridable at scan time
            'ch_names': None,  # loaded from pkl
            'label_names': ['hello', 'help me', 'stop', 'thank you', 'yes'],
            'prompt': "Classify the imagined word.\nRespond with one of: hello, help me, stop, thank you, yes.",
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
            'eeg_shape_group': None,  # resolved dynamically
        },
        # ---- Added: BCICV_2a (motor imagery, 22ch, 4-class) ----
        'bcicv_2a': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': None,
            'label_key': 'y',
            'label_is_array': False,
            'label_offset': 0,
            'eeg_reshape': 'auto',
            'eeg_truncate_patches': None,
            'patch_size': 200,
            'ch_names': None,   # loaded from pkl
            'label_names': ['Left hand', 'Right hand', 'Feet', 'Tongue'],
            'prompt': "Identify the motor imagery class.\nRespond with one of: Left hand, Right hand, Feet, Tongue.",
            'dir_pattern': {'train': 'train', 'eval': 'val', 'test': 'test'},
            'eeg_shape_group': None,
        },
    }

    # EEGImage 子数据集的标签/prompt 映射
    EEGIMAGE_LABEL_DICT = {
        "SEED": ['Negative', 'Neutral', 'Positive'],
        "SEED-IV": ['Neutral', 'Sad', 'Fear', 'Happy'],
        "SEED-V": ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy'],
        "SEED-VII": ['Disgust', 'Fear', 'Sad', 'Happy', 'Neutral', 'Anger', 'Surprise'],
        "SEED-DV": ['Neutral', 'Negative', 'Positive'],
        "Game": ["Negative", "Positive"],
    }

    EEGIMAGE_PROMPT_DICT = {
        "SEED": (
            "You should classify the emotional state of the subject who is watching the video.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Negative — negative valence such as sadness or displeasure\n"
            "    • Neutral — calm or indifferent state with no strong valence\n"
            "    • Positive — positive valence such as happiness or excitement\n"
            "Respond with exactly one word from: Negative, Neutral, Positive."
        ),
        "SEED-IV": (
            "You should classify the emotional state of the subject who is watching the video.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Neutral — calm or indifferent state\n"
            "    • Sad — feelings of sorrow or unhappiness\n"
            "    • Fear — feelings of threat or anxiety\n"
            "    • Happy — feelings of joy or pleasure\n"
            "Respond with exactly one word from: Neutral, Sad, Fear, Happy."
        ),
        "SEED-V": (
            "You should classify the emotional state of the subject who is watching the video.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Disgust — feelings of revulsion or strong disapproval\n"
            "    • Fear — feelings of threat or anxiety\n"
            "    • Sad — feelings of sorrow or unhappiness\n"
            "    • Neutral — calm or indifferent state\n"
            "    • Happy — feelings of joy or pleasure\n"
            "Respond with exactly one word from: Disgust, Fear, Sad, Neutral, Happy."
        ),
        "SEED-VII": (
            "You should classify the emotional state of the subject who is watching the video.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Disgust — feelings of revulsion or strong disapproval\n"
            "    • Fear — feelings of threat or anxiety\n"
            "    • Sad — feelings of sorrow or unhappiness\n"
            "    • Happy — feelings of joy or pleasure\n"
            "    • Neutral — calm or indifferent state\n"
            "    • Anger — feelings of hostility or irritation\n"
            "    • Surprise — reaction to unexpected events\n"
            "Respond with exactly one word from: Disgust, Fear, Sad, Happy, Neutral, Anger, Surprise."
        ),
        "SEED-DV": (
            "You should classify the emotional state of the subject who is watching the video.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Negative — negative valence such as sadness or displeasure\n"
            "    • Neutral — calm or indifferent state with no strong valence\n"
            "    • Positive — positive valence such as happiness or excitement\n"
            "Respond with exactly one word from: Negative, Neutral, Positive."
        ),
        "Game": (
            "You should classify the emotional state of the subject who is playing the game.\n"
            "The possible emotions are strictly limited to:\n"
            "    • Negative — negative valence such as frustration or displeasure\n"
            "    • Positive — positive valence such as enjoyment or excitement\n"
            "Respond with exactly one word from: Negative, Positive."
        ),
    }

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        mode: str = "train",
        balanced_sampling: bool = True,
        max_samples_per_dataset: Optional[int] = None,
        cache_dir: Optional[str] = None,
        force_rebuild_cache: bool = False,
        image_size: int = 256,
    ):
        self.mode = mode
        self.balanced_sampling = balanced_sampling
        self.max_samples_per_dataset = max_samples_per_dataset
        self.image_size = image_size
        self.dataset_configs = dataset_configs
        # 本地样本文件缓存目录(通常指向节点本地SSD): 命中先读本地, 未命中拷blob到本地再读.
        # 空/None=关闭(直读blob). 由 env EVAL_CACHE_DIR 或 WDS_CACHE_DIR 提供.
        self.data_cache_dir = (os.environ.get('EVAL_CACHE_DIR', '')
                               or os.environ.get('WDS_CACHE_DIR', '') or '').strip() or None

        # ---- 构建/加载索引 ----
        self.index: List[Dict] = []          # 全局扁平索引
        self.shape_groups: Dict[Tuple, List[int]] = {}  # (ch, patches) → [idx...]

        cache_path = self._get_cache_path(dataset_configs, mode, cache_dir)

        if not force_rebuild_cache and cache_path and os.path.exists(cache_path):
            t0 = __import__('time').time()
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.index = cached['index']
            if is_main_process():
                logger.info(f"UnifiedEEGDataset_v2: Loaded {len(self.index)} samples from cache "
                            f"in {__import__('time').time() - t0:.2f}s — {cache_path}")
        else:
            t0 = __import__('time').time()
            self.index = self._build_index(dataset_configs)
            if is_main_process():
                logger.info(f"UnifiedEEGDataset_v2: Built index with {len(self.index)} samples "
                            f"in {__import__('time').time() - t0:.2f}s")
            if len(self.index) == 0 and is_main_process():
                logger.warning('UnifiedEEGDataset_v2: 索引为空(0样本)! 可能blobfuse列目录失败/挂载未就绪; 不写缓存以免污染, 依赖重试重扫.')
            if cache_path and is_main_process() and len(self.index) > 0:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_path = cache_path + f".tmp.{os.getpid()}"
                    with open(tmp_path, 'wb') as f:
                        pickle.dump({'index': self.index}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, cache_path)
                    logger.info(f"  Cached index to {cache_path}")
                except OSError as e:
                    logger.warning(f"  Failed to write cache {cache_path}: {e}")
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

        # A late rank can see the cache just published by rank 0. Both cache
        # paths must join the same collective before any rank proceeds to DDP.
        if dist.is_initialized():
            dist.barrier()

        # ---- 重新注入当前 config 的 weight (缓存权重无关 → scaling law 改权重复用索引缓存) ----
        self._reinject_weights(dataset_configs)

        # ---- 均衡采样 / 截断 ----
        if balanced_sampling and mode == "train":
            self.index = self._apply_balanced_sampling(self.index, dataset_configs)
        elif max_samples_per_dataset is not None:
            self.index = self._apply_max_cap(self.index, max_samples_per_dataset)

        # ---- 构建 shape group (用于 GroupedBatchSampler) ----
        self._build_shape_groups()

        # ---- 预缓存 prompt ----
        self._prompt_cache: Dict[Tuple[str, int], str] = {}

        if is_main_process():
            ds_counts = {}
            for entry in self.index:
                ds_id = entry.get('dataset_id', entry['dataset_type'])
                ds_counts[ds_id] = ds_counts.get(ds_id, 0) + 1
            logger.info(f"UnifiedEEGDataset_v2: mode={mode}, total={len(self.index)}, "
                        f"n_datasets={len(ds_counts)}, shape_groups={len(self.shape_groups)}")
            for ds_id, count in sorted(ds_counts.items()):
                logger.info(f"  {ds_id}: {count} samples")

    # ------------------------------------------------------------------ #
    #  缓存路径
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_ch_name(ch):
        """归一化通道名到 LaBraM standard_1020 形式.
        处理 TUH 等原始命名: 'EEG FP1-REF'/'EEG FP1-LE'->'FP1', 去前缀/参考后缀, 大写.
        """
        c = str(ch).strip().upper()
        if c.startswith('EEG '):
            c = c[4:]
        # 去参考后缀 -REF/-LE/-RE (TUH 双极/单极命名)
        for suf in ('-REF', '-LE', '-RE'):
            if c.endswith(suf):
                c = c[:-len(suf)]
                break
        return c.strip()

    @staticmethod
    def _config_dataset_id(config):
        """数据集唯一名 (隔离消融/采样/统计/缓存的粒度单位). name 必填."""
        name = config.get('name')
        if not name:
            raise ValueError(
                f"数据集 config 缺少必填 'name': type={config.get('type')}, data_dir={config.get('data_dir')}"
            )
        return name

    def _reinject_weights(self, configs):
        """按 dataset_id 重新注入 config 的 weight (缓存权重无关, scaling law 改权重不重建索引)."""
        wmap = {}
        for c in configs:
            wmap[self._config_dataset_id(c)] = float(c.get('weight', 1.0))
        for e in self.index:
            did = e.get('dataset_id')
            if did in wmap:
                e['weight'] = wmap[did]

    @staticmethod
    def _get_cache_path(configs, mode, cache_dir):
        import hashlib
        # 只用 type+data_dir+name (剔除 weight/max_samples) → scaling law 改权重时缓存复用
        key_parts = [
            {
                'type': c.get('type', '').lower(),
                'data_dir': c.get('data_dir', ''),
                'name': c.get('name', ''),
                'index_path': c.get('index_path', ''),
                'split_manifest': c.get('split_manifest', ''),
            }
            for c in configs
        ]
        config_str = json.dumps(key_parts, sort_keys=True) + f"_mode={mode}"
        h = hashlib.md5(config_str.encode()).hexdigest()[:12]
        if cache_dir is None:
            first_dir = configs[0].get('data_dir', '/tmp') if configs else '/tmp'
            cache_dir = os.path.join(os.path.dirname(first_dir), '.unified_cache_v2')
        return os.path.join(cache_dir, f"unified_v2_{mode}_{h}.pkl")

    # ------------------------------------------------------------------ #
    #  构建索引
    # ------------------------------------------------------------------ #

    def _build_index(self, configs: List[Dict]) -> List[Dict]:
        all_entries = []
        # 校验 name: 必填且唯一 (数据集级隔离消融/采样/统计的粒度单位)
        _seen = {}
        for _c in configs:
            _nm = _c.get('name')
            if not _nm:
                raise ValueError(f"数据集 config 缺少必填 'name' 字段: type={_c.get('type')}, data_dir={_c.get('data_dir')}")
            if _nm in _seen:
                raise ValueError(f"数据集 name 重复: '{_nm}' — 每个数据集 name 必须唯一")
            _seen[_nm] = True
        for config in configs:
            ds_type = config.get('type', '').lower()
            data_dir = config.get('data_dir', '')
            weight = config.get('weight', 1.0)

            if ds_type not in self.DATASET_RULES:
                if is_main_process():
                    logger.warning(f"Unknown dataset type: {ds_type}, skipping")
                continue

            rule = self.DATASET_RULES[ds_type]

            # Skip 非人类头皮 EEG (动物/超高密度阵列/非头皮, 见 excluded_datasets.py)
            try:
                from eeg_data.excluded_datasets import is_excluded
                _ds_id = config.get('name') or os.path.basename(str(data_dir).rstrip('/'))
                if is_excluded(_ds_id):
                    if is_main_process():
                        logger.info(f"  Excluding {_ds_id}: 非人类头皮 EEG (动物/阵列/非头皮)")
                    continue
            except ImportError:
                pass

            # Skip datasets with empty or missing data_dir
            if not data_dir or not os.path.exists(data_dir):
                if is_main_process():
                    logger.info(f"  Skipping {ds_type}: data_dir is empty or does not exist ('{data_dir}')")
                continue

            if ds_type == 'downstream' and config.get('index_path') and config.get('split_manifest'):
                entries = self._scan_indexed_downstream_dataset(data_dir, rule, config)
            elif ds_type in ('tuev', 'tuab', 'hmc',
                              'tusz', 'tusl', 'workload',
                              'faced', 'shumi', 'siena',
                              'bcic2020_3', 'bcicv_2a', 'downstream', 'seedimg'):
                entries = self._scan_pkl_dataset(ds_type, data_dir, rule, config)
            elif ds_type == 'eegimage':
                entries = self._scan_eegimage_dataset(data_dir, rule, config)
            elif ds_type in ('things', 'things_v1'):
                entries = self._scan_things_dataset(data_dir, rule, config, ds_type=ds_type)
            elif ds_type == 'alljoined':
                entries = self._scan_alljoined_dataset(data_dir, rule, config)
            elif ds_type == 'eegpt_pkl':
                entries = self._scan_eegpt_pkl_dataset(data_dir, rule, config)
            else:
                entries = []

            # Per-config max_samples subsampling (deterministic stride sampling)
            max_samples = config.get('max_samples', None)
            if max_samples is not None and len(entries) > max_samples:
                step = len(entries) / max_samples
                entries = [entries[int(i * step)] for i in range(max_samples)]
                if is_main_process():
                    logger.info(f"  Subsampled {ds_type}: -> {len(entries)} (max_samples={max_samples})")

            ds_name = self._config_dataset_id(config)
            for e in entries:
                e['weight'] = weight
                e['dataset_id'] = ds_name
            all_entries.extend(entries)

            if is_main_process():
                logger.info(f"  Scanned {ds_type}: {len(entries)} samples from {data_dir}")

        return all_entries

    def _scan_indexed_downstream_dataset(self, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        with open(config['index_path'], 'r', encoding='utf-8') as handle:
            index_payload = json.load(handle)
        with open(config['split_manifest'], 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        source_entries = index_payload.get('entries')
        if not isinstance(source_entries, list) or not source_entries:
            raise ValueError(f"invalid indexed downstream sample index: {config['index_path']}")
        split_name = 'val' if self.mode == 'eval' else self.mode
        indices = manifest.get('splits', {}).get(split_name)
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"indexed downstream split {split_name!r} is missing or empty")
        entries = []
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(source_entries):
                raise ValueError(f"indexed downstream split contains invalid index {index!r}")
            source = source_entries[index]
            file_path = str(source.get('path', ''))
            if not os.path.isabs(file_path):
                file_path = os.path.join(data_dir, file_path)
            ch_names = [str(name) for name in source.get('ch_names', [])]
            time_samples = int(source.get('time_samples', 0) or 0)
            shape_group = None
            if ch_names and time_samples >= int(rule.get('patch_size', 200)):
                shape_group = (
                    len(ch_names),
                    max(1, time_samples // int(rule.get('patch_size', 200))),
                    tuple(ch_names),
                )
            item_index = source.get('item_index')
            if item_index is not None and (not isinstance(item_index, int) or item_index < 0):
                raise ValueError(f"invalid indexed downstream item_index {item_index!r}")
            entries.append({
                'dataset_type': 'downstream',
                'file_path': file_path,
                'subject': str(source.get('subject_id', '')),
                'sample_id': str(source.get('sample_id', index)),
                'shape_group': shape_group,
                'item_index': item_index,
                'label_override': int(source['label']),
                'indexed_ch_names': ch_names,
                'source_dataset_type': config.get('source_dataset_type'),
            })
        return entries

    def _scan_pkl_dataset(self, ds_type: str, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        entries = []
        dir_pattern = rule['dir_pattern']
        mode_dir_name = dir_pattern.get(self.mode, self.mode)
        mode_dir = os.path.join(data_dir, mode_dir_name)

        # eval(验证)模式回退: 有些数据集验证集目录名为 'val' 而非 dir_pattern 指定的 'eval'。
        # 若指定目录不存在或无 .pkl, 尝试同级 'val/' 目录, 避免回退到 test 导致 val==test。
        if self.mode == 'eval':
            def _has_pkl(d):
                return os.path.isdir(d) and any(f.endswith('.pkl') for f in os.listdir(d))
            if not _has_pkl(mode_dir):
                alt_dir = os.path.join(data_dir, 'val')
                if _has_pkl(alt_dir):
                    if is_main_process():
                        logger.info(f"  eval split '{mode_dir_name}/' missing/empty; using 'val/' instead")
                    mode_dir = alt_dir

        if not os.path.isdir(mode_dir):
            if is_main_process():
                logger.warning(f"  Directory not found: {mode_dir}")
            return entries

        shape_group = rule['eeg_shape_group']

        for filename in _robust_listdir(mode_dir):
            if not (filename.endswith('.pkl') or filename.endswith('.pt')):
                continue
            file_path = os.path.join(mode_dir, filename)
            basename = filename.rsplit('.', 1)[0]
            if ds_type == 'tuev':
                parts = basename.split('_')
                subject = parts[0] if parts else basename
            elif ds_type == 'tuab':
                parts = basename.split('_')
                subject = parts[0] if parts else basename
            elif ds_type == 'hmc':
                parts = basename.split('-')
                subject = parts[0] if parts else basename
            elif ds_type in ('tusz', 'tusl'):
                # Same format as TUAB/TUEV (TUH-style naming)
                parts = basename.split('_')
                subject = parts[0] if parts else basename
            elif ds_type == 'workload':
                # Subject00_1-0 → subject = Subject00
                parts = basename.split('_')
                subject = parts[0] if parts else basename
            elif ds_type in ('faced', 'shumi', 'siena', 'bcic2020_3', 'bcicv_2a'):
                # Subject__trial_XXXX naming convention from our preprocessing
                parts = basename.split('__')
                subject = parts[0] if parts else basename
            else:
                subject = basename

            # For datasets without a predefined shape_group (e.g., BCIC variants),
            # peek at the first file to derive (n_ch, n_patches)
            entry_shape_group = shape_group
            if entry_shape_group is None and ds_type in ('bcic2020_3', 'bcicv_2a', 'downstream'):
                try:
                    with open(file_path, 'rb') as f:
                        sample = pickle.load(f)
                    if ds_type == 'downstream' and sample.get('subject') not in (None, ''):
                        subject = str(sample['subject'])
                    X = sample.get('X')
                    ch = sample.get('ch_names', [])
                    if X is not None and X.ndim == 2:
                        C, T = X.shape
                        n_p = max(1, T // rule.get('patch_size', 200))
                        entry_shape_group = (C, n_p, tuple(ch)) if ch else (C, n_p)
                except Exception:
                    pass

            entries.append({
                'dataset_type': ds_type,
                'file_path': file_path,
                'subject': subject,
                'sample_id': basename,
                'shape_group': entry_shape_group,
            })

        return entries

    def _scan_eegimage_dataset(self, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        entries = []
        shape_group = rule['eeg_shape_group']
        skip_game = config.get('if_skip_game', False)
        target_datasets = config.get('datasets', None)

        for ds_name in sorted(os.listdir(data_dir)):
            ds_path = os.path.join(data_dir, ds_name)
            if not os.path.isdir(ds_path):
                continue
            if ds_name.startswith('.') or ds_name.endswith('.json'):
                continue
            if skip_game and ds_name == 'Game':
                continue
            if target_datasets and ds_name not in target_datasets:
                continue

            mode_dir = os.path.join(ds_path, self.mode)
            if not os.path.isdir(mode_dir):
                continue

            for filename in _robust_listdir(mode_dir):
                if not filename.endswith('.pt'):
                    continue
                file_path = os.path.join(mode_dir, filename)
                basename = filename.rsplit('.', 1)[0]
                entries.append({
                    'dataset_type': 'eegimage',
                    'file_path': file_path,
                    'subject': '',
                    'sample_id': f"{ds_name}/{basename}",
                    'sub_dataset': ds_name,
                    'shape_group': shape_group,
                })

        return entries

    def _scan_things_dataset(self, data_dir: str, rule: Dict, config: Dict,
                             ds_type: str = 'things') -> List[Dict]:
        """扫描 ThingsEEG per-sample .pt 格式 — 优先 metadata.json.

        同时用于 things (v2, 17ch) 和 things_v1 (63ch)，通过 ds_type 区分。
        """
        entries = []
        shape_group = rule['eeg_shape_group']
        subjects_filter = config.get('subjects', None)

        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            if isinstance(all_metadata, dict) and 'samples' in all_metadata:
                all_metadata = all_metadata['samples']

            for m in all_metadata:
                if m.get('mode', '') != self.mode:
                    continue
                if subjects_filter and m.get('subject', '') not in subjects_filter:
                    continue

                subject = m.get('subject', '')
                sample_idx = m.get('sample_idx', 0)
                file_path = os.path.join(data_dir, subject, self.mode, f"sample_{sample_idx:06d}.pt")
                entries.append({
                    'dataset_type': ds_type,
                    'file_path': file_path,
                    'subject': subject,
                    'sample_id': f"{subject}/sample_{sample_idx:06d}",
                    'text': m.get('text', ''),
                    'label': m.get('label', 0),
                    'shape_group': shape_group,
                })
        else:
            for subject in sorted(os.listdir(data_dir)):
                subject_dir = os.path.join(data_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue
                if subjects_filter and subject not in subjects_filter:
                    continue
                mode_dir = os.path.join(subject_dir, self.mode)
                if not os.path.isdir(mode_dir):
                    continue
                for filename in _robust_listdir(mode_dir):
                    if not filename.endswith('.pt'):
                        continue
                    file_path = os.path.join(mode_dir, filename)
                    basename = filename.rsplit('.', 1)[0]
                    entries.append({
                        'dataset_type': ds_type,
                        'file_path': file_path,
                        'subject': subject,
                        'sample_id': f"{subject}/{basename}",
                        'text': '',
                        'label': 0,
                        'shape_group': shape_group,
                    })

        return entries

    def _scan_alljoined_dataset(self, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        entries = []
        shape_group = rule['eeg_shape_group']
        subjects_filter = config.get('subjects', None)

        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            for m in all_metadata:
                if m.get('mode', '') != self.mode:
                    continue
                if subjects_filter and m.get('subject', '') not in subjects_filter:
                    continue
                subject = m.get('subject', '')
                sample_idx = m.get('sample_idx', 0)
                if 'file_path' in m:
                    file_path = m['file_path']
                else:
                    file_path = os.path.join(data_dir, self.mode, f"sample_{sample_idx:06d}.pt")
                entries.append({
                    'dataset_type': 'alljoined',
                    'file_path': file_path,
                    'subject': subject,
                    'sample_id': f"alljoined/{subject}/sample_{sample_idx:06d}",
                    'category_name': m.get('category_name', ''),
                    'shape_group': shape_group,
                })
        else:
            mode_dir = os.path.join(data_dir, self.mode)
            if os.path.isdir(mode_dir):
                self._scan_alljoined_dir(mode_dir, entries, shape_group, subjects_filter)
            else:
                for subject in sorted(os.listdir(data_dir)):
                    subject_dir = os.path.join(data_dir, subject)
                    if not os.path.isdir(subject_dir) or not subject.startswith('sub'):
                        continue
                    if subjects_filter and subject not in subjects_filter:
                        continue
                    sub_mode_dir = os.path.join(subject_dir, self.mode)
                    if os.path.isdir(sub_mode_dir):
                        self._scan_alljoined_dir(sub_mode_dir, entries, shape_group, None, subject)

        return entries

    def _scan_alljoined_dir(self, directory: str, entries: list, shape_group: tuple,
                            subjects_filter=None, subject_hint: str = ''):
        for item_name in sorted(os.listdir(directory)):
            item_path = os.path.join(directory, item_name)
            if os.path.isdir(item_path):
                self._scan_alljoined_dir(item_path, entries, shape_group, subjects_filter, subject_hint)
            elif item_name.endswith('.pt'):
                basename = item_name.rsplit('.', 1)[0]
                entries.append({
                    'dataset_type': 'alljoined',
                    'file_path': item_path,
                    'subject': subject_hint,
                    'sample_id': f"alljoined/{basename}",
                    'category_name': '',
                    'shape_group': shape_group,
                })

    def _scan_eegpt_pkl_dataset(self, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        _scan_ds_id = self._config_dataset_id(config)
        """扫描 EEGPT TUH 预训练 pkl 数据集.

        每个 pkl 文件: {X: np.array(C, T), ch_names: [str]}
        通道需过滤：仅保留在 LaBraM standard_1020 中的通道。
        shape_group 设为 (n_ch_filtered, n_patches, tuple(ch_names_filtered))，
        确保 GroupedBatchSampler 按通道名称分组。

        支持两种模式：
          1. manifest.json 快速路径 — 预构建索引避免逐文件扫描
          2. 逐文件扫描慢速路径
        """
        import sys
        labram_utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        if labram_utils_path not in sys.path:
            sys.path.insert(0, labram_utils_path)
        from labram_utils import standard_1020 as labram_1020
        labram_set = set(labram_1020)

        entries = []
        manifest_path = os.path.join(data_dir, 'manifest.json')

        if os.path.exists(manifest_path):
            # ---- 快速路径：从 manifest 加载 ----
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            from eeg_data.ch_name_mappings import resolve_channels
            for item in manifest:
                raw_ch = item['ch_names']
                n_xch = item.get('n_channels', len(raw_ch))
                keep_idx_m, ch = resolve_channels(raw_ch, labram_set, n_xch, dataset_id=_scan_ds_id)
                if len(ch) == 0:
                    continue
                n_patches = item['n_patches']
                # 截断 n_patches 与 eeg_truncate_patches 对齐
                max_patches = rule.get('eeg_truncate_patches', None)
                if max_patches is not None:
                    n_patches = min(n_patches, max_patches)
                file_path = item['file_path']
                if not os.path.isabs(file_path):
                    file_path = os.path.join(data_dir, file_path)
                entries.append({
                    'dataset_type': 'eegpt_pkl',
                    'file_path': file_path,
                    'subject': item.get('subject', ''),
                    'sample_id': os.path.basename(file_path),
                    'shape_group': (len(ch), n_patches, tuple(ch)),
                    'filtered_ch_names': ch,
                })
        else:
            # ---- 慢速路径：逐文件扫描 ----
            if is_main_process():
                logger.warning(f"  No manifest.json found in {data_dir}, scanning all pkl files "
                               f"(slow — consider running tools/build_eegpt_manifest.py)")
            for fname in sorted(os.listdir(data_dir)):
                if not fname.endswith('.pkl'):
                    continue
                fpath = os.path.join(data_dir, fname)
                try:
                    with open(fpath, 'rb') as f:
                        sample = pickle.load(f)
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"  Failed to load {fpath}: {e}")
                    continue
                raw_ch = sample.get('ch_names', [])
                n_xch = sample['X'].shape[0]
                from eeg_data.ch_name_mappings import resolve_channels
                keep_idx, ch = resolve_channels(raw_ch, labram_set, n_xch, dataset_id=_scan_ds_id)
                if len(ch) == 0:
                    continue
                T = sample['X'].shape[1]
                n_patches = T // 200
                if n_patches == 0:
                    continue
                entries.append({
                    'dataset_type': 'eegpt_pkl',
                    'file_path': fpath,
                    'subject': '',
                    'sample_id': fname,
                    'shape_group': (len(ch), n_patches, tuple(ch)),
                    'filtered_ch_names': ch,
                })

        return entries

    # ------------------------------------------------------------------ #
    #  均衡采样
    # ------------------------------------------------------------------ #

    def _apply_balanced_sampling(self, index: List[Dict], configs: List[Dict]) -> List[Dict]:
        # 按数据集名 (dataset_id) 分组, 而非 dataset_type — 让每个数据集独立平衡
        ds_groups: Dict[str, List[Dict]] = {}
        for entry in index:
            ds_id = entry.get('dataset_id', entry['dataset_type'])
            if ds_id not in ds_groups:
                ds_groups[ds_id] = []
            ds_groups[ds_id].append(entry)

        weight_map = {}
        for config in configs:
            weight_map[self._config_dataset_id(config)] = config.get('weight', 1.0)

        total_weight = sum(weight_map.get(k, 1.0) for k in ds_groups)
        if total_weight <= 0:
            return index

        max_samples_per_weight = 0
        for ds_id, entries in ds_groups.items():
            w = weight_map.get(ds_id, 1.0)
            if w > 0:
                max_samples_per_weight = max(max_samples_per_weight, len(entries) / w)

        balanced_index = []
        for ds_id, entries in ds_groups.items():
            w = weight_map.get(ds_id, 1.0)
            target_n = int(max_samples_per_weight * w)
            if self.max_samples_per_dataset is not None:
                target_n = min(target_n, self.max_samples_per_dataset)

            if target_n <= 0:
                if is_main_process():
                    logger.info(f"  Balanced: {ds_id} {len(entries)} → 0 (weight={w}, excluded)")
                continue
            if target_n >= len(entries):
                repeated = entries * (target_n // len(entries))
                remainder = target_n - len(repeated)
                repeated.extend(entries[:remainder])
                balanced_index.extend(repeated)
            else:
                step = len(entries) / target_n
                selected = [entries[int(i * step)] for i in range(target_n)]
                balanced_index.extend(selected)

            if is_main_process():
                logger.info(f"  Balanced: {ds_id} {len(entries)} → {target_n} (weight={w})")

        random.shuffle(balanced_index)
        return balanced_index

    def _apply_max_cap(self, index: List[Dict], max_samples: int) -> List[Dict]:
        # 按数据集名 (dataset_id) 分组 — 每个数据集独立 cap (而非所有同 type 合并)
        ds_groups: Dict[str, List[Dict]] = {}
        for entry in index:
            ds_id = entry.get('dataset_id', entry['dataset_type'])
            if ds_id not in ds_groups:
                ds_groups[ds_id] = []
            ds_groups[ds_id].append(entry)

        capped_index = []
        for ds_id, entries in ds_groups.items():
            if len(entries) > max_samples:
                step = len(entries) / max_samples
                selected = [entries[int(i * step)] for i in range(max_samples)]
                capped_index.extend(selected)
                if is_main_process():
                    logger.info(f"  Capped: {ds_id} {len(entries)} → {max_samples}")
            else:
                capped_index.extend(entries)

        return capped_index

    # ------------------------------------------------------------------ #
    #  Shape 分组
    # ------------------------------------------------------------------ #

    def _effective_truncate_patches(self, entry: Dict, rule: Dict) -> Optional[int]:
        max_patches = entry.get('eeg_truncate_patches')
        if max_patches is None and entry.get('dataset_type') == 'downstream':
            source_type = str(entry.get('source_dataset_type') or entry.get('dataset_id', '')).lower()
            source_rule = self.DATASET_RULES.get(source_type, {})
            max_patches = source_rule.get('eeg_truncate_patches')
        return int(max_patches) if max_patches is not None else None

    def _build_shape_groups(self):
        self.shape_groups = {}
        for i, entry in enumerate(self.index):
            ds_type = entry.get('dataset_type', '')
            rule = self.DATASET_RULES.get(ds_type, {})
            sg = rule.get('eeg_shape_group')
            if sg is None:
                sg = entry.get('shape_group', (0, 0))
            max_patches = self._effective_truncate_patches(entry, rule)
            if (
                max_patches is not None
                and isinstance(sg, (tuple, list))
                and len(sg) >= 2
            ):
                sg = (sg[0], min(int(sg[1]), max_patches), *sg[2:])
            entry['shape_group'] = sg
            if sg not in self.shape_groups:
                self.shape_groups[sg] = []
            self.shape_groups[sg].append(i)

    def get_shape_groups(self) -> Dict[Tuple, List[int]]:
        return self.shape_groups

    # ------------------------------------------------------------------ #
    #  __len__ / __getitem__
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.index)

    def _cache_local(self, file_path):
        """本地样本缓存: 启用且命中→返回本地路径; 未命中→从 blob 原子拷到本地再返回;
        未启用或任何异常→回退返回原路径(直读 blob). 写本地磁盘 os.replace 原子生效,
        磁盘满/权限等失败均安全降级为直读 blob。"""
        cd = self.data_cache_dir
        if not cd:
            return file_path
        tmp = None
        try:
            local = os.path.join(cd, file_path.lstrip('/'))   # 保留完整路径结构避免碰撞
            if os.path.exists(local) and os.path.getsize(local) > 0:
                return local                                  # 命中
            os.makedirs(os.path.dirname(local), exist_ok=True)
            tmp = local + ".tmp.%d" % os.getpid()
            shutil.copyfile(file_path, tmp)
            os.replace(tmp, local)                            # 本地盘原子替换
            return local
        except Exception:
            if tmp:
                try: os.remove(tmp)
                except OSError: pass
            return file_path                                  # 降级: 直读 blob

    def _read_one(self, entry):
        """单次读取一个样本文件 (无重试)。返回 (data, ds_type, rule)。"""
        ds_type = entry['dataset_type']
        rule = self.DATASET_RULES[ds_type]
        file_path = self._cache_local(entry['file_path'])   # 命中读本地, 否则拷blob->本地再读(失败回退blob)
        if rule['file_format'] == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:  # 'pt'
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        item_index = entry.get('item_index')
        if item_index is not None:
            data = dict(data)
            selected_signal = False
            for key in ('X', 'eeg', 'signal'):
                if key not in data:
                    continue
                value = data[key]
                if not hasattr(value, 'ndim') or value.ndim < 3 or item_index >= len(value):
                    raise ValueError(f"invalid indexed shard field {key!r} at item {item_index}")
                data[key] = value[item_index]
                selected_signal = True
            if not selected_signal:
                raise ValueError(f"indexed shard has no batched EEG field: {sorted(data)}")
            for key in ('y', 'label'):
                value = data.get(key)
                if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                    if item_index >= len(value):
                        raise ValueError(f"invalid indexed shard label at item {item_index}")
                    data[key] = value[item_index]
        if 'label_override' in entry:
            data = dict(data)
            data['y'] = entry['label_override']
            if entry.get('subject'):
                data['subject'] = entry['subject']
            if entry.get('indexed_ch_names'):
                data['ch_names'] = entry['indexed_ch_names']
        return data, ds_type, rule

    def _robust_load(self, idx):
        """读取样本, 带 blobfuse 高并发容错:
        1) 同文件指数退避重试 (应对 blobfuse 瞬时 ENOTCONN/超时);
        2) 仍失败则换同 shape_group 其它样本兜底 (避免单文件拖垮分布式 job);
        3) 全失败抛出, 交由 job 级 max_attempts 重启 (新节点重新挂载 blobfuse)。
        """
        import time as _time, random as _random
        entry = self.index[idx]
        last_exc = None
        for attempt in range(8):
            try:
                data, ds_type, rule = self._read_one(entry)
                return data, entry, ds_type, rule
            except (OSError, IOError, EOFError, pickle.UnpicklingError, RuntimeError) as e:
                last_exc = e
                _time.sleep(min(0.5 * (2 ** attempt), 10.0))
        sg = entry.get('shape_group')
        candidates = self.shape_groups.get(sg, [])
        if candidates:
            for alt in _random.sample(candidates, min(len(candidates), 32)):
                if alt == idx:
                    continue
                alt_entry = self.index[alt]
                try:
                    data, ds_type, rule = self._read_one(alt_entry)
                    return data, alt_entry, ds_type, rule
                except (OSError, IOError, EOFError, pickle.UnpicklingError, RuntimeError):
                    continue
        raise last_exc

    def __getitem__(self, idx):
        data, entry, ds_type, rule = self._robust_load(idx)

        # ---- 提取 EEG (RC5: eeg_key缺失时回退到对齐格式'X') ----
        _ek = rule['eeg_key']
        eeg = data[_ek] if _ek in data else data.get('X')
        if eeg is None:
            raise KeyError(f"eeg_key {_ek!r} 及回退键'X'均不在样本: keys={list(data.keys())}")
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float()
        elif isinstance(eeg, torch.Tensor):
            eeg = eeg.float()
        else:
            eeg = torch.tensor(eeg, dtype=torch.float32)

        # ---- EEGPT 通道过滤 (EGI映射/归一化 + 对齐 ch/X 数量) ----
        if ds_type == 'eegpt_pkl' and 'filtered_ch_names' in entry:
            raw_ch = data.get('ch_names', [])
            import sys
            labram_utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
            if labram_utils_path not in sys.path:
                sys.path.insert(0, labram_utils_path)
            from labram_utils import standard_1020 as labram_1020
            from eeg_data.ch_name_mappings import resolve_channels
            labram_set = set(labram_1020)
            keep_idx, _ = resolve_channels(raw_ch, labram_set, eeg.shape[0], dataset_id=entry.get('dataset_id'))
            if keep_idx:
                eeg = eeg[keep_idx]

        if rule['eeg_reshape'] == 'auto':
            if eeg.dim() == 2:
                ps = rule['patch_size']
                _nf = max(1, eeg.shape[1] // ps)
                eeg = eeg[:, :_nf*ps].reshape(eeg.shape[0], _nf, ps)

        if ds_type in ('things', 'things_v1', 'alljoined') and eeg.dim() == 2:
            eeg = eeg.unsqueeze(1)

        max_patches = self._effective_truncate_patches(entry, rule)
        if max_patches is not None:
            eeg = eeg[:, :max_patches, :]

        # ---- 提取图像 ----
        img_key = rule['image_key']
        img = data.get(img_key, None) if img_key is not None else None

        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            elif isinstance(img, torch.Tensor):
                img = img.float()
            else:
                img = torch.tensor(img, dtype=torch.float32)

            if img.dim() == 3 and img.shape[-1] in (1, 3, 4):
                img = img.permute(2, 0, 1)

            if img.shape[-2] != self.image_size or img.shape[-1] != self.image_size:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        else:
            img = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        # ---- 提取标签 ----
        label = self._extract_label(data, entry, rule, ds_type)

        # ---- 提取元数据 ----
        subject = entry.get('subject', '') or str(data.get('subject', ''))
        sample_id = entry.get('sample_id', '')
        if ds_type == 'eegpt_pkl' and 'filtered_ch_names' in entry:
            ch_names = entry['filtered_ch_names']
        else:
            ch_names = rule['ch_names']
            if ch_names is None:
                ch_names = data.get('ch_names', [])

        # ---- 获取标签名 + Prompt ----
        label_name, prompt = self._get_label_and_prompt(ds_type, label, data, entry, rule)
        label_names = self._get_label_names(ds_type, data, entry, rule)

        return {
            'eeg': eeg,
            'img': img,
            'label': label,
            'label_idx': label,
            'meta': {
                'dataset': ds_type if ds_type != 'eegimage' else entry.get('sub_dataset', ds_type),
                'subject': subject,
                'sample_id': sample_id,
                'true_label': label_name,
                'true_label_idx': label,
                'candicate_label_choice': label_names,
                'system_prompt': prompt,
                'user_prompt': prompt,
                'ch_names': ch_names,
            }
        }

    # ------------------------------------------------------------------ #
    #  标签提取
    # ------------------------------------------------------------------ #

    def _extract_label(self, data, entry, rule, ds_type):
        label_key = rule['label_key']
        if label_key is None:
            return 0  # dummy label for pretraining data
        if label_key == '__eegimage__':
            label = data.get('true_label_idx', -1)
            if isinstance(label, (list, np.ndarray)):
                label = int(label[0]) if len(label) > 0 else -1
            return int(label)
        if label_key == '__alljoined__':
            return 0
        raw_label = data[label_key] if label_key in data else data.get('y', 0)  # RC5b: label_key缺失回退到对齐'y'
        if rule.get('label_is_array', False):
            if isinstance(raw_label, (list, np.ndarray)):
                raw_label = raw_label[0] if len(raw_label) > 0 else 0
        label = int(raw_label) + rule.get('label_offset', 0)
        return label

    def _get_label_names(self, ds_type, data, entry, rule):
        if ds_type == 'eegimage':
            sub_ds = entry.get('sub_dataset', '')
            return self.EEGIMAGE_LABEL_DICT.get(sub_ds, [])
        if rule['label_names'] is not None:
            return list(rule['label_names'])
        return []

    def _get_label_and_prompt(self, ds_type, label, data, entry, rule):
        if rule['label_key'] is None:
            return '', ''  # pretraining data — no label/prompt

        cache_key = (ds_type, entry.get('sub_dataset', ''), label)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        label_names = self._get_label_names(ds_type, data, entry, rule)
        if label_names and 0 <= label < len(label_names):
            label_name = label_names[label]
        else:
            label_name = str(label)

        if ds_type == 'eegimage':
            sub_ds = entry.get('sub_dataset', '')
            prompt = self.EEGIMAGE_PROMPT_DICT.get(sub_ds, '')
        elif ds_type in ('things', 'things_v1'):
            label_name = entry.get('text', '') or data.get('text', '') or str(label)
            prompt = (
                "You should identify the primary object or concept depicted in the visual stimulus.\n"
                "The stimulus belongs to one of the object categories from a large-scale image dataset.\n"
                "Respond with exactly one word or short phrase describing the object."
            )
        elif ds_type == 'alljoined':
            label_name = entry.get('category_name', '') or data.get('category_name', '') or str(label)
            prompt = (
                "You should identify the primary object or concept depicted in the visual stimulus.\n"
                "The stimulus belongs to one of the object categories from a large-scale image dataset.\n"
                "Respond with exactly one word or short phrase describing the object."
            )
        elif rule['prompt'] is not None:
            prompt = rule['prompt']
        else:
            prompt = ''

        self._prompt_cache[cache_key] = (label_name, prompt)
        return label_name, prompt

    # ------------------------------------------------------------------ #
    #  数据集信息
    # ------------------------------------------------------------------ #

    def get_dataset_info(self) -> Dict[str, Any]:
        ds_counts = {}
        for entry in self.index:
            ds_type = entry['dataset_type']
            ds_counts[ds_type] = ds_counts.get(ds_type, 0) + 1
        return {
            'total_samples': len(self.index),
            'mode': self.mode,
            'balanced_sampling': self.balanced_sampling,
            'num_shape_groups': len(self.shape_groups),
            'shape_groups': {str(k): len(v) for k, v in self.shape_groups.items()},
            'datasets': ds_counts,
        }


# ====================================================================== #
#  GroupedBatchSampler
# ====================================================================== #

class GroupedBatchSampler:
    """
    按 EEG shape 分组的 BatchSampler — 确保同一 batch 内所有样本具有相同的 EEG 形状.

    用法:
        dataset = UnifiedEEGDataset_v2(configs, mode='train')
        sampler = GroupedBatchSampler(dataset, batch_size=32, shuffle=True)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=unified_collate_fn_v2)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.shape_groups = dataset.get_shape_groups()
        self._build_batches()

    def _build_batches(self):
        all_batches = []
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        for group_key, indices in self.shape_groups.items():
            indices = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[p] for p in perm]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[p] for p in perm]

        if self.num_replicas > 1:
            n_batches = len(all_batches)
            n_per_rank = (n_batches + self.num_replicas - 1) // self.num_replicas
            if len(all_batches) < n_per_rank * self.num_replicas:
                all_batches += all_batches[:n_per_rank * self.num_replicas - len(all_batches)]
            self.batches = all_batches[self.rank * n_per_rank:(self.rank + 1) * n_per_rank]
        else:
            self.batches = all_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._build_batches()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)




# ====================================================================== #
#  WeightedGroupedBatchSampler  (for scaling-law dataset weighting)
# ====================================================================== #

class WeightedGroupedBatchSampler:
    """按数据集采样权重 + EEG shape 分组的 BatchSampler.

    与 GroupedBatchSampler 的区别:
      - 每个 epoch 按权重概率重采样 samples_per_epoch 个样本 (而非遍历全部),
        高权重数据集被采样更多 (可重复 oversample), 权重 0 的数据集被排除。
      - 仍保证同一 batch 内所有样本 shape 相同 (兼容 unified_collate_fn_v2)。

    权重语义 (size_aware=False, 默认, scaling-law 推荐):
      每个数据集在一个 epoch 中的样本占比 ∝ 该数据集 weight, 与数据集原始
      大小解耦 (集内样本均匀采样)。即 weight 直接 = 数据集贡献占比。
    权重语义 (size_aware=True):
      per-sample 概率 ∝ weight, 大数据集因样本多自然占更大比例。

    Scaling-law 用法:
      - D-量轴:     调 samples_per_epoch (每 epoch 总样本预算)
      - D-多样性轴: 调每个 config 的 weight (0 = 排除该集, 均匀 = 全用)
      - 改权重无需重建 index (dataset.index 不变, 索引缓存复用)

    用法:
        ds = UnifiedEEGDataset_v2(configs, mode='train', balanced_sampling=False)
        sampler = WeightedGroupedBatchSampler(ds, batch_size=32,
                                              samples_per_epoch=500000)
        loader = DataLoader(ds, batch_sampler=sampler,
                            collate_fn=unified_collate_fn_v2)
        # 每个 epoch 调用 sampler.set_epoch(epoch) 以重采样
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        samples_per_epoch: Optional[int] = None,
        size_aware: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        from collections import Counter
        index = dataset.index
        n = len(index)

        def _did(e):
            return e.get('dataset_id', e.get('dataset_type', '?'))

        ds_weight: Dict[Any, float] = {}
        for e in index:
            ds_weight[_did(e)] = float(e.get('weight', 1.0))
        ds_count = Counter(_did(e) for e in index)

        if size_aware:
            raw = np.array([ds_weight[_did(e)] for e in index], dtype=np.float64)
        else:
            # 数据集级占比 ∝ weight, 集内均匀 (与数据集大小解耦)
            raw = np.array(
                [ds_weight[_did(e)] / max(ds_count[_did(e)], 1) for e in index],
                dtype=np.float64,
            )
        total = raw.sum()
        if total > 0:
            self.probs = raw / total
        else:
            self.probs = np.full(n, 1.0 / max(n, 1), dtype=np.float64)

        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else n

        self.sample_to_group: Dict[int, Any] = {}
        for gk, indices in dataset.get_shape_groups().items():
            for i in indices:
                self.sample_to_group[i] = gk

        if is_main_process():
            tot_w = sum(ds_weight.values()) or 1.0
            logger.info(
                f"WeightedGroupedBatchSampler: {len(ds_count)} datasets, "
                f"samples_per_epoch={self.samples_per_epoch}, size_aware={size_aware}"
            )
            for did, w in sorted(ds_weight.items(), key=lambda x: -x[1])[:30]:
                share = (w / tot_w * 100.0) if not size_aware else (w * ds_count[did] / max(sum(ds_weight[d] * ds_count[d] for d in ds_count), 1) * 100.0)
                logger.info(f"    {did}: weight={w:.3f} N={ds_count[did]} -> ~{share:.1f}%/epoch")

        self._build_batches()

    def _build_batches(self):
        n = len(self.probs)
        if n == 0:
            self.batches = []
            return
        rng = np.random.default_rng(self.seed + self.epoch)

        sampled = rng.choice(n, size=self.samples_per_epoch, replace=True, p=self.probs)

        from collections import defaultdict
        buckets: Dict[Any, List[int]] = defaultdict(list)
        for idx in sampled.tolist():
            buckets[self.sample_to_group.get(idx)].append(idx)

        all_batches = []
        for gk, indices in buckets.items():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            perm = rng.permutation(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in perm]

        if self.num_replicas > 1:
            nb = len(all_batches)
            n_per_rank = (nb + self.num_replicas - 1) // self.num_replicas
            if len(all_batches) < n_per_rank * self.num_replicas:
                all_batches += all_batches[:n_per_rank * self.num_replicas - len(all_batches)]
            self.batches = all_batches[self.rank * n_per_rank:(self.rank + 1) * n_per_rank]
        else:
            self.batches = all_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._build_batches()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# ====================================================================== #
#  unified_collate_fn_v2
# ====================================================================== #

def unified_collate_fn_v2(batch):
    """UnifiedEEGDataset_v2 的 collate 函数."""
    def to_float_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        return x

    eeg_list = [to_float_tensor(item['eeg']) for item in batch]

    eeg_shapes = [tuple(e.shape) for e in eeg_list]
    if len(set(eeg_shapes)) == 1:
        eeg_batch = torch.stack(eeg_list)
    else:
        ndim = eeg_list[0].dim()
        max_shape = [max(e.shape[d] for e in eeg_list) for d in range(ndim)]
        padded = []
        for e in eeg_list:
            if list(e.shape) == max_shape:
                padded.append(e)
            else:
                new_e = torch.zeros(*max_shape, dtype=e.dtype)
                slices = tuple(slice(0, s) for s in e.shape)
                new_e[slices] = e
                padded.append(new_e)
        eeg_batch = torch.stack(padded)

    img_batch = torch.stack([to_float_tensor(item['img']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    label_idxs = torch.tensor([item['label_idx'] for item in batch], dtype=torch.long)

    meta_keys = ['dataset', 'subject', 'sample_id', 'true_label', 'true_label_idx',
                 'candicate_label_choice', 'system_prompt', 'user_prompt', 'ch_names']
    meta = {}
    for key in meta_keys:
        meta[key] = [item['meta'].get(key, '') for item in batch]

    return {
        'eeg': eeg_batch,
        'img': img_batch,
        'label': labels,
        'label_idx': label_idxs,
        'meta': meta,
    }


# ====================================================================== #
#  ThingsEEGDataset_2
# ====================================================================== #

class ThingsEEGDataset_2(Dataset):
    """
    ThingsEEG Dataset v2 - 加载从 process_things.py 生成的 .pt 文件

    目录结构:
        data_dir/
        ├── metadata.json
        ├── sub-01/
        │   ├── train/
        │   │   ├── sample_000000.pt
        │   │   └── ...
        │   └── test/
        └── sub-02/
            └── ...

    每个 .pt 文件包含:
        - eeg: [channels, time_points] 已重采样的 EEG 数据
        - image: [H, W, C] 或 [C, H, W] 图像数据
        - label: int 标签索引
        - text: str 文本描述
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        num_candidate: int = 10,
        input_type: str = "img",
        subjects: List[str] = None,
        if_cross: bool = False,
        target_sub: str = None,
        sub_ratio: float = 1.0,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.num_candidate = num_candidate
        self.input_type = input_type
        self.subjects = subjects
        self.if_cross = if_cross
        self.target_sub = target_sub
        self.sub_ratio = sub_ratio

        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            if isinstance(all_metadata, dict) and 'samples' in all_metadata:
                all_metadata = all_metadata['samples']

            if if_cross and target_sub is not None:
                self.metadata = [m for m in all_metadata
                                if m['mode'] == mode and m['subject'] == target_sub]
                if mode == 'train' and 0 < sub_ratio < 1.0:
                    self.metadata = self._uniform_sample(self.metadata, sub_ratio)
            else:
                self.metadata = [m for m in all_metadata if m['mode'] == mode]
                if subjects is not None:
                    self.metadata = [m for m in self.metadata if m['subject'] in subjects]
        else:
            self.metadata = self._scan_directory()

        self.label_dict = {}
        self.idx_dict = {}
        for m in self.metadata:
            text = m['text']
            if text not in self.label_dict:
                self.label_dict[text] = len(self.label_dict)
                self.idx_dict[len(self.idx_dict)] = text

        self.len_label = len(self.label_dict)

        self.input_type_prompt_dict = {
            "eeg": "The following is the feature of the subject's EEG.",
            "img": "The following is the image that the subject is viewing and is reconstructed from the subject's EEG.",
            "mix": "The following is the image reconstructed from the subject's EEG and the feature of the subject's EEG.",
            "img_tokens": "The following image tokens are reconstructed from the subject's EEG.",
        }
        self.system_prompt = "You should identify the primary object shown in the image that the subject is viewing. The possible objects are strictly limited to:"
        self.base_prompt = "Respond with exactly one word, and only one word, chosen from the options above. "

        if is_main_process():
            print(f"ThingsEEGDataset_2: Loaded {len(self.metadata)} samples for {mode} mode")
            print(f"  - Unique labels: {self.len_label}")
            if subjects:
                print(f"  - Subjects: {subjects}")
            if if_cross and target_sub:
                print(f"  - Few-shot mode: target_sub={target_sub}, sub_ratio={sub_ratio}")

    def _uniform_sample(self, metadata: List[Dict], ratio: float) -> List[Dict]:
        if ratio >= 1.0:
            return metadata
        if ratio <= 0:
            return []

        label_to_samples = {}
        for m in metadata:
            label = m.get('text', 'unknown')
            if label not in label_to_samples:
                label_to_samples[label] = []
            label_to_samples[label].append(m)

        sampled_metadata = []
        for label, samples in label_to_samples.items():
            n_samples = len(samples)
            n_select = max(1, int(n_samples * ratio))
            step = n_samples / n_select
            indices = [int(i * step) for i in range(n_select)]
            sampled_metadata.extend([samples[i] for i in indices])

        if is_main_process():
            print(f"  - Stratified sampling: {len(metadata)} -> {len(sampled_metadata)} samples (ratio={ratio})")

        return sampled_metadata

    def _scan_directory(self) -> List[Dict]:
        metadata = []
        for subject in os.listdir(self.data_dir):
            subject_dir = os.path.join(self.data_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            if self.subjects is not None and subject not in self.subjects:
                continue
            mode_dir = os.path.join(subject_dir, self.mode)
            if not os.path.exists(mode_dir):
                continue
            for filename in _robust_listdir(mode_dir):
                if filename.endswith('.pt'):
                    file_path = os.path.join(mode_dir, filename)
                    try:
                        data = torch.load(file_path, map_location='cpu', weights_only=False)
                        metadata.append({
                            'subject': subject,
                            'mode': self.mode,
                            'file_path': file_path,
                            'label': data.get('label', 0),
                            'text': data.get('text', ''),
                        })
                    except Exception as e:
                        print(f"Warning: Failed to load {file_path}: {e}")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]

        if 'file_path' in meta:
            file_path = meta['file_path']
        else:
            file_path = os.path.join(
                self.data_dir,
                meta['subject'],
                meta['mode'],
                f"sample_{meta['sample_idx']:06d}.pt"
            )

        data = torch.load(file_path, map_location='cpu', weights_only=False)

        eeg = data['eeg']
        image = data['image']
        label = data.get('label', meta.get('label', 0))
        text = data.get('text', meta.get('text', ''))

        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(1)

        if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        if text in self.label_dict:
            label_idx = self.label_dict[text]
            candidate_indices = [(i + label_idx) % self.len_label for i in range(self.num_candidate)]
        else:
            candidate_indices = list(range(min(self.num_candidate, self.len_label)))

        candidate_index = np.random.choice(candidate_indices, size=min(self.num_candidate, len(candidate_indices)), replace=False)
        image_candidates = [self.idx_dict[i] for i in candidate_index]
        image_candidates_str = "\n".join(f"• {c}" for c in image_candidates)
        prompt = f"{self.system_prompt}\n{image_candidates_str}\n{self.base_prompt}{self.input_type_prompt_dict[self.input_type]}"

        return {
            'eeg': eeg.float(),
            'img': image.float(),
            'meta': {
                'subject': meta['subject'],
                'start_second': idx,
                'label': label,
                'text': text,
                'index_img': meta.get('index_img', 0),
                'index_time': meta.get('index_time', 0),
                'image_candidates': image_candidates,
                'prompt': prompt,
            }
        }


# ====================================================================== #
#  custom_collate_fn_things_2
# ====================================================================== #

def custom_collate_fn_things_2(batch):
    """Custom collate function for ThingsEEGDataset_2."""
    collated_batch = {
        'eeg': torch.stack([item['eeg'] for item in batch]),
        'img': torch.stack([item['img'] for item in batch]),
        'meta': {
            'subject': [item['meta']['subject'] for item in batch],
            'start_second': [item['meta']['start_second'] for item in batch],
            'label': [item['meta']['label'] for item in batch],
            'text': [item['meta']['text'] for item in batch],
            'index_img': [item['meta']['index_img'] for item in batch],
            'index_time': [item['meta']['index_time'] for item in batch],
            'image_candidates': [item['meta']['image_candidates'] for item in batch],
            'prompt': [item['meta']['prompt'] for item in batch]
        }
    }
    return collated_batch
