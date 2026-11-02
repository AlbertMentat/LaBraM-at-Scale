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
            'dir_pattern': {'train': 'processed_train', 'eval': 'processed_eval', 'test': 'processed_test'},
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
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
            'eeg_shape_group': (23, 10),
        },
        'hmc': {
            'file_format': 'pkl',
            'eeg_key': 'X',
            'image_key': 'generated_image',
            'label_key': 'y',
            'label_is_array': False,
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
            'dir_pattern': {'train': 'train', 'eval': 'eval', 'test': 'test'},
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
            if cache_path and is_main_process():
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
            if dist.is_initialized():
                dist.barrier()

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
                ds_type = entry['dataset_type']
                ds_counts[ds_type] = ds_counts.get(ds_type, 0) + 1
            logger.info(f"UnifiedEEGDataset_v2: mode={mode}, total={len(self.index)}, "
                        f"shape_groups={len(self.shape_groups)}")
            for ds_type, count in sorted(ds_counts.items()):
                rule = self.DATASET_RULES.get(ds_type, {})
                sg = rule.get('eeg_shape_group', '?')
                logger.info(f"  {ds_type}: {count} samples, shape_group={sg}")

    # ------------------------------------------------------------------ #
    #  缓存路径
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_cache_path(configs, mode, cache_dir):
        import hashlib
        config_str = json.dumps(configs, sort_keys=True) + f"_mode={mode}"
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
        for config in configs:
            ds_type = config.get('type', '').lower()
            data_dir = config.get('data_dir', '')
            weight = config.get('weight', 1.0)

            if ds_type not in self.DATASET_RULES:
                if is_main_process():
                    logger.warning(f"Unknown dataset type: {ds_type}, skipping")
                continue

            rule = self.DATASET_RULES[ds_type]

            # Skip datasets with empty or missing data_dir
            if not data_dir or not os.path.exists(data_dir):
                if is_main_process():
                    logger.info(f"  Skipping {ds_type}: data_dir is empty or does not exist ('{data_dir}')")
                continue

            if ds_type in ('tuev', 'tuab', 'hmc'):
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

            for e in entries:
                e['weight'] = weight
            all_entries.extend(entries)

            if is_main_process():
                logger.info(f"  Scanned {ds_type}: {len(entries)} samples from {data_dir}")

        return all_entries

    def _scan_pkl_dataset(self, ds_type: str, data_dir: str, rule: Dict, config: Dict) -> List[Dict]:
        entries = []
        dir_pattern = rule['dir_pattern']
        mode_dir_name = dir_pattern.get(self.mode, self.mode)
        mode_dir = os.path.join(data_dir, mode_dir_name)

        if not os.path.isdir(mode_dir):
            if is_main_process():
                logger.warning(f"  Directory not found: {mode_dir}")
            return entries

        shape_group = rule['eeg_shape_group']

        for filename in sorted(os.listdir(mode_dir)):
            if not filename.endswith('.pkl'):
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
            else:
                subject = basename

            entries.append({
                'dataset_type': ds_type,
                'file_path': file_path,
                'subject': subject,
                'sample_id': basename,
                'shape_group': shape_group,
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

            for filename in sorted(os.listdir(mode_dir)):
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
                for filename in sorted(os.listdir(mode_dir)):
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
            for item in manifest:
                raw_ch = item['ch_names']
                ch = [c for c in raw_ch if c in labram_set]
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
                keep_idx = [i for i, c in enumerate(raw_ch) if c in labram_set]
                ch = [raw_ch[i] for i in keep_idx]
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
        ds_groups: Dict[str, List[Dict]] = {}
        for entry in index:
            ds_type = entry['dataset_type']
            if ds_type not in ds_groups:
                ds_groups[ds_type] = []
            ds_groups[ds_type].append(entry)

        weight_map = {}
        for config in configs:
            ds_type = config.get('type', '').lower()
            weight_map[ds_type] = config.get('weight', 1.0)

        total_weight = sum(weight_map.get(k, 1.0) for k in ds_groups)
        if total_weight <= 0:
            return index

        max_samples_per_weight = 0
        for ds_type, entries in ds_groups.items():
            w = weight_map.get(ds_type, 1.0)
            if w > 0:
                max_samples_per_weight = max(max_samples_per_weight, len(entries) / w)

        balanced_index = []
        for ds_type, entries in ds_groups.items():
            w = weight_map.get(ds_type, 1.0)
            target_n = int(max_samples_per_weight * w)
            if self.max_samples_per_dataset is not None:
                target_n = min(target_n, self.max_samples_per_dataset)

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
                logger.info(f"  Balanced: {ds_type} {len(entries)} → {target_n} (weight={w})")

        random.shuffle(balanced_index)
        return balanced_index

    def _apply_max_cap(self, index: List[Dict], max_samples: int) -> List[Dict]:
        ds_groups: Dict[str, List[Dict]] = {}
        for entry in index:
            ds_type = entry['dataset_type']
            if ds_type not in ds_groups:
                ds_groups[ds_type] = []
            ds_groups[ds_type].append(entry)

        capped_index = []
        for ds_type, entries in ds_groups.items():
            if len(entries) > max_samples:
                step = len(entries) / max_samples
                selected = [entries[int(i * step)] for i in range(max_samples)]
                capped_index.extend(selected)
                if is_main_process():
                    logger.info(f"  Capped: {ds_type} {len(entries)} → {max_samples}")
            else:
                capped_index.extend(entries)

        return capped_index

    # ------------------------------------------------------------------ #
    #  Shape 分组
    # ------------------------------------------------------------------ #

    def _build_shape_groups(self):
        self.shape_groups = {}
        for i, entry in enumerate(self.index):
            ds_type = entry.get('dataset_type', '')
            rule = self.DATASET_RULES.get(ds_type, {})
            sg = rule.get('eeg_shape_group', entry.get('shape_group', (0, 0)))
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

    def __getitem__(self, idx):
        entry = self.index[idx]
        ds_type = entry['dataset_type']
        file_path = entry['file_path']
        rule = self.DATASET_RULES[ds_type]

        # ---- 加载文件 ----
        if rule['file_format'] == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:  # 'pt'
            data = torch.load(file_path, map_location='cpu', weights_only=False)

        # ---- 提取 EEG ----
        eeg = data[rule['eeg_key']]
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float()
        elif isinstance(eeg, torch.Tensor):
            eeg = eeg.float()
        else:
            eeg = torch.tensor(eeg, dtype=torch.float32)

        # ---- EEGPT 通道过滤 ----
        if ds_type == 'eegpt_pkl' and 'filtered_ch_names' in entry:
            raw_ch = data.get('ch_names', [])
            filtered = entry['filtered_ch_names']
            if len(filtered) < len(raw_ch):
                import sys
                labram_utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
                if labram_utils_path not in sys.path:
                    sys.path.insert(0, labram_utils_path)
                from labram_utils import standard_1020 as labram_1020
                labram_set = set(labram_1020)
                keep_idx = [i for i, c in enumerate(raw_ch) if c in labram_set]
                eeg = eeg[keep_idx]

        if rule['eeg_reshape'] == 'auto':
            if eeg.dim() == 2:
                ps = rule['patch_size']
                eeg = eeg.reshape(eeg.shape[0], -1, ps)

        if ds_type in ('things', 'things_v1', 'alljoined') and eeg.dim() == 2:
            eeg = eeg.unsqueeze(1)

        if rule['eeg_truncate_patches'] is not None:
            eeg = eeg[:, :rule['eeg_truncate_patches'], :]

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
        raw_label = data.get(label_key, 0)
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
            for filename in sorted(os.listdir(mode_dir)):
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
