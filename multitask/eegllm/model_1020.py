"""EEGLLM for the 23-task runs (1020): model_1019b behaviour with the 1020 multi-template prompts."""
from __future__ import annotations

from . import prompts_1020 as P
from .model_1019 import _CandCache
from .model_1019b import EEGLLM1019b


class _PrefixCache1020(dict):
    def __init__(self, model, base):
        super().__init__(base)
        self.model = model

    def __missing__(self, key):
        task, tpl, order = key
        classes = self.model.specs[task].classes
        ids = self.model.tok(P.prefix_text_v(task, tpl, [classes[i] for i in order], self.model.K),
                             add_special_tokens=False).input_ids
        if ids.count(self.model.eeg_pad_id) != self.model.K:
            raise RuntimeError(f'{key}: wrong EEG placeholder count')
        self[key] = ids
        return ids


class EEGLLM1020(EEGLLM1019b):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        P.init_templates(self.specs)
        self.prefix_ids = _PrefixCache1020(self, dict(self.prefix_ids))
        self.cand_ids = _CandCache(dict(self.cand_ids))
