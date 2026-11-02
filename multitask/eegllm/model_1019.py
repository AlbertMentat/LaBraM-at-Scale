"""EEGLLM variant (1019): per-sample prompt keys (template id + option order) and a differentiable per-sample loss.

A prompt key is either a task name (canonical 1018 prompt) or a tuple (task, template_id, option_order) where
option_order is a tuple of class indices.  Answers/candidates stay in canonical class order, so closed-set scores
index classes identically for every prompt variant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import EEGLLM, autocast
from .prompts_1019 import prefix_text_v


class _PrefixCache(dict):
    def __init__(self, model, base):
        super().__init__(base)
        self.model = model

    def __missing__(self, key):
        task, tpl, order = key
        classes = self.model.specs[task].classes
        text = prefix_text_v(task, tpl, [classes[i] for i in order], self.model.K)
        ids = self.model.tok(text, add_special_tokens=False).input_ids
        if ids.count(self.model.eeg_pad_id) != self.model.K:
            raise RuntimeError(f'{key}: wrong EEG placeholder count')
        self[key] = ids
        return ids


class _CandCache(dict):
    def __missing__(self, key):
        return self[key[0]]


class EEGLLM1019(EEGLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prefix_ids = _PrefixCache(self, self.prefix_ids)
        self.cand_ids = _CandCache(self.cand_ids)

    def loss_per_sample(self, batch: dict, device) -> torch.Tensor:
        """Same computation as EEGLLM.forward but returns the (differentiable) per-sample mean answer-token CE."""
        tasks, labels = batch['tasks'], batch['labels'].tolist()
        eeg = self.encode(batch['groups'], len(tasks), device)
        with autocast(self.compute):
            emb, attn, tgt = self.assemble(tasks, eeg, labels, left_pad=False)
            hidden = self.backbone(inputs_embeds=emb, attention_mask=attn, use_cache=False).last_hidden_state
            sel = tgt[:, 1:] != -100
            logits = self.lm_head(hidden[:, :-1][sel])
        y = tgt[:, 1:][sel]
        tok_loss = F.cross_entropy(logits.float(), y, reduction='none')
        rows = sel.nonzero(as_tuple=True)[0]
        per = torch.zeros(len(tasks), device=device).index_add(0, rows, tok_loss)
        return per / sel.sum(1).clamp(min=1)
