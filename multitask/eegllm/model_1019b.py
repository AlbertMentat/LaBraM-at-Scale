"""EEGLLM variant for 40 GB GPUs (1019b).

* LM-head logits only where needed: training uses the hidden states at answer positions only
  (inherited loss_per_sample); closed-set scoring uses the last prefix position + candidate positions, with
  candidate rows processed in chunks (`cand_rows`) so the repeated KV cache stays bounded, and LM-head rows in
  chunks (`logit_chunk`).  The full [B, T, vocab] logits tensor is never built.
* Optional gradient checkpointing on the LLM (non-reentrant; active only in train mode).
Numerically equivalent to full-logit computation (tests/test_1019b.py, tools/mem_1019b.py --check).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import autocast
from .model_1019 import EEGLLM1019


class EEGLLM1019b(EEGLLM1019):
    cand_rows: int = 256
    logit_chunk: int = 512

    def enable_gradient_checkpointing(self):
        self.causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.causal_lm.config.use_cache = False

    @torch.no_grad()
    def score(self, batch: dict, device, logit_chunk: int | None = None) -> list[torch.Tensor]:
        from transformers import DynamicCache
        chunk = int(logit_chunk or self.logit_chunk)
        tasks = batch['tasks']
        eeg = self.encode(batch['groups'], len(tasks), device)
        results: list = [None] * len(tasks)
        with autocast(self.compute):
            emb, attn, _ = self.assemble(tasks, eeg, None, left_pad=True)
            out = self.backbone(inputs_embeds=emb, attention_mask=attn, use_cache=True)
            lp_first = F.log_softmax(self.lm_head(out.last_hidden_state[:, -1]).float(), -1)
            cache = out.past_key_values
            del out, emb
            for key in dict.fromkeys(tasks):
                rows_all = [i for i, t in enumerate(tasks) if t == key]
                cands = self.cand_ids[key]
                M, Lc = len(cands), max(len(c) for c in cands)
                cid = torch.full((M, Lc), self.pad_id, dtype=torch.long, device=device)
                cm = torch.zeros((M, Lc), dtype=torch.long, device=device)
                for m, c in enumerate(cands):
                    cid[m, :len(c)] = torch.tensor(c, device=device)
                    cm[m, :len(c)] = 1
                step = max(1, int(self.cand_rows) // M)
                for s in range(0, len(rows_all), step):
                    rows = rows_all[s:s + step]
                    R = len(rows)
                    r = torch.tensor(rows, device=device)
                    total = lp_first[r].repeat_interleave(M, 0).gather(1, cid[:, :1].repeat(R, 1)).squeeze(1)
                    if Lc > 1:
                        sub = DynamicCache()
                        for layer in range(len(cache.key_cache)):
                            sub.update(cache.key_cache[layer][r].repeat_interleave(M, 0),
                                       cache.value_cache[layer][r].repeat_interleave(M, 0), layer)
                        cmask = cm.repeat(R, 1)
                        am = torch.cat([attn[r].repeat_interleave(M, 0), cmask[:, :-1]], dim=1)
                        hid = self.backbone(input_ids=cid[:, :-1].repeat(R, 1), attention_mask=am, past_key_values=sub,
                                            use_cache=True).last_hidden_state
                        del sub
                        w = cmask[:, 1:].float()
                        flat_h = hid.reshape(-1, hid.shape[-1])
                        flat_y = cid[:, 1:].repeat(R, 1).reshape(-1)
                        keep = w.reshape(-1) > 0  # only real candidate tokens go through the LM head
                        lp = torch.zeros(flat_y.shape[0], device=device)
                        idx = keep.nonzero(as_tuple=True)[0]
                        for c0 in range(0, idx.numel(), chunk):
                            ii = idx[c0:c0 + chunk]
                            lg = F.log_softmax(self.lm_head(flat_h[ii]).float(), -1)
                            lp[ii] = lg.gather(1, flat_y[ii, None]).squeeze(1)
                        total = total + lp.view_as(w).sum(1)
                    total = total.view(R, M).float().cpu()
                    for k, i in enumerate(rows):
                        results[i] = total[k]
        return results
