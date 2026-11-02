"""EEG encoder -> projector -> K EEG tokens spliced into a Qwen3 prompt; answer = class name text."""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .prompts import EEG_PAD, IM_END, prefix_text, target_text

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def autocast(dtype: torch.dtype | None):
    if dtype is None or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=dtype)


class EEGLLM(nn.Module):
    def __init__(self, encoder, projector, causal_lm, tokenizer, specs: dict, input_chans: dict[str, list[int]],
                 num_tokens: int, input_scale: float, zscore: bool, encoder_amp: str, compute: str):
        super().__init__()
        self.encoder, self.projector, self.causal_lm = encoder, projector, causal_lm
        self._peft = []  # peft wrapper kept outside the module tree (params live in causal_lm)
        self.tok = tokenizer
        self.specs = specs
        self.K = num_tokens
        self.input_scale, self.zscore = float(input_scale), bool(zscore)
        self.encoder_amp = DTYPES[encoder_amp]
        self.compute = DTYPES[compute]
        self.eeg_pad_id = tokenizer.convert_tokens_to_ids(EEG_PAD)
        self.im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.input_chans = {t: torch.tensor(c, dtype=torch.long) for t, c in input_chans.items()}
        # per-task, per-dimension standardisation of frozen patch features (train-split statistics).  Atom-L patch
        # tokens are highly anisotropic (mean pairwise cosine ~0.99 between samples), so without this the projector
        # output is almost sample-independent and the LLM learns only the label prior.
        self.feat_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.prefix_ids, self.cand_ids = {}, {}
        for t, s in specs.items():
            ids = tokenizer(prefix_text(s.instruction, s.classes, num_tokens), add_special_tokens=False).input_ids
            if ids.count(self.eeg_pad_id) != num_tokens:
                raise RuntimeError(f'{t}: expected {num_tokens} EEG placeholders, got {ids.count(self.eeg_pad_id)}')
            self.prefix_ids[t] = ids
            self.cand_ids[t] = [tokenizer(target_text(c), add_special_tokens=False).input_ids for c in s.classes]
            for c in self.cand_ids[t]:
                if c[-1] != self.im_end_id or self.eeg_pad_id in c:
                    raise RuntimeError(f'{t}: bad candidate tokenization {c}')

    # ------------------------------------------------------------------ helpers
    @property
    def backbone(self):
        return self.causal_lm.model

    @property
    def lm_head(self):
        return self.causal_lm.lm_head

    def add_lora(self, lcfg: dict):
        from peft import LoraConfig, get_peft_model
        conf = LoraConfig(r=int(lcfg['r']), lora_alpha=int(lcfg['alpha']), lora_dropout=float(lcfg.get('dropout', 0.0)),
                          target_modules=list(lcfg['targets']), bias='none', task_type='CAUSAL_LM')
        wrapper = get_peft_model(self.causal_lm, conf)
        self._peft = [wrapper]
        self.causal_lm = wrapper.base_model.model
        return wrapper

    @property
    def has_lora(self) -> bool:
        return bool(self._peft)

    # ------------------------------------------------------------------ EEG -> tokens
    def encode(self, groups: dict, batch_size: int, device) -> torch.Tensor:
        """groups: task -> {'pos': [...], 'eeg': [B_t, C, N*200]} or, with precomputed frozen encoder outputs,
        {'pos': [...], 'feats': [B_t, C*A, D], 'shape': (C, A)}; returns [B, K, H] in batch order."""
        out = None
        enc_trainable = self.encoder is not None and any(p.requires_grad for p in self.encoder.parameters())
        for task, g in groups.items():
            if 'feats' in g:
                feats = g['feats'].to(device, non_blocking=True)
                n_ch, n_p = g['shape']
            else:
                x = g['eeg'].to(device, non_blocking=True).float()
                if self.zscore:  # dimensionless_zscore contract (79k encoders); not used for Atom-L
                    x = ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)).clamp(-15, 15)
                x = x * self.input_scale
                n_ch = x.shape[1]
                x = rearrange(x, 'B C (A T) -> B C A T', T=200)
                n_p = x.shape[2]
                chans = self.input_chans[task].to(device)
                with torch.set_grad_enabled(enc_trainable and torch.is_grad_enabled()), autocast(self.encoder_amp):
                    feats = self.encoder.forward_features(x, input_chans=chans, return_patch_tokens=True)
            feats = feats.float()
            if task in self.feat_stats:
                mu, sd = self.feat_stats[task]
                feats = (feats - mu.to(device)) / sd.to(device)
            with autocast(self.compute):
                tok = self.projector(feats, n_ch, n_p)
            if out is None:
                out = tok.new_zeros(batch_size, tok.shape[1], tok.shape[2])
            idx = torch.tensor(g['pos'], device=device)
            out = out.index_copy(0, idx, tok.to(out.dtype))
        return out

    @torch.no_grad()
    def encoder_features(self, task: str, eeg: torch.Tensor, device) -> torch.Tensor:
        x = eeg.to(device).float()
        if self.zscore:
            x = ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)).clamp(-15, 15)
        x = rearrange(x * self.input_scale, 'B C (A T) -> B C A T', T=200)
        with autocast(self.encoder_amp):
            f = self.encoder.forward_features(x, input_chans=self.input_chans[task].to(device), return_patch_tokens=True)
        return f.float()

    # ------------------------------------------------------------------ sequence assembly
    def assemble(self, tasks: list[str], eeg_tokens: torch.Tensor, labels: list[int] | None, left_pad: bool):
        device = eeg_tokens.device
        seqs, n_prefix = [], []
        for i, t in enumerate(tasks):
            ids = list(self.prefix_ids[t])
            n_prefix.append(len(ids))
            if labels is not None:
                ids = ids + self.cand_ids[t][int(labels[i])]
            seqs.append(ids)
        L = max(len(s) for s in seqs)
        B = len(seqs)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            off = L - len(s) if left_pad else 0
            ids[i, off:off + len(s)] = torch.tensor(s)
            attn[i, off:off + len(s)] = 1
            if labels is not None:
                tgt[i, off + n_prefix[i]:off + len(s)] = torch.tensor(s[n_prefix[i]:])
        ids, attn, tgt = ids.to(device), attn.to(device), tgt.to(device)
        emb = self.causal_lm.get_input_embeddings()(ids)
        mask = (ids == self.eeg_pad_id).unsqueeze(-1)
        emb = emb.masked_scatter(mask, eeg_tokens.to(emb.dtype).reshape(-1, emb.shape[-1]))
        return emb, attn, tgt

    # ------------------------------------------------------------------ training loss
    def forward(self, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
        tasks, labels = batch['tasks'], batch['labels'].tolist()
        eeg = self.encode(batch['groups'], len(tasks), device)
        with autocast(self.compute):
            emb, attn, tgt = self.assemble(tasks, eeg, labels, left_pad=False)
            hidden = self.backbone(inputs_embeds=emb, attention_mask=attn, use_cache=False).last_hidden_state
            sel = tgt[:, 1:] != -100
            h = hidden[:, :-1][sel]
            logits = self.lm_head(h)
        y = tgt[:, 1:][sel]
        tok_loss = F.cross_entropy(logits.float(), y, reduction='none')
        rows = sel.nonzero(as_tuple=True)[0]
        per_sample = torch.zeros(len(tasks), device=device).index_add_(0, rows, tok_loss)
        per_sample = per_sample / sel.sum(1).clamp(min=1)
        return per_sample.mean(), per_sample.detach()

    # ------------------------------------------------------------------ closed-set scoring
    @torch.no_grad()
    def score(self, batch: dict, device, logit_chunk: int = 512) -> list[torch.Tensor]:
        """Log-likelihood of every allowed answer (class name + <|im_end|>) given the prompt.
        Prefix is run once (left padded, KV cache); candidates reuse the cache repeated per class."""
        from transformers import DynamicCache
        tasks = batch['tasks']
        eeg = self.encode(batch['groups'], len(tasks), device)
        results: list[torch.Tensor | None] = [None] * len(tasks)
        with autocast(self.compute):
            emb, attn, _ = self.assemble(tasks, eeg, None, left_pad=True)
            out = self.backbone(inputs_embeds=emb, attention_mask=attn, use_cache=True)
            lp_first = F.log_softmax(self.lm_head(out.last_hidden_state[:, -1]).float(), -1)
            cache = out.past_key_values
            for task in dict.fromkeys(tasks):
                rows = [i for i, t in enumerate(tasks) if t == task]
                cands = self.cand_ids[task]
                M, Lc = len(cands), max(len(c) for c in cands)
                cid = torch.full((M, Lc), self.pad_id, dtype=torch.long, device=device)
                cm = torch.zeros((M, Lc), dtype=torch.long, device=device)
                for m, c in enumerate(cands):
                    cid[m, :len(c)] = torch.tensor(c, device=device)
                    cm[m, :len(c)] = 1
                r = torch.tensor(rows, device=device)
                sub = DynamicCache()
                for layer in range(len(cache.key_cache)):
                    sub.update(cache.key_cache[layer][r].repeat_interleave(M, 0),
                               cache.value_cache[layer][r].repeat_interleave(M, 0), layer)
                R = len(rows)
                inp = cid[:, :-1].repeat(R, 1)
                cmask = cm.repeat(R, 1)
                am = torch.cat([attn[r].repeat_interleave(M, 0), cmask[:, :-1]], dim=1)
                total = lp_first[r].repeat_interleave(M, 0).gather(1, cid[:, :1].repeat(R, 1)).squeeze(1)
                if Lc > 1:
                    hid = self.backbone(input_ids=inp, attention_mask=am, past_key_values=sub,
                                        use_cache=True).last_hidden_state
                    nxt = cid[:, 1:].repeat(R, 1)
                    w = cmask[:, 1:].float()
                    flat_h = hid.reshape(-1, hid.shape[-1])
                    flat_y = nxt.reshape(-1)
                    lps = []
                    for s in range(0, flat_h.shape[0], logit_chunk):
                        lg = F.log_softmax(self.lm_head(flat_h[s:s + logit_chunk]).float(), -1)
                        lps.append(lg.gather(1, flat_y[s:s + logit_chunk, None]).squeeze(1))
                    total = total + (torch.cat(lps).view_as(w) * w).sum(1)
                total = total.view(R, M).float().cpu()
                for k, i in enumerate(rows):
                    results[i] = total[k]
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ free generation
    @torch.no_grad()
    def generate(self, batch: dict, device, max_new_tokens: int = 16) -> list[str]:
        tasks = batch['tasks']
        eeg = self.encode(batch['groups'], len(tasks), device)
        with autocast(self.compute):
            emb, attn, _ = self.assemble(tasks, eeg, None, left_pad=True)
            out = self.causal_lm.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new_tokens,
                                          do_sample=False, temperature=None, top_p=None, top_k=None,
                                          eos_token_id=[self.im_end_id, self.tok.eos_token_id], pad_token_id=self.pad_id)
        return self.tok.batch_decode(out, skip_special_tokens=False)

    # ------------------------------------------------------------------ trainable state
    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state: dict[str, torch.Tensor]):
        params = dict(self.named_parameters())
        missing = [k for k in state if k not in params]
        if missing:
            raise KeyError(f'unknown trainable params: {missing[:5]}')
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v.to(params[k].dtype))
