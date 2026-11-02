"""EEG patch features [B, C*A, D_enc] -> K LLM tokens [B, K, H_llm] (fixed K for any channel/duration)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGProjector(nn.Module):
    """pooling='attn': K learned queries cross-attend over all C*A patch tokens (1 layer, pre-LN), then MLP.
    pooling='avg'  : mean over channels -> [B, A, D], adaptive average pool (or linear interpolation when A < K)
                     over time to K, then MLP.
    The output is LayerNorm-ed (no affine) and multiplied by a learned scale initialised to the RMS of the LLM
    input embeddings, so EEG tokens start on the same scale as text tokens."""

    def __init__(self, enc_dim: int, llm_dim: int, num_tokens: int = 32, pooling: str = 'attn',
                 num_heads: int = 8, mlp_ratio: float = 1.0, embed_rms: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if pooling not in ('attn', 'avg'):
            raise ValueError(pooling)
        self.pooling, self.num_tokens = pooling, num_tokens
        if pooling == 'attn':
            if enc_dim % num_heads:
                raise ValueError(f'enc_dim {enc_dim} not divisible by heads {num_heads}')
            self.queries = nn.Parameter(torch.randn(num_tokens, enc_dim) * 0.02)
            self.ln_q = nn.LayerNorm(enc_dim)
            self.ln_kv = nn.LayerNorm(enc_dim)
            self.attn = nn.MultiheadAttention(enc_dim, num_heads, dropout=dropout, batch_first=True)
            self.ln_ff = nn.LayerNorm(enc_dim)
            self.ff = nn.Sequential(nn.Linear(enc_dim, 4 * enc_dim), nn.GELU(), nn.Linear(4 * enc_dim, enc_dim))
        hidden = int(llm_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, hidden), nn.GELU(), nn.Linear(hidden, llm_dim))
        self.out_norm = nn.LayerNorm(llm_dim, elementwise_affine=False)
        self.out_scale = nn.Parameter(torch.tensor(float(embed_rms)))

    def forward(self, feats: torch.Tensor, n_channels: int, n_patches: int) -> torch.Tensor:
        b, l, d = feats.shape
        if l != n_channels * n_patches:
            raise ValueError(f'{l} tokens != {n_channels} x {n_patches}')
        if self.pooling == 'attn':
            q = self.ln_q(self.queries).unsqueeze(0).expand(b, -1, -1)
            kv = self.ln_kv(feats)
            x = self.queries.unsqueeze(0) + self.attn(q, kv, kv, need_weights=False)[0]
            x = x + self.ff(self.ln_ff(x))
        else:
            x = feats.view(b, n_channels, n_patches, d).mean(1).transpose(1, 2)  # [B, D, A]
            if n_patches >= self.num_tokens:
                x = F.adaptive_avg_pool1d(x, self.num_tokens)
            else:
                x = F.interpolate(x, size=self.num_tokens, mode='linear', align_corners=False)
            x = x.transpose(1, 2)
        return self.out_norm(self.mlp(x)) * self.out_scale
