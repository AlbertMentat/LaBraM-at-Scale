"""
Task-aware Query Selection (TQS) module — adapted from UniMind (arxiv 2506.18962).

In the original paper, TQS sits between the EEG encoder and a frozen LLM,
selecting task-adaptive query tokens that drive cross-attention with EEG features.
Here we adapt it for direct EEG classification (no LLM): EEG encoder → TQS →
classification head.

Architecture:
  1. Pretrained encoder produces patch tokens [B, N, D]
  2. Router: simple linear projection of pooled features → routing scores over
     a pool of M learnable queries
  3. Top-k selection: pick top-k queries per sample based on routing scores,
     weighted by softmax of scores
  4. Cross-attention: selected queries attend to EEG patch features
  5. Pool → classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TQSHead(nn.Module):
    """Task-aware Query Selection head replacing the simple linear classifier.

    Args:
        embed_dim: encoder output dim
        num_classes: number of classes (1 for binary, k for multiclass)
        query_pool_size: M, total queries in the pool
        top_k: k, number of queries selected per sample
        num_heads: number of attention heads in cross-attention
        router_dim: hidden dim for the router (0 = direct linear)
        use_static_query: also include 1 static query (always activated)
    """

    def __init__(self, embed_dim, num_classes,
                 query_pool_size=16, top_k=4,
                 num_heads=4, router_dim=0,
                 use_static_query=True,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_pool_size = query_pool_size
        self.top_k = min(top_k, query_pool_size)
        self.use_static_query = use_static_query

        # Learnable query pool
        self.query_pool = nn.Parameter(torch.randn(query_pool_size, embed_dim) * 0.02)

        # Optional static query (UniMind paper found 1 static + k dynamic works best)
        if use_static_query:
            self.static_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Router: linear or MLP
        if router_dim > 0:
            self.router = nn.Sequential(
                nn.Linear(embed_dim, router_dim),
                nn.GELU(),
                nn.Linear(router_dim, query_pool_size),
            )
        else:
            self.router = nn.Linear(embed_dim, query_pool_size)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, features, return_routing=False):
        """
        Args:
            features: [B, N, D] patch tokens from the encoder (no CLS)
            return_routing: if True, also return routing weights for analysis

        Returns:
            logits: [B, num_classes]
            (optional) routing_info: dict with 'scores' and 'topk_idx'
        """
        B, N, D = features.shape

        # 1. Pool features for routing context
        ctx = features.mean(dim=1)  # [B, D]

        # 2. Router scores
        scores = self.router(ctx)  # [B, M]

        # 3. Top-k selection
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)  # [B, k]
        topk_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # [B, k, 1]

        # Gather selected queries
        # query_pool: [M, D] → broadcast → gather along dim=1
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        pool_expanded = self.query_pool.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        dynamic_q = torch.gather(pool_expanded, 1, idx_expanded)  # [B, k, D]
        dynamic_q = dynamic_q * topk_weights

        # 4. Optionally prepend static query
        if self.use_static_query:
            static = self.static_query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            queries = torch.cat([static, dynamic_q], dim=1)  # [B, k+1, D]
        else:
            queries = dynamic_q

        # 5. Cross-attention
        q = self.norm_q(queries)
        kv = self.norm_kv(features)
        attended, _ = self.cross_attn(query=q, key=kv, value=kv)
        attended = self.norm_out(queries + attended)  # residual + norm

        # 6. Pool selected queries → classify
        pooled = attended.mean(dim=1)  # [B, D]
        logits = self.head(pooled)

        if return_routing:
            return logits, {
                'scores': scores,
                'topk_idx': topk_idx,
                'topk_weights': topk_weights.squeeze(-1),
            }
        return logits


class TQSModelWrapper(nn.Module):
    """Wraps a LaBraM encoder with a TQS head for classification.

    The encoder forward should be called with `return_patch_tokens=True` to get
    patch features; the original head is bypassed.
    """

    def __init__(self, base_encoder, num_classes,
                 query_pool_size=16, top_k=4, num_heads=4,
                 router_dim=0, use_static_query=True, dropout=0.1):
        super().__init__()
        self.encoder = base_encoder
        embed_dim = base_encoder.embed_dim

        self.tqs = TQSHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            query_pool_size=query_pool_size,
            top_k=top_k,
            num_heads=num_heads,
            router_dim=router_dim,
            use_static_query=use_static_query,
            dropout=dropout,
        )
        # Disable the encoder's built-in head so it doesn't waste params
        self.encoder.head = nn.Identity()

    @property
    def patch_size(self):
        return self.encoder.patch_size

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    def no_weight_decay(self):
        no_wd = self.encoder.no_weight_decay() if hasattr(self.encoder, 'no_weight_decay') else set()
        no_wd = set(no_wd)
        no_wd.update({'tqs.query_pool', 'tqs.static_query'})
        return no_wd

    def forward(self, x, input_chans=None, return_routing=False, **kwargs):
        # Get patch features (without classification, without mean pooling)
        # forward_features with return_patch_tokens=True returns [B, N, D] (patch tokens, no CLS)
        features = self.encoder.forward_features(
            x, input_chans=input_chans, return_patch_tokens=True,
        )
        if return_routing:
            return self.tqs(features, return_routing=True)
        logits = self.tqs(features)
        return logits
