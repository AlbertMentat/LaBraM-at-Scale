"""Build the EEG encoder with the same model and input settings as Full-FT."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from .config import resolve_path

ALLOWED_MISSING = {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias', 'logit_scale'}  # = FT runtime's new params
SAFE_GLOBALS = [argparse.Namespace, np.dtype, np.dtypes.Float64DType, np._core.multiarray.scalar,
                (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar')]


def _import_repo(repo_root: str):
    root = resolve_path(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import labram_utils  # noqa: E402
    import models.labram  # noqa: F401,E402  (registers labram_* with timm)
    return labram_utils


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_load_checkpoint(path: str) -> dict:
    if Path(path).suffix == '.safetensors':
        from safetensors.torch import load_file
        return {'model': load_file(path, device='cpu')}
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return torch.load(path, map_location='cpu', weights_only=True)


def build_encoder(ecfg: dict, verify_sha: bool = True):
    """Returns (encoder, labram_utils module, load report)."""
    lu = _import_repo(ecfg.get('repo_root', '..'))
    from timm.models import create_model
    model = create_model(
        ecfg['model'], pretrained=False, num_classes=1, drop_rate=0.0,
        drop_path_rate=float(ecfg.get('drop_path', 0.0)), attn_drop_rate=0.0, drop_block_rate=None,
        use_mean_pooling=True, use_no_pool=False, no_pool_tokens=None, init_scale=0.001,
        use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    report = {'checkpoint': ecfg['checkpoint']}
    path = ecfg['checkpoint']
    if verify_sha and ecfg.get('checkpoint_sha256'):
        sha = file_sha256(path)
        report['sha256'] = sha
        if sha != ecfg['checkpoint_sha256']:
            raise RuntimeError(f'encoder checkpoint sha mismatch {sha} != {ecfg["checkpoint_sha256"]}')
    ckpt = safe_load_checkpoint(path)
    sd = None
    for key in 'model|module'.split('|'):
        if key in ckpt:
            sd = ckpt[key]
            break
    sd = dict(sd if sd is not None else ckpt)
    if any(k.startswith('student.') for k in sd):  # model_filter_name branch of the FT runtime
        sd = {k[8:]: v for k, v in sd.items() if k.startswith('student.')}
    state = model.state_dict()
    for k in ('head.weight', 'head.bias'):
        if k in sd and (k not in state or sd[k].shape != state[k].shape):
            del sd[k]
    sd = {k: v for k, v in sd.items() if 'relative_position_index' not in k}
    missing = sorted(set(state) - set(sd))
    unexpected = sorted(set(sd) - set(state))
    lu.load_state_dict(model, sd, prefix='')
    bad = [k for k in missing if k not in ALLOWED_MISSING]
    if bad:
        raise RuntimeError(f'encoder weights missing from checkpoint: {bad}')
    report.update({'missing': missing, 'unexpected': unexpected, 'epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None,
                   'embed_dim': model.embed_dim, 'depth': len(model.blocks)})
    return model, lu, report


def set_encoder_trainable(encoder, unfreeze_last_n: int) -> list[str]:
    for p in encoder.parameters():
        p.requires_grad_(False)
    names = []
    if unfreeze_last_n > 0:
        for blk in encoder.blocks[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in encoder.fc_norm.parameters():
            p.requires_grad_(True)
        names = [n for n, p in encoder.named_parameters() if p.requires_grad]
    return names

