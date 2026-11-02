"""Public entrypoint helpers; numerical implementations remain in the dated modules."""
import importlib


def validate_config(cfg):
    if 'implementation' not in cfg:
        # Existing experiment contracts predate the public routing field.
        name = cfg.get('run_name', '')
        if name.startswith('llm1020d__'):
            cfg['implementation'] = 'original'
        elif name.startswith('llm1020dg__'):
            cfg['implementation'] = 'grid'
    if cfg.get('implementation') not in ('original', 'grid'):
        raise ValueError('implementation must be original or grid')
    if not cfg.get('tasks') or len(cfg['tasks']) != len(set(cfg['tasks'])):
        raise ValueError('tasks must be a nonempty list without duplicates')
    if cfg['train']['batch_size'] <= 0 or cfg['train']['eval_every'] <= 0:
        raise ValueError('batch_size and eval_every must be positive')
    if cfg['implementation'] == 'original' and cfg['encoder']['model'] != 'labram_large_patch200_200':
        raise ValueError('The original implementation requires a Large encoder; use grid for other widths')


def backend(cfg):
    # Import just one path: the grid path patches the shared projector constructor.
    validate_config(cfg)
    name = 'train_1020d_r3' if cfg['implementation'] == 'original' else 'train_1020d_grid'
    return importlib.import_module(name)


def load_data(cfg, distributed):
    import torch
    names = cfg['tasks']
    if cfg['implementation'] == 'original':
        import train_1020
        dev = distributed.device if cfg['data'].get('feats_on_gpu', False) else torch.device('cpu')
        meta, data, stats = train_1020.load_seed(cfg, names, cfg['seed'], dev)
    else:
        import train_1020b
        meta, data, stats = train_1020b.load_seed_views(cfg, names, cfg['seed'], distributed)
    stats = {t: (mean.to(distributed.device), std.to(distributed.device)) for t, (mean, std) in stats.items()}
    return meta, data, stats

