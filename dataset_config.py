"""Dataset settings aligned with MST-OATD (KDD 2024)."""

DATASETS = {
    'porto': {
        'map_size': (51, 119),
        'time_interval': 15,
        'data_prefix': './data/porto/processed_porto',
        'mst_data_dir': './data/porto',
        'ckpt_root': './ckpt/porto',
        'pretrain_root': './pretrain/porto',
    },
    'cd': {
        'map_size': (167, 154),
        'time_interval': 10,
        'data_prefix': './data/cd/processed_cd',
        'mst_data_dir': './data/cd',
        'ckpt_root': './ckpt/cd',
        'pretrain_root': './pretrain/cd',
    },
}


def apply_dataset_args(args):
    """Override paths and map size from --dataset when provided."""
    dataset = getattr(args, 'dataset', None) or ''
    dataset = dataset.strip()
    if not dataset:
        return args
    if dataset not in DATASETS:
        raise ValueError('dataset must be one of: {}'.format(list(DATASETS.keys())))
    cfg = DATASETS[dataset]
    args.map_size = cfg['map_size']
    args.data_filename = cfg['data_prefix'] + '.csv'
    mode = getattr(args, 'mode', '')
    if mode == 'pretrain':
        if getattr(args, 'model_dir', None) in (None, 'ckpt', './ckpt', 'pretrain', './pretrain'):
            args.model_dir = cfg['pretrain_root']
    else:
        if getattr(args, 'model_dir', None) in (None, 'ckpt', './ckpt'):
            args.model_dir = cfg['ckpt_root']
        if mode != 'pretrain' and getattr(args, 'pretrain_dir', None) in (None, '', 'pretrain', './pretrain'):
            args.pretrain_dir = cfg['pretrain_root']
    args.time_interval = cfg['time_interval']
    args.mst_data_dir = cfg['mst_data_dir']
    return args
