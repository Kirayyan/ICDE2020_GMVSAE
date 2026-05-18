#!/usr/bin/env python3
"""
Score only explicitly listed outliers_data_*.npy files (each must have outliers_idx_ pair).

Usage:
  python3 scripts/score_selected_npy.py --config configs/cd_selected_anomalies.json
  python3 scripts/score_selected_npy.py --dataset cd --data outliers_data_init_stay_0p3_25_1.npy ...
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_io import infer_anomaly_kind_from_filename
from dataset_config import DATASETS


def idx_path_for(data_path):
    return data_path.replace('outliers_data', 'outliers_idx', 1)


def resolve_data_file(data_dir, name):
    if os.path.isabs(name):
        return name
    return os.path.join(data_dir, os.path.basename(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='JSON job list, see configs/cd_selected_anomalies.json')
    parser.add_argument('--dataset', default='cd')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--data', nargs='*', default=[], help='outliers_data_*.npy basenames or paths')
    parser.add_argument('--eval_split', default='val', choices=['train', 'val'])
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--train_if_missing', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    py = sys.executable

    jobs = []
    dataset = args.dataset
    data_dir = args.data_dir
    eval_split = args.eval_split

    if args.config:
        with open(args.config, 'r') as fp:
            cfg = json.load(fp)
        dataset = cfg.get('dataset', dataset)
        data_dir = cfg.get('data_dir', data_dir)
        eval_split = cfg.get('eval_split', eval_split)
        jobs = [j['data'] for j in cfg.get('jobs', [])]

    jobs.extend(args.data)
    if not jobs:
        parser.error('Provide --config or at least one --data file')

    dcfg = DATASETS[dataset]
    data_dir = data_dir or dcfg['mst_data_dir']

    if not os.path.isfile(dcfg['data_prefix'] + '_val.csv'):
        train_npy = os.path.join(data_dir, 'train_data_init.npy')
        test_npy = os.path.join(data_dir, 'test_data_init.npy')
        if os.path.isfile(train_npy) and os.path.isfile(test_npy):
            subprocess.check_call([
                py, os.path.join(root, 'scripts/convert_mst_npy_to_gmvsae.py'),
                '--dataset', dataset,
            ], env=env)
        else:
            raise FileNotFoundError('Need processed csv or train/test npy under ' + data_dir)

    ckpt = os.path.join(dcfg['ckpt_root'], 'gmvsae_32_256_5')
    if args.train_if_missing and not os.path.exists(os.path.join(ckpt, 'checkpoint')):
        for mode, extra in [
            ('pretrain', ['--num_epochs', str(args.num_epochs)]),
            ('train', ['--num_epochs', str(args.num_epochs), '--pretrain_dir', dcfg['pretrain_root']]),
        ]:
            subprocess.check_call([
                py, os.path.join(root, 'run_loop.py'), '--mode', mode,
                '--dataset', dataset, '--cluster_num', str(args.cluster_num),
                '--batch_size', '128', '--gpu_id', args.gpu_id,
            ] + extra, env=env)

    for data_name in jobs:
        data_path = resolve_data_file(data_dir, data_name)
        idx_path = idx_path_for(data_path)
        if not os.path.isfile(data_path):
            print('SKIP missing:', data_path)
            continue
        if not os.path.isfile(idx_path):
            print('SKIP missing idx pair:', idx_path)
            continue

        base = os.path.basename(data_path)
        otype = infer_anomaly_kind_from_filename(base)
        out_csv = os.path.join(
            data_dir, 'scores_' + base.replace('outliers_data_', '').replace('.npy', '.csv'),
        )
        cmd = [
            py, os.path.join(root, 'run_loop.py'),
            '--mode', 'score',
            '--dataset', dataset,
            '--cluster_num', str(args.cluster_num),
            '--eval_data', eval_split,
            '--anomaly_path', data_path,
            '--output_scores', out_csv,
            '--batch_size', '128',
            '--gpu_id', args.gpu_id,
            '--otype', otype,
        ]
        print('[{}] {}'.format(dataset, base))
        if args.dry_run:
            print('  data:', data_path)
            print('  idx :', idx_path)
            print('  out :', out_csv)
            continue
        subprocess.check_call(cmd, env=env)
        print('Done:', out_csv)


if __name__ == '__main__':
    main()
