#!/usr/bin/env python3
"""
Score pre-built stay/speed anomalies (no regeneration).

Place your files under data/porto/ and data/cd/, for example:
  - anomalies_stay_val.json / anomalies_speed_val.json
  - or outliers_data_*_2_0.2_1.0.npy + outliers_idx_*_2_0.2_1.0.npy

Requires trained checkpoints in ckpt/porto/ and ckpt/cd/ (run pretrain+train once per dataset).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_io import find_anomaly_source
from dataset_config import DATASETS


def resolve_path(dataset, anomaly_type, split, user_path):
    if user_path:
        return user_path
    cfg = DATASETS[dataset]
    src = find_anomaly_source(cfg['mst_data_dir'], anomaly_type, split)
    if not src:
        return None
    if src[0] == 'json':
        return src[1]
    if src[0] == 'npy':
        return src[1][0]
    return src[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['porto', 'cd'])
    parser.add_argument('--anomaly_types', nargs='+', default=['stay', 'speed'])
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--train_if_missing', action='store_true',
                        help='train checkpoints when missing (default: score only)')
    parser.add_argument('--stay_path', default='', help='override stay manifest for all datasets')
    parser.add_argument('--speed_path', default='', help='override speed manifest for all datasets')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    py = sys.executable
    type_paths = {'stay': args.stay_path, 'speed': args.speed_path}

    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        ckpt = os.path.join(cfg['ckpt_root'], 'gmvsae_32_256_5')
        if args.train_if_missing and not os.path.exists(os.path.join(ckpt, 'checkpoint')):
            for mode, extra in [
                ('pretrain', ['--num_epochs', '2']),
                ('train', ['--num_epochs', '2', '--pretrain_dir', cfg['pretrain_root']]),
            ]:
                subprocess.check_call([
                    py, os.path.join(root, 'run_loop.py'), '--mode', mode,
                    '--dataset', dataset, '--cluster_num', str(args.cluster_num),
                    '--batch_size', '64', '--log_steps', '5',
                ] + extra, env=env)

        for atype in args.anomaly_types:
            manifest = resolve_path(dataset, atype, args.split, type_paths.get(atype, ''))
            if not manifest:
                print('SKIP {} {}: no anomaly file under {}'.format(
                    dataset, atype, cfg['mst_data_dir']))
                continue
            out_csv = os.path.join(
                cfg['mst_data_dir'], 'scores_{}_{}.csv'.format(atype, args.split),
            )
            eval_split = args.split
            cmd = [
                py, os.path.join(root, 'run_loop.py'),
                '--mode', 'score',
                '--dataset', dataset,
                '--cluster_num', str(args.cluster_num),
                '--eval_data', eval_split,
                '--anomaly_path', manifest,
                '--output_scores', out_csv,
                '--batch_size', '64',
            ]
            print('>>', ' '.join(cmd))
            subprocess.check_call(cmd, env=env)
            print('Done:', out_csv)


if __name__ == '__main__':
    main()
