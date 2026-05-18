#!/usr/bin/env python3
"""
Score custom MST-OATD npy pairs by filename pattern.

Example filenames (your Porto folder):
  outliers_data_init_stay_accelerate4_0p1_1.npy
  outliers_idx_init_stay_accelerate4_0p1_1.npy
  outliers_data_10_stay_accelerate3_0p1_1.npy
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_io import find_npy_pairs, infer_anomaly_kind_from_filename
from dataset_config import DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['porto', 'cd'])
    parser.add_argument('--pattern', default='_stay_',
                        help='substring matched in outliers_data_*.npy filename')
    parser.add_argument('--patterns', nargs='+', default=[],
                        help='score multiple patterns in one run, e.g. _stay_ speed_accelerate detour')
    parser.add_argument('--split', default='',
                        help='init or 1-10; empty = score all matching files')
    parser.add_argument('--eval_split', default='val', choices=['train', 'val'])
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--train_if_missing', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--list_only', action='store_true')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    py = sys.executable
    split_tag = args.split if args.split else None
    pattern_list = args.patterns if args.patterns else [args.pattern]

    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        data_dir = cfg['mst_data_dir']

        if not os.path.isfile(cfg['data_prefix'] + '_val.csv'):
            train_npy = os.path.join(data_dir, 'train_data_init.npy')
            test_npy = os.path.join(data_dir, 'test_data_init.npy')
            if os.path.isfile(train_npy) and os.path.isfile(test_npy):
                subprocess.check_call([
                    py, os.path.join(root, 'scripts/convert_mst_npy_to_gmvsae.py'),
                    '--dataset', dataset,
                ], env=env)
            else:
                raise FileNotFoundError('Missing csv and train/test npy under ' + data_dir)

        ckpt = os.path.join(cfg['ckpt_root'], 'gmvsae_32_256_5')
        if args.train_if_missing and not os.path.exists(os.path.join(ckpt, 'checkpoint')):
            for mode, extra in [
                ('pretrain', ['--num_epochs', str(args.num_epochs)]),
                ('train', ['--num_epochs', str(args.num_epochs), '--pretrain_dir', cfg['pretrain_root']]),
            ]:
                subprocess.check_call([
                    py, os.path.join(root, 'run_loop.py'), '--mode', mode,
                    '--dataset', dataset, '--cluster_num', str(args.cluster_num),
                    '--batch_size', '128', '--gpu_id', args.gpu_id,
                ] + extra, env=env)

        for pattern in pattern_list:
            pairs = find_npy_pairs(data_dir, pattern, split_tag=split_tag)
            if not pairs:
                print('[{}] no files for pattern="{}"'.format(dataset, pattern))
                continue

            if args.list_only:
                for traj, idx, base in pairs:
                    print(dataset, pattern, base)
                continue

            for traj_npy, idx_npy, base in pairs:
                otype = infer_anomaly_kind_from_filename(base)
                out_csv = os.path.join(
                    data_dir, 'scores_' + base.replace('outliers_data_', '').replace('.npy', '.csv'),
                )
                cmd = [
                    py, os.path.join(root, 'run_loop.py'),
                    '--mode', 'score',
                    '--dataset', dataset,
                    '--cluster_num', str(args.cluster_num),
                    '--eval_data', args.eval_split,
                    '--anomaly_path', traj_npy,
                    '--output_scores', out_csv,
                    '--batch_size', '128',
                    '--gpu_id', args.gpu_id,
                    '--otype', otype,
                ]
                print('[{}][{}] {} + {}'.format(
                    dataset, pattern, os.path.basename(traj_npy), os.path.basename(idx_npy),
                ))
                print('>>', ' '.join(cmd))
                subprocess.check_call(cmd, env=env)
                print('Done:', out_csv)


if __name__ == '__main__':
    main()
