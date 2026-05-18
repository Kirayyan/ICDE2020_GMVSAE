#!/usr/bin/env python3
"""Score MST-OATD official outliers_*.npy on Porto and Chengdu."""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_io import official_npy_paths
from dataset_config import DATASETS


def main():
    parser = argparse.ArgumentParser(description='GMVSAE scoring for MST-OATD official npy')
    parser.add_argument('--datasets', nargs='+', default=['porto', 'cd'])
    parser.add_argument('--distance', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--observed_ratio', type=float, default=1.0)
    parser.add_argument('--month', default='',
                        help='evolving split tag, e.g. 3; empty uses init (test_data_init)')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--train_if_missing', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    py = sys.executable
    month = int(args.month) if args.month not in ('', 'init') else None

    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        data_dir = cfg['mst_data_dir']

        train_npy = os.path.join(data_dir, 'train_data_init.npy')
        test_npy = os.path.join(data_dir, 'test_data_init.npy')
        if not os.path.isfile(cfg['data_prefix'] + '_val.csv'):
            if os.path.isfile(train_npy) and os.path.isfile(test_npy):
                subprocess.check_call([
                    py, os.path.join(root, 'scripts/convert_mst_npy_to_gmvsae.py'),
                    '--dataset', dataset,
                ], env=env)
            else:
                raise FileNotFoundError(
                    'Need {} and {} or processed csv under {}'.format(train_npy, test_npy, data_dir)
                )

        traj_npy, idx_npy = official_npy_paths(
            data_dir, args.distance, args.fraction, args.observed_ratio, month=month,
        )
        if not traj_npy:
            raise FileNotFoundError(
                'Official npy not found in {} (expected outliers_data_init_{}_{}_{}.npy)'.format(
                    data_dir, args.distance, args.fraction, args.observed_ratio,
                )
            )
        print('[{}] using {} + {}'.format(dataset, os.path.basename(traj_npy), os.path.basename(idx_npy)))

        ckpt = os.path.join(cfg['ckpt_root'], 'gmvsae_32_256_5')
        if args.train_if_missing or not os.path.exists(os.path.join(ckpt, 'checkpoint')):
            for mode, extra in [
                ('pretrain', ['--num_epochs', str(args.num_epochs)]),
                ('train', [
                    '--num_epochs', str(args.num_epochs),
                    '--pretrain_dir', cfg['pretrain_root'],
                ]),
            ]:
                subprocess.check_call([
                    py, os.path.join(root, 'run_loop.py'), '--mode', mode,
                    '--dataset', dataset, '--cluster_num', str(args.cluster_num),
                    '--batch_size', '128', '--log_steps', '10', '--gpu_id', args.gpu_id,
                ] + extra, env=env)

        tag = 'init' if month is None else str(month)
        out_csv = os.path.join(
            data_dir,
            'scores_official_{}_{}_{}_{}.csv'.format(
                tag, args.distance, args.fraction, args.observed_ratio,
            ),
        )
        cmd = [
            py, os.path.join(root, 'run_loop.py'),
            '--mode', 'score',
            '--dataset', dataset,
            '--cluster_num', str(args.cluster_num),
            '--eval_data', args.split,
            '--anomaly_path', traj_npy,
            '--output_scores', out_csv,
            '--batch_size', '128',
            '--gpu_id', args.gpu_id,
            '--otype', 'full',
        ]
        print('>>', ' '.join(cmd))
        subprocess.check_call(cmd, env=env)
        print('Done:', out_csv)


if __name__ == '__main__':
    main()
