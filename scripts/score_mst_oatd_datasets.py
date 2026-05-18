#!/usr/bin/env python3
"""Score stay and speed anomalies on Porto and Chengdu with trained GMVSAE checkpoints."""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import DATASETS


def run_cmd(cmd, env):
    print('>>', ' '.join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['porto', 'cd'])
    parser.add_argument('--anomaly_types', nargs='+', default=['stay', 'speed'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    py = sys.executable

    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        if not args.skip_train:
            for mode, extra in [('pretrain', ['--num_epochs', '2']), ('train', ['--num_epochs', '2', '--pretrain_dir', cfg['pretrain_root']])]:
                run_cmd([
                    py, os.path.join(root, 'run_loop.py'),
                    '--mode', mode,
                    '--dataset', dataset,
                    '--cluster_num', str(args.cluster_num),
                    '--batch_size', '64',
                    '--log_steps', '5',
                ] + extra, env)

        for atype in args.anomaly_types:
            manifest = os.path.join(
                cfg['mst_data_dir'],
                'anomalies_{}_{}.json'.format(atype, args.split),
            )
            if not os.path.exists(manifest):
                raise FileNotFoundError(
                    'Missing {} — run scripts/generate_mst_anomalies.py --dataset {} first'.format(
                        manifest, dataset,
                    )
                )
            out_csv = os.path.join(
                cfg['mst_data_dir'],
                'scores_{}_{}.csv'.format(atype, args.split),
            )
            eval_split = 'val' if args.split == 'val' else 'train'
            run_cmd([
                py, os.path.join(root, 'run_loop.py'),
                '--mode', 'score',
                '--dataset', dataset,
                '--cluster_num', str(args.cluster_num),
                '--eval_data', eval_split,
                '--anomaly_path', manifest,
                '--output_scores', out_csv,
                '--batch_size', '64',
            ], env)


if __name__ == '__main__':
    main()
