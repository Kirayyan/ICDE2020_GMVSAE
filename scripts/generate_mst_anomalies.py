#!/usr/bin/env python3
"""
Generate stay / speed anomalies with MST-OATD rules for Porto and Chengdu.

Outputs GMVSAE-ready JSON manifests under data/<dataset>/anomalies_<type>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import DATASETS
from mst_oatd_outliers import (
    generate_outliers,
    grid_only_to_gmvsae_stay,
    traj_to_grid_only,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['porto', 'cd'], required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--anomaly_type', choices=['stay', 'speed', 'both'], default='both')
    parser.add_argument('--distance', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--observed_ratio', type=float, default=1.0)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mst_dir', default='')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    mst_dir = args.mst_dir or cfg['mst_data_dir']
    npy_name = 'train_data_init.npy' if args.split == 'train' else 'test_data_init.npy'
    trajs = np.load(os.path.join(mst_dir, npy_name), allow_pickle=True).tolist()

    types = ['stay', 'speed'] if args.anomaly_type == 'both' else [args.anomaly_type]
    os.makedirs(mst_dir, exist_ok=True)

    for atype in types:
        corrupted, selected = generate_outliers(
            trajs,
            map_size=cfg['map_size'],
            time_interval=cfg['time_interval'],
            ratio=args.ratio,
            distance=args.distance,
            fraction=args.fraction,
            observed_ratio=args.observed_ratio,
            anomaly_type=atype,
            seed=args.seed,
        )
        records = []
        for idx in selected:
            st_traj = corrupted[idx]
            if atype == 'stay':
                grid_traj = grid_only_to_gmvsae_stay(st_traj, cfg['map_size'], cfg['time_interval'])
            else:
                grid_traj = traj_to_grid_only(st_traj)
            records.append({
                'tid': int(idx),
                'trajectory': grid_traj,
                'label': 0,
                'type': atype,
            })

        out_json = os.path.join(mst_dir, 'anomalies_{}_{}.json'.format(atype, args.split))
        with open(out_json, 'w') as fp:
            json.dump({'records': records, 'dataset': args.dataset, 'split': args.split}, fp)

        out_traj = os.path.join(
            mst_dir,
            'outliers_data_{}_{}_{}_{}.npy'.format(atype, args.distance, args.fraction, args.observed_ratio),
        )
        out_idx = os.path.join(
            mst_dir,
            'outliers_idx_{}_{}_{}_{}.npy'.format(atype, args.distance, args.fraction, args.observed_ratio),
        )
        np.save(out_traj, np.array(corrupted, dtype=object))
        np.save(out_idx, np.array(selected))

        print('{}: {} anomalies -> {}'.format(args.dataset, atype, out_json))


if __name__ == '__main__':
    main()
