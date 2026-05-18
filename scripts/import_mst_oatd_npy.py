#!/usr/bin/env python3
"""Import MST-OATD outliers_*.npy (official repo output) into GMVSAE JSON manifest."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import DATASETS
from mst_oatd_outliers import grid_only_to_gmvsae_stay, traj_to_grid_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['porto', 'cd'], required=True)
    parser.add_argument('--traj_npy', required=True, help='outliers_data_*.npy from MST-OATD')
    parser.add_argument('--idx_npy', required=True, help='outliers_idx_*.npy from MST-OATD')
    parser.add_argument('--anomaly_type', choices=['stay', 'speed', 'full'], default='full')
    parser.add_argument('--split', default='val')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    trajs = np.load(args.traj_npy, allow_pickle=True)
    idxs = np.load(args.idx_npy, allow_pickle=True)

    records = []
    for idx in idxs:
        st = trajs[int(idx)]
        if args.anomaly_type == 'stay':
            grid = grid_only_to_gmvsae_stay(st, cfg['map_size'], cfg['time_interval'])
        else:
            grid = traj_to_grid_only(st)
        records.append({'tid': int(idx), 'trajectory': grid, 'label': 0, 'type': args.anomaly_type})

    out = args.output or os.path.join(
        cfg['mst_data_dir'], 'anomalies_{}_{}.json'.format(args.anomaly_type, args.split),
    )
    with open(out, 'w') as fp:
        json.dump({'records': records, 'dataset': args.dataset}, fp)
    print('Wrote', len(records), 'records to', out)


if __name__ == '__main__':
    main()
