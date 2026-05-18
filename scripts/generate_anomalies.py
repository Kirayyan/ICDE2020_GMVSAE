#!/usr/bin/env python3
"""Generate stay/speed anomaly manifests for GMVSAE scoring."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import DataGenerator


class _Args:
    def __init__(self, data_filename, map_size):
        self.data_filename = data_filename
        self.map_size = map_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', default='./data/processed_porto.csv')
    parser.add_argument('--map_size', type=int, nargs=2, default=[51, 158])
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='fraction of trajectories to corrupt per type')
    parser.add_argument('--stay_len', type=int, default=8)
    parser.add_argument('--speed_skip_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='./data/anomalies_stay_speed.json')
    args = parser.parse_args()

    np.random.seed(args.seed)
    gen = DataGenerator(_Args(args.data_filename, tuple(args.map_size)))
    trajectories = gen.train_trajectories if args.split == 'train' else gen.val_trajectories
    traj_num = len(trajectories)

    n_each = max(1, int(traj_num * args.ratio))
    stay_idx = np.random.choice(traj_num, n_each, replace=False)
    remain = np.setdiff1d(np.arange(traj_num), stay_idx)
    speed_idx = np.random.choice(remain, min(n_each, len(remain)), replace=False)

    records = []
    for idx in stay_idx:
        corrupted = gen.stay_batch([trajectories[idx]], stay_len=args.stay_len)[0]
        records.append({
            'tid': int(idx),
            'trajectory': corrupted,
            'label': 0,
            'type': 'stay',
        })
    for idx in speed_idx:
        corrupted = gen.speed_batch([trajectories[idx]], skip_ratio=args.speed_skip_ratio)[0]
        records.append({
            'tid': int(idx),
            'trajectory': corrupted,
            'label': 0,
            'type': 'speed',
        })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as fp:
        json.dump({'records': records}, fp)

    print('Wrote {} anomalies (stay={}, speed={}) to {}'.format(
        len(records), len(stay_idx), len(speed_idx), args.output))


if __name__ == '__main__':
    main()
