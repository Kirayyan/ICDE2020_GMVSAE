#!/usr/bin/env python3
"""Convert MST-OATD .npy trajectories to GMVSAE csv splits."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import DATASETS
from mst_oatd_outliers import traj_to_grid_only


def write_csv(path, trajs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fp:
        for traj in trajs:
            fp.write(str(traj_to_grid_only(traj)) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['porto', 'cd'], required=True)
    parser.add_argument('--mst_dir', default='')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    mst_dir = args.mst_dir or cfg['mst_data_dir']
    prefix = cfg['data_prefix']

    train_path = os.path.join(mst_dir, 'train_data_init.npy')
    test_path = os.path.join(mst_dir, 'test_data_init.npy')
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)

    write_csv(prefix + '_train.csv', train)
    write_csv(prefix + '_val.csv', test)
    print('Wrote {} train / {} val trajectories for {}'.format(len(train), len(test), args.dataset))


if __name__ == '__main__':
    main()
