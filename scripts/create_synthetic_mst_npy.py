#!/usr/bin/env python3
"""Create minimal MST-OATD-style npy files for Porto/CD smoke tests."""
import argparse
import os
import random

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import DATASETS


def fake_traj(length, map_size):
    h, w = map_size
    x, y = random.randint(0, h - 1), random.randint(0, w - 1)
    seq = []
    t = [8, 0, 0, 2013, 7, 1]
    for _ in range(length):
        seq.append([x * w + y, list(t)])
        t[2] = (t[2] + 15) % 60
        if t[2] == 0:
            t[1] = (t[1] + 1) % 60
        dx, dy = random.choice([(0, 1), (1, 0), (-1, 0), (0, -1)])
        x = min(max(x + dx, 0), h - 1)
        y = min(max(y + dy, 0), w - 1)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['porto', 'cd'])
    args = parser.parse_args()
    random.seed(1234)

    for name in args.datasets:
        cfg = DATASETS[name]
        os.makedirs(cfg['mst_data_dir'], exist_ok=True)
        train = [fake_traj(random.randint(25, 40), cfg['map_size']) for _ in range(400)]
        test = [fake_traj(random.randint(25, 40), cfg['map_size']) for _ in range(100)]
        np.save(os.path.join(cfg['mst_data_dir'], 'train_data_init.npy'), np.array(train, dtype=object))
        np.save(os.path.join(cfg['mst_data_dir'], 'test_data_init.npy'), np.array(test, dtype=object))
        print('Synthetic MST npy for {}: train={}, test={}'.format(name, len(train), len(test)))


if __name__ == '__main__':
    main()
