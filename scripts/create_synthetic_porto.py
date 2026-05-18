#!/usr/bin/env python3
"""Create a tiny synthetic Porto-style dataset for smoke testing the pipeline."""
import os
import random


def make_traj(length, grid_w, grid_h):
    x, y = random.randint(0, grid_h - 1), random.randint(0, grid_w - 1)
    traj = [x * grid_w + y]
    for _ in range(length - 1):
        dx, dy = random.choice([(0, 1), (1, 0), (-1, 0), (0, -1)])
        x = min(max(x + dx, 0), grid_h - 1)
        y = min(max(y + dy, 0), grid_w - 1)
        traj.append(x * grid_w + y)
    return traj


def write_split(path, n_trajs, grid_size):
    grid_h, grid_w = grid_size
    with open(path, 'w') as fp:
        for _ in range(n_trajs):
            traj = make_traj(random.randint(25, 60), grid_w, grid_h)
            fp.write(str(traj) + '\n')


def main():
    os.makedirs('./data', exist_ok=True)
    grid_size = (51, 158)
    # Enough trajectories per OD pair for preprocess_sd rules.
    write_split('./data/processed_porto_train.csv', 800, grid_size)
    write_split('./data/processed_porto_val.csv', 200, grid_size)
    print('Synthetic data written to ./data/processed_porto_{train,val}.csv')


if __name__ == '__main__':
    main()
