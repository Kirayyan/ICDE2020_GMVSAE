"""
MST-OATD-compatible outlier generation (spatial + temporal).

Reference: https://github.com/chwang0721/MST-OATD generate_outliers.py

For GMVSAE (spatial-only) we expose two anomaly types used with MST-OATD benchmarks:
  - stay: temporal offset on a segment (grid cells unchanged, timestamps stretched)
  - speed: spatial offset on a segment (grid cells perturbed, timestamps unchanged)

The combined MST-OATD default (spatial + temporal) is available as anomaly_type='full'.
"""
import datetime
import math
from datetime import timedelta

import numpy as np


def _convert(point, map_size):
    x = int(point // map_size[1])
    y = int(point % map_size[1])
    return [x, y]


def _distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _time_calculate(vec, seconds):
    a = datetime.datetime(vec[3], vec[4], vec[5], vec[0], vec[1], vec[2])
    t = a + timedelta(seconds=seconds)
    return [t.hour, t.minute, t.second, t.year, t.month, t.day]


def perturb_point(point, level, map_size, offset=None):
    grid_id, times = point[0], point[1]
    x, y = int(grid_id // map_size[1]), int(grid_id % map_size[1])
    if offset is None:
        offset = [
            [0, 1], [1, 0], [-1, 0], [0, -1],
            [1, 1], [-1, -1], [-1, 1], [1, -1],
        ]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
    else:
        x_offset, y_offset = offset
    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
        x += x_offset * level
        y += y_offset * level
    return [int(x * map_size[1] + y), times]


def perturb_time(traj, st_loc, end_loc, time_offset, interval):
    for i in range(st_loc, end_loc):
        traj[i][1] = _time_calculate(traj[i][1], int((i - st_loc + 1) * time_offset * interval))
    for i in range(end_loc, len(traj)):
        traj[i][1] = _time_calculate(traj[i][1], int((end_loc - st_loc) * time_offset * interval))
    return traj


def perturb_batch(trajs, map_size, level, prob, selected_idx, time_interval,
                  observed_ratio, anomaly_type):
    noisy = []
    for idx, traj in enumerate(trajs):
        traj = [list(p) for p in traj]
        anomaly_len = int(len(traj) * prob)
        if anomaly_len < 1 or len(traj) <= anomaly_len + 2:
            noisy.append(traj[: int(len(traj) * observed_ratio)])
            continue
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
        if idx in selected_idx:
            anomaly_ed_loc = anomaly_st_loc + anomaly_len
            if anomaly_type == 'stay':
                segment = [list(p) for p in traj[anomaly_st_loc:anomaly_ed_loc]]
                p_traj = traj[:anomaly_st_loc] + segment + traj[anomaly_ed_loc:]
                dis = max(
                    _distance(
                        _convert(traj[anomaly_st_loc][0], map_size),
                        _convert(traj[anomaly_ed_loc - 1][0], map_size),
                    ),
                    1,
                )
                time_offset = (level * 2) / dis
                p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, time_interval)
            elif anomaly_type == 'speed':
                p_traj = (
                    traj[:anomaly_st_loc]
                    + [perturb_point(p, level, map_size) for p in traj[anomaly_st_loc:anomaly_ed_loc]]
                    + traj[anomaly_ed_loc:]
                )
            else:  # full — original MST-OATD
                p_traj = (
                    traj[:anomaly_st_loc]
                    + [perturb_point(p, level, map_size) for p in traj[anomaly_st_loc:anomaly_ed_loc]]
                    + traj[anomaly_ed_loc:]
                )
                dis = max(
                    _distance(
                        _convert(traj[anomaly_st_loc][0], map_size),
                        _convert(traj[anomaly_ed_loc][0], map_size),
                    ),
                    1,
                )
                time_offset = (level * 2) / dis
                p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, time_interval)
        else:
            p_traj = traj
        noisy.append(p_traj[: int(len(p_traj) * observed_ratio)])
    return noisy


def generate_outliers(trajs, map_size, time_interval, ratio=0.05, distance=2,
                      fraction=0.2, observed_ratio=1.0, anomaly_type='full', seed=1234):
    np.random.seed(seed)
    traj_num = len(trajs)
    selected_idx = set(np.random.randint(0, traj_num, size=int(traj_num * ratio)))
    new_trajs = perturb_batch(
        trajs, map_size, distance, fraction, selected_idx, time_interval,
        observed_ratio, anomaly_type,
    )
    return new_trajs, sorted(selected_idx)


def traj_to_grid_only(traj):
    return [int(p[0]) for p in traj]


def grid_only_to_gmvsae_stay(traj_spatiotemporal, map_size, time_interval=15):
    """
    Map temporal stay anomaly to a spatial sequence for GMVSAE:
    duplicate grid cells proportional to timestamp gaps in the anomalous segment.
    """
    if len(traj_spatiotemporal) < 2:
        return traj_to_grid_only(traj_spatiotemporal)

    def t_seconds(t):
        return t[0] * 3600 + t[1] * 60 + t[2]

    out = [int(traj_spatiotemporal[0][0])]
    for i in range(1, len(traj_spatiotemporal)):
        gid = int(traj_spatiotemporal[i][0])
        prev_t = t_seconds(traj_spatiotemporal[i - 1][1])
        cur_t = t_seconds(traj_spatiotemporal[i][1])
        dt = cur_t - prev_t
        if dt < 0:
            dt += 24 * 3600
        repeats = max(1, int(round(dt / float(time_interval))))
        if traj_spatiotemporal[i][0] == traj_spatiotemporal[i - 1][0]:
            repeats = max(repeats, 2)
        out.extend([gid] * repeats)
    return out
