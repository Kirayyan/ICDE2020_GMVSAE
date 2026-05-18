"""Load user-provided anomaly samples (JSON / pickle / MST-OATD npy)."""
import glob
import json
import os
import pickle

import numpy as np

from mst_oatd_outliers import grid_only_to_gmvsae_stay, traj_to_grid_only


def _is_spatiotemporal(traj):
    if not traj:
        return False
    first = traj[0]
    return isinstance(first, (list, tuple)) and len(first) >= 2 and not isinstance(first[0], (list, tuple))


def normalize_trajectory(traj, map_size, time_interval, anomaly_type):
    if _is_spatiotemporal(traj):
        if anomaly_type == 'stay':
            return grid_only_to_gmvsae_stay(traj, map_size, time_interval)
        return traj_to_grid_only(traj)
    return [int(p[0]) if isinstance(p, (list, tuple)) else int(p) for p in traj]


def _records_from_npy_pair(traj_npy, idx_npy, map_size, time_interval, anomaly_type):
    trajs = np.load(traj_npy, allow_pickle=True)
    idxs = np.load(idx_npy, allow_pickle=True)
    records = []
    for idx in idxs:
        st = trajs[int(idx)]
        records.append({
            'tid': int(idx),
            'trajectory': normalize_trajectory(st, map_size, time_interval, anomaly_type),
            'label': 0,
            'type': anomaly_type,
        })
    return records


def find_anomaly_source(data_dir, anomaly_type, split='val'):
    """Locate user anomaly files under data_dir. Returns (kind, path_or_tuple)."""
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        return None

    json_names = [
        'anomalies_{}_{}.json'.format(anomaly_type, split),
        'anomalies_{}.json'.format(anomaly_type),
        'anomaly_{}_{}.json'.format(anomaly_type, split),
    ]
    for name in json_names:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return ('json', path)

    for path in sorted(glob.glob(os.path.join(data_dir, '*{}*.json'.format(anomaly_type)))):
        if split in os.path.basename(path) or 'anomal' in os.path.basename(path).lower():
            return ('json', path)

  # MST-OATD npy pairs: outliers_data* + outliers_idx* containing anomaly_type
    traj_candidates = sorted(glob.glob(os.path.join(data_dir, 'outliers_data*.npy')))
    for traj_npy in traj_candidates:
        base = os.path.basename(traj_npy)
        if anomaly_type not in base and 'outliers_data' in base:
            # generic MST file — only match if no type-specific file exists
            pass
        elif anomaly_type not in base:
            continue
        idx_npy = traj_npy.replace('outliers_data', 'outliers_idx', 1)
        if os.path.isfile(idx_npy):
            return ('npy', (traj_npy, idx_npy))

    for traj_npy in sorted(glob.glob(os.path.join(data_dir, 'outliers_data*.npy'))):
        idx_npy = traj_npy.replace('outliers_data', 'outliers_idx', 1)
        if os.path.isfile(idx_npy) and anomaly_type == 'full':
            return ('npy', (traj_npy, idx_npy))

    pkl_path = os.path.join(data_dir, 'anomalies_{}.pkl'.format(anomaly_type))
    if os.path.isfile(pkl_path):
        return ('pkl', pkl_path)

    return None


def load_anomaly_records(source, map_size, time_interval, anomaly_type):
    """Load records list from resolved source."""
    kind, payload = source
    if kind == 'json':
        with open(payload, 'r') as fp:
            data = json.load(fp)
        records = data.get('records', data)
    elif kind == 'pkl':
        with open(payload, 'rb') as fp:
            data = pickle.load(fp)
        if isinstance(data, dict) and 'records' in data:
            records = data['records']
        elif isinstance(data, dict):
            records = [
                {'tid': int(k), 'trajectory': v, 'label': 0, 'type': anomaly_type}
                for k, v in data.items()
            ]
        else:
            records = data
    elif kind == 'npy':
        records = _records_from_npy_pair(payload[0], payload[1], map_size, time_interval, anomaly_type)
    else:
        raise ValueError('Unknown source kind: {}'.format(kind))

    for rec in records:
        atype = rec.get('type', rec.get('anomaly_type', anomaly_type))
        rec['trajectory'] = normalize_trajectory(rec['trajectory'], map_size, time_interval, atype)
        rec['type'] = atype
        rec['label'] = int(rec.get('label', 0))
        rec['tid'] = int(rec['tid'])
    return records
