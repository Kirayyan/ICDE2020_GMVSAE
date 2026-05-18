import json
import os
import pickle
import numpy as np
from collections import defaultdict

from tf_compat import tf
from anomaly_io import load_anomaly_records, find_anomaly_source


class DataGenerator:
    def __init__(self, args):
        print("Loading data...")
        self.args = args
        self.map_size = args.map_size
        self.outlier_idx = []

        self.train_trajectories, self.train_sd, self.train_traj_num = self.build_dataset('train')
        self.val_trajectories, self.val_sd, self.val_traj_num = self.build_dataset('val')

        self.data_iterator = {
            'train': lambda: self.iterate_data(data_type='train'),
            'val': lambda: self.iterate_data(data_type='val'),
        }

    def build_dataset(self, data_type):
        args = self.args
        data_name = args.data_filename.split('.')
        data_name[-2] += "_{}".format(data_type)
        data_name = ".".join(data_name)
        trajectories = sorted(
            [eval(eachline) for eachline in open(data_name, 'r').readlines()],
            key=lambda k: len(k),
        )
        traj_num = len(trajectories)
        print("{} {} trajectories loading complete.".format(traj_num, data_type))

        traj_sd = defaultdict(list)
        for idx, traj in enumerate(trajectories):
            traj_sd[(traj[0], traj[-1])].append(idx)

        return trajectories, traj_sd, traj_num

    def _trajectories(self, data_type):
        if data_type == "train":
            return self.train_trajectories
        if data_type == "val":
            return self.val_trajectories
        raise ValueError("data_type must be 'train' or 'val'.")

    def inject_outliers(self, otype, data_type, ratio=0.05, level=2, point_prob=0.3, vary=False,
                        stay_len=8, speed_skip_ratio=0.5):
        trajectories = self._trajectories(data_type)
        traj_num = len(trajectories)

        self.outlier_idx = selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
        batch = [trajectories[idx] for idx in selected_idx]

        if otype == 'random':
            outliers = self.perturb_batch(batch, level=level, prob=point_prob)
        elif otype == 'shift':
            outliers = self.shift_batch(batch, level=level, prob=point_prob, vary=vary)
        elif otype == 'stay':
            outliers = self.stay_batch(batch, stay_len=stay_len)
        elif otype == 'speed':
            outliers = self.speed_batch(batch, skip_ratio=speed_skip_ratio)
        else:
            raise ValueError("otype must be one of: random, shift, stay, speed.")

        for i, idx in enumerate(selected_idx):
            trajectories[idx] = outliers[i]

        out_filename = 'porto_outliers_{}.pkl'.format(otype)
        with open(os.path.join('./data', out_filename), 'wb') as fp:
            pickle.dump(dict(zip(selected_idx.tolist(), outliers)), fp)

        print("{} {} outliers injected into {}.".format(len(outliers), otype, data_type))

    def load_outliers(self, filename, data_type):
        trajectories = self._trajectories(data_type)
        path = filename if os.path.isabs(filename) else os.path.join('./data', filename)
        with open(path, 'rb') as fp:
            idx_outliers = pickle.load(fp)
        for idx, traj in idx_outliers.items():
            trajectories[int(idx)] = traj
        self.outlier_idx = [int(i) for i in idx_outliers.keys()]

    def load_anomaly_manifest(self, manifest_path, data_type, anomaly_type='unknown'):
        """Load pre-built anomalies from JSON, pickle, npy pair, or auto-discover under mst_data_dir."""
        trajectories = self._trajectories(data_type)
        map_size = self.map_size
        time_interval = getattr(self.args, 'time_interval', 15)

        records = None
        resolved_from = manifest_path

        if manifest_path in ('', 'auto', 'AUTO'):
            mst_dir = getattr(self.args, 'mst_data_dir', './data')
            split = 'val' if data_type == 'val' else 'train'
            for atype in ('stay', 'speed'):
                src = find_anomaly_source(mst_dir, atype, split)
                if src and atype == anomaly_type:
                    records = load_anomaly_records(src, map_size, time_interval, atype)
                    resolved_from = str(src)
                    break
            if records is None:
                for atype in ('stay', 'speed'):
                    src = find_anomaly_source(mst_dir, atype, split)
                    if src:
                        records = load_anomaly_records(src, map_size, time_interval, atype)
                        resolved_from = str(src)
                        anomaly_type = atype
                        break
        else:
            path = manifest_path
            if not os.path.isabs(path) and not os.path.exists(path):
                candidates = [
                    path,
                    os.path.join('./data', path),
                    os.path.join('./data', os.path.basename(path)),
                ]
                mst_dir = getattr(self.args, 'mst_data_dir', None)
                if mst_dir:
                    candidates.append(os.path.join(mst_dir, os.path.basename(path)))
                    candidates.append(os.path.join(mst_dir, path))
                path = next((p for p in candidates if os.path.exists(p)), path)

            if os.path.isdir(path):
                split = 'val' if data_type == 'val' else 'train'
                src = find_anomaly_source(path, anomaly_type, split)
                if not src:
                    raise FileNotFoundError('No anomaly files found under {}'.format(path))
                records = load_anomaly_records(src, map_size, time_interval, anomaly_type)
                resolved_from = str(src)
            elif path.endswith('.npy') and 'outliers_data' in os.path.basename(path):
                idx_path = path.replace('outliers_data', 'outliers_idx', 1)
                if not os.path.isfile(idx_path):
                    raise FileNotFoundError(
                        'MST-OATD idx npy not found: {} (pair of {})'.format(idx_path, path)
                    )
                atype = anomaly_type if anomaly_type not in ('unknown', '') else 'full'
                records = load_anomaly_records(
                    ('npy', (path, idx_path)), map_size, time_interval, atype,
                )
                resolved_from = '{} + {}'.format(path, idx_path)
            elif path.endswith('.json'):
                records = load_anomaly_records(('json', path), map_size, time_interval, anomaly_type)
                resolved_from = path
            else:
                records = load_anomaly_records(('pkl', path), map_size, time_interval, anomaly_type)
                resolved_from = path

        meta = []
        for rec in records:
            tid = int(rec['tid'])
            trajectories[tid] = list(rec['trajectory'])
            meta.append({
                'tid': tid,
                'label': int(rec.get('label', 0)),
                'type': rec.get('type', rec.get('anomaly_type', anomaly_type)),
            })
            self.outlier_idx.append(tid)

        self.anomaly_meta = meta
        print("Loaded {} anomalous trajectories from {}.".format(len(meta), resolved_from))
        return meta

    def pad_and_mask(self, batch_x):
        max_len = max(len(x) for x in batch_x)
        batch_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_x]
        batch_x = [x + [0] * (max_len - len(x)) for x in batch_x]
        return batch_x, batch_mask

    def next_batch(self, batch_size, partial_ratio=1.0, sd=False):
        anchor_idx = np.random.randint(0, self.train_traj_num)
        shortest_idx = max(0, anchor_idx - batch_size * 2)
        longest_idx = min(self.train_traj_num, anchor_idx + batch_size * 2)
        batch_idx = np.random.randint(shortest_idx, longest_idx, size=batch_size)
        batch_trajectories = []
        for tid in batch_idx:
            partial = int(len(self.train_trajectories[tid]) * partial_ratio)
            batch_trajectories.append(self.train_trajectories[tid][:partial])
        batch_seq_length = [len(traj) for traj in batch_trajectories]
        batch_x, batch_mask = self.pad_and_mask(batch_trajectories)
        return [tf.convert_to_tensor(x) for x in [batch_x, batch_mask, batch_seq_length]]

    def iterate_data(self, data_type='val', partial_ratio=1.0, sd=False):
        batch_size = self.args.batch_size

        if data_type == 'train':
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories + self.train_trajectories[:batch_size]
        elif data_type == 'val':
            traj_num = self.val_traj_num
            trajectories = self.val_trajectories + self.val_trajectories[:batch_size]
        else:
            raise ValueError("data_type must be 'train' or 'val'.")

        for shortest_idx in range(0, traj_num, batch_size):
            longest_idx = shortest_idx + batch_size
            batch_trajectories = []
            for tid in range(shortest_idx, longest_idx):
                partial = int(len(trajectories[tid]) * partial_ratio)
                batch_trajectories.append(trajectories[tid][:partial])
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_x, batch_mask = self.pad_and_mask(batch_trajectories)
            yield batch_x, batch_mask, batch_seq_length

    def _perturb_point(self, point, level, offset=None):
        map_size = self.map_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)

    def perturb_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        for traj in batch_x:
            noisy_batch_x.append(
                [traj[0]]
                + [
                    self._perturb_point(p, level) if p != 0 and np.random.random() < prob else p
                    for p in traj[1:-1]
                ]
                + [traj[-1]]
            )
        return noisy_batch_x

    def shift_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = max(1, int((len(traj) - 2) * prob))
            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [
                int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1]),
            ]
            div0 = abs(offset[0]) if offset[0] != 0 else 1
            div1 = abs(offset[1]) if offset[1] != 0 else 1

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(
                traj[:anomaly_st_loc]
                + [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]]
                + traj[anomaly_ed_loc:]
            )
        return noisy_batch_x

    def stay_batch(self, batch_x, stay_len=8):
        """Stay anomaly: repeat one grid cell for an unusually long dwell."""
        noisy_batch_x = []
        for traj in batch_x:
            if len(traj) < 4:
                noisy_batch_x.append(traj)
                continue
            stay_len_eff = min(stay_len, len(traj) - 2)
            st = np.random.randint(1, len(traj) - stay_len_eff - 1)
            cell = traj[st]
            noisy_batch_x.append(traj[:st] + [cell] * stay_len_eff + traj[st + 1:])
        return noisy_batch_x

    def speed_batch(self, batch_x, skip_ratio=0.5):
        """Speed anomaly: collapse a middle segment to simulate unrealistic fast movement."""
        noisy_batch_x = []
        for traj in batch_x:
            if len(traj) < 6:
                noisy_batch_x.append(traj)
                continue
            seg_len = max(3, int((len(traj) - 2) * skip_ratio))
            st = np.random.randint(1, len(traj) - seg_len - 1)
            ed = st + seg_len
            # Keep endpoints of the segment only (skip intermediate cells).
            noisy_batch_x.append(traj[:st] + [traj[st], traj[ed - 1]] + traj[ed:])
        return noisy_batch_x
