import csv
import math
import os
import argparse
import numpy as np
from tf_compat import tf

from data_generator import DataGenerator
from model import Model
from dataset_config import apply_dataset_args
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

optimizers = {
    'sgd': lambda lr: tf.train.MomentumOptimizer(lr, 0.0),
    'momentum': lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
    'adagrad': lambda lr: tf.train.AdagradOptimizer(lr),
    'adam': lambda lr: tf.train.AdamOptimizer(lr),
}


def define_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                        help='pretrain, train, eval, or score')
    parser.add_argument('--dataset', type=str, default='',
                        help='porto or cd — sets map_size, data paths, and ckpt dirs (MST-OATD aligned)')
    parser.add_argument('--data_filename', type=str, default="./data/processed_porto.csv",
                        help='processed trajectory file prefix')
    parser.add_argument('--map_size', type=int, nargs=2, default=None,
                        help='grid map height and width (auto when --dataset is set)')
    parser.add_argument('--model', type=str, default="gmvsae",
                        help='model name tag for checkpoint paths')
    parser.add_argument('--token_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--rnn_dim', type=int, default=256,
                        help='RNN hidden size')
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='Gaussian mixture components')
    parser.add_argument('--model_dir', type=str, default="ckpt",
                        help='checkpoint directory')
    parser.add_argument('--pretrain_dir', type=str, default="",
                        help='pretrain checkpoint directory (for init_mu_c)')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='logging interval in steps')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='training epochs')
    parser.add_argument('--num_negs', type=int, default=64,
                        help='negative samples for softmax')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='fraction of each trajectory used at eval/score time')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer name')
    parser.add_argument('--gpu_id', type=str, default="0")

    # evaluation / scoring
    parser.add_argument('--eval_data', type=str, default='train',
                        help='train or val split for eval/score')
    parser.add_argument('--otype', type=str, default='random',
                        help='random, shift, stay, speed (built-in injection)')
    parser.add_argument('--outlier_ratio', type=float, default=0.05,
                        help='fraction of trajectories to corrupt when injecting')
    parser.add_argument('--anomaly_path', type=str, default='',
                        help='user anomaly manifest (.json/.pkl); overrides injection')
    parser.add_argument('--output_scores', type=str, default='./data/anomaly_scores.csv',
                        help='CSV path for per-trajectory scores (score mode)')
    parser.add_argument('--stay_len', type=int, default=8,
                        help='repeated cells for stay anomaly injection')
    parser.add_argument('--speed_skip_ratio', type=float, default=0.5,
                        help='segment length ratio for speed anomaly injection')

    args = parser.parse_args()
    if args.map_size is None:
        args.map_size = (51, 158)
    else:
        args.map_size = tuple(args.map_size)
    args = apply_dataset_args(args)
    return args


def checkpoint_dir(args):
    return '{}/{}_{}_{}_{}'.format(
        args.model_dir, args.model, args.token_dim, args.rnn_dim, args.cluster_num)


def run_train(model, args, master='', is_chief=True):
    sampler = DataGenerator(args)
    batch_size = args.batch_size
    batch_data = sampler.next_batch(batch_size)

    loss, pretrain_loss, z = model(batch_data)

    optimizer_class = optimizers.get(args.optimizer)
    optimizer = optimizer_class(args.learning_rate)
    global_step = tf.train.get_or_create_global_step()

    if args.mode == 'train':
        train_op = optimizer.minimize(loss, global_step=global_step)
    elif args.mode == 'pretrain':
        train_op = optimizer.minimize(pretrain_loss, global_step=global_step)

    hooks = []
    num_steps = int(sampler.train_traj_num // batch_size * args.num_epochs)
    hooks.append(tf.train.StopAtStepHook(last_step=num_steps))
    tensor_to_log = {'step': global_step, 'loss': loss, 'total_steps': tf.constant(num_steps)}
    hooks.append(tf.train.LoggingTensorHook(tensor_to_log, every_n_iter=args.log_steps))

    ckpt_dir = checkpoint_dir(args)
    print("checkpoint dir: {}".format(ckpt_dir))
    hooks.append(tf.train.CheckpointSaverHook(
        checkpoint_dir=ckpt_dir, save_steps=500, saver=tf.train.Saver(max_to_keep=1)))

    with tf.train.MonitoredTrainingSession(
            master=master, is_chief=is_chief, checkpoint_dir=ckpt_dir,
            log_step_count_steps=None, hooks=hooks, config=config) as sess:
        while not sess.should_stop():
            sess.run(train_op)

    if args.mode == 'pretrain':
        sample_num = 10000
        num_sample_steps = int(sample_num // batch_size)
        x_embedded = []
        with tf.train.MonitoredTrainingSession(
                master=master, is_chief=is_chief, checkpoint_dir=ckpt_dir,
                save_checkpoint_secs=None, log_step_count_steps=None, config=config) as sess:
            for _ in range(num_sample_steps):
                x_embedded.append(sess.run(z))
        x_embedded = np.concatenate(x_embedded, axis=0)[:sample_num]
        print("embedded shape:", x_embedded.shape)
        kmeans = KMeans(n_clusters=args.cluster_num)
        kmeans.fit(x_embedded)
        np.savez("{}/init_mu_c".format(ckpt_dir), kmeans.cluster_centers_)


def collect_scores(model, args, sampler, eval_data):
    traj_num = sampler.train_traj_num if eval_data == 'train' else sampler.val_traj_num
    batch_size = args.batch_size

    dataset = tf.data.Dataset.from_generator(
        sampler.data_iterator[eval_data],
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([batch_size, None]), tf.TensorShape([batch_size, None]), tf.TensorShape([batch_size])),
    )
    source = dataset.make_one_shot_iterator().get_next()
    scores = model(source)

    ckpt_dir = checkpoint_dir(args)
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if not ckpt_path:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    num_batches = int(math.ceil(traj_num / float(batch_size)))
    score_vals = []
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_path)
        for _ in range(num_batches):
            score_vals.append(sess.run(scores))

    score_vals = np.concatenate(score_vals, axis=-1)[:traj_num]
    return score_vals


def run_evaluate(model, args):
    def auc_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    sampler = DataGenerator(args)
    eval_data = args.eval_data
    sd_tids = sampler.train_sd if eval_data == 'train' else sampler.val_sd
    traj_num = sampler.train_traj_num if eval_data == 'train' else sampler.val_traj_num

    if args.anomaly_path:
        sampler.load_anomaly_manifest(args.anomaly_path, eval_data)
    else:
        sampler.inject_outliers(
            otype=args.otype, data_type=eval_data, ratio=args.outlier_ratio,
            level=3, point_prob=0.3, vary=False,
            stay_len=args.stay_len, speed_skip_ratio=args.speed_skip_ratio,
        )

    score_vals = collect_scores(model, args, sampler, eval_data)

    y_true = np.ones_like(score_vals)
    for idx in sampler.outlier_idx:
        if idx < y_true.shape[0]:
            y_true[idx] = 0

    sd_auc = []
    for tids in sd_tids.values():
        if np.sum(y_true[tids]) > 0:
            sd_auc.append(auc_score(y_true=y_true[tids], y_score=score_vals[tids]))
    print("Average AUC:", float(np.mean(sd_auc)))


def run_score(model, args):
    """Score trajectories and write CSV (for user-generated stay/speed anomalies)."""
    sampler = DataGenerator(args)
    eval_data = args.eval_data
    traj_num = sampler.train_traj_num if eval_data == 'train' else sampler.val_traj_num

    meta_by_tid = {}
    if args.anomaly_path:
        meta = sampler.load_anomaly_manifest(args.anomaly_path, eval_data)
        meta_by_tid = {m['tid']: m for m in meta}
    else:
        sampler.inject_outliers(
            otype=args.otype, data_type=eval_data, ratio=args.outlier_ratio,
            stay_len=args.stay_len, speed_skip_ratio=args.speed_skip_ratio,
        )
        for idx in sampler.outlier_idx:
            meta_by_tid[idx] = {'tid': idx, 'label': 0, 'type': args.otype}

    score_vals = collect_scores(model, args, sampler, eval_data)

    os.makedirs(os.path.dirname(args.output_scores) or '.', exist_ok=True)
    with open(args.output_scores, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['tid', 'label', 'anomaly_type', 'score'])
        writer.writeheader()
        for tid in range(traj_num):
            info = meta_by_tid.get(tid, {'label': 1, 'type': 'normal'})
            writer.writerow({
                'tid': tid,
                'label': info.get('label', 1),
                'anomaly_type': info.get('type', 'normal'),
                'score': float(score_vals[tid]),
            })

    print("Wrote scores for {} trajectories to {}.".format(traj_num, args.output_scores))


if __name__ == '__main__':
    args = define_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model = Model(args)
    if args.mode in ('train', 'pretrain'):
        run_train(model, args)
    elif args.mode == 'eval':
        run_evaluate(model, args)
    elif args.mode == 'score':
        run_score(model, args)
    else:
        raise ValueError("mode must be one of: pretrain, train, eval, score")
