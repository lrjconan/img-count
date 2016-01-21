"""
This code implements Recurrent Instance Segmentation [1].

[1] B. Romera-Paredes, P. Torr. Recurrent Instance Segmentation. arXiv preprint
arXiv:1511.08250, 2015.
"""

import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')

from data_api import mnist
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
from utils.time_series_logger import TimeSeriesLogger
import argparse
import datetime
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time

log = logger.get()


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _add_conv_lstm(inp_height, inp_width, hid_depth):

    w_xi = weight_variable([])
    def unroll(inp, time):
        pass

    return unroll


def get_model(opt, device='/cpu:0', train=True):
    """Get model."""
    inp_height = opt['']
    pass


def save_ckpt(folder, sess, opt, global_step=None):
    """Save checkpoint.

    Args:
        folder:
        sess:
        global_step:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    ckpt_path = os.path.join(
        folder, 'model.ckpt'.format(model_id))
    log.info('Saving checkpoint to {}'.format(ckpt_path))
    saver.save(sess, ckpt_path, global_step=global_step)
    opt_path = os.path.join(folder, 'opt.pkl')
    with open(opt_path, 'wb') as f_opt:
        pkl.dump(opt, f_opt)

    pass


def parse_args():
    """Parse input arguments."""
    # Number of steps
    kNumSteps = 500000
    # Number of steps per checkpoint
    kStepsPerCkpt = 1000
    parser = argparse.ArgumentParser(
        description='Train DRAW')
    parser.add_argument('-num_steps', default=kNumSteps,
                        type=int, help='Number of steps to train')
    parser.add_argument('-steps_per_ckpt', default=kStepsPerCkpt,
                        type=int, help='Number of steps per checkpoint')
    parser.add_argument('-results', default='../results',
                        help='Model results folder')
    parser.add_argument('-logs', default='../results',
                        help='Training curve logs folder')
    parser.add_argument('-localhost', default='localhost',
                        help='Local domain name')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    parser.add_argument('-seed', default=100, type=int,
                        help='Training seed')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'

    opt = {
    }

    # Train loop options
    loop_config = {
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt
    }

    # dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)

    m = get_model(opt, device=device)
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    task_name = 'draw_mnist'
    time_obj = datetime.datetime.now()
    model_id = timestr = '{}-{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
        task_name, time_obj.year, time_obj.month, time_obj.day,
        time_obj.hour, time_obj.minute, time_obj.second)
    results_folder = args.results
    logs_folder = args.logs
    exp_folder = os.path.join(results_folder, model_id)
    exp_logs_folder = os.path.join(logs_folder, model_id)

    # Create time series logger
    train_ce_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'train_ce.csv'), 'train_ce',
        buffer_size=25)
    valid_ce_logger = TimeSeriesLogger(
        os.path.join(exp_logs_folder, 'valid_ce.csv'), 'valid_ce',
        buffer_size=2)
    log.info(
        'Curves can be viewed at: http://{}/visualizer?id={}'.format(
            args.localhost, model_id))

    random = np.random.RandomState(args.seed)

    step = 0
    while step < loop_config['num_steps']:
        # Validation
        valid_ce = 0
        log.info('Running validation')
        for ii in xrange(100):
            batch = dataset.test.next_batch(100)
            x = preprocess(batch[0], opt)
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['hid_dim']])
            ce = sess.run(m['ce'], feed_dict={
                m['x']: x,
                m['u']: u
            })
            valid_ce += ce * 100 / 10000.0
        log.info('step {:d}, valid ce {:.4f}'.format(step, valid_ce))
        valid_ce_logger.add(step, valid_ce)

        # Train
        for ii in xrange(500):
            batch = dataset.train.next_batch(100)
            x = preprocess(batch[0], opt)
            u = random.normal(
                0, 1, [x.shape[0], opt['timespan'], opt['hid_dim']])
            r = sess.run([m['ce'], m['train_step']], feed_dict={
                m['x']: x,
                m['u']: u
            })
            if step % 10 == 0:
                ce = r[0]
                log.info('{:d} train ce {:.4f} t {:.2f}ms'.format(
                    step, ce, (time.time() - st) * 1000))
                train_ce_logger.add(step, ce)

            step += 1

            # Save model
            if step % args.steps_per_ckpt == 0:
                save_ckpt(exp_folder, sess, opt, global_step=step)

    sess.close()
    pass
